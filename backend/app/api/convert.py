"""
FastAPI endpoint for full document conversion pipeline.
Handles image upload, processing, and Word document generation.
Supports single and multiple image uploads.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import os
import logging
from typing import Optional, List, Dict
import traceback
import cv2
import numpy as np

from app.services.preprocessing import preprocess_image
from app.services.layout import detect_layout
from app.services.text_ocr import TextOCR
from app.services.math_ocr import MathOCR
from app.services.gemini_ocr import get_gemini_cleanup
from app.services.qwen_vl_ocr import get_qwen_vl_ocr  # Primary OCR (Qwen-VL-OCR)
from app.services.text_fusion import TextFusion
from app.services.document_builder import DocumentBuilder
from app.services.docx_generator import DOCXGenerator
from app.utils.file_manager import FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================================
# CONFIGURATION FLAGS
# ============================================================
# Use Qwen-VL-OCR as PRIMARY and EXCLUSIVE OCR engine
# NO FALLBACK: If Qwen-VL-OCR fails, processing HALTS with error
USE_QWEN_VL_OCR = True  # Primary and exclusive OCR - no silent fallback

# Initialize services (singleton pattern)
_text_ocr = None
_math_ocr = None
_file_manager = None


def get_text_ocr() -> TextOCR:
    """Get or create TextOCR instance."""
    global _text_ocr
    if _text_ocr is None:
        _text_ocr = TextOCR()
    return _text_ocr


def get_math_ocr() -> MathOCR:
    """Get or create MathOCR instance."""
    global _math_ocr
    if _math_ocr is None:
        _math_ocr = MathOCR()
    return _math_ocr


def get_file_manager() -> FileManager:
    """Get or create FileManager instance."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager


def _generate_diagram_description(region: Dict, image: np.ndarray) -> str:
    """
    Generate a short text description for a diagram region.
    
    CRITICAL: This is LAST RESORT - only used when OCR returns empty string.
    If OCR returns even 1 character, that text is used instead of this placeholder.
    
    Args:
        region: Diagram region dictionary with bbox and type
        image: Full processed image (numpy array)
        
    Returns:
        Text description like "[diagram showing flow of process]"
        ONLY used when OCR returns empty string
    """
    import numpy as np
    
    # Simple description based on region properties
    bbox = region.get('bbox', [])
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Determine diagram type based on aspect ratio and size
        if aspect_ratio > 1.5:
            # Wide diagram - likely flowchart or process diagram
            return "[diagram showing flow of process]"
        elif aspect_ratio < 0.7:
            # Tall diagram - likely vertical flow or tree
            return "[diagram showing hierarchical structure]"
        elif 0.8 < aspect_ratio < 1.2:
            # Square-ish diagram - likely chart or graph
            return "[diagram showing chart or graph]"
        else:
            # Generic diagram
            return "[diagram]"
    
    # Default fallback
    return "[diagram]"


async def process_single_image(
    image_file: UploadFile,
    image_index: int,
    total_images: int,
    uploads_dir: Path,
    outputs_dir: Path
) -> Optional[dict]:
    """
    Process a single image through the full pipeline.
    
    Args:
        image_file: Uploaded image file
        image_index: Index of current image (0-based)
        total_images: Total number of images
        uploads_dir: Directory for uploads
        outputs_dir: Directory for outputs
        
    Returns:
        Dictionary with structured_json and diagram_dir, or None if processing failed
    """
    temp_files = []
    
    try:
        logger.info(f"📸 Processing image {image_index + 1}/{total_images}: {image_file.filename}")
        logger.info(f"   File size: {len(await image_file.read()) / 1024:.2f} KB")
        await image_file.seek(0)  # Reset file pointer
        
        # Save uploaded file temporarily
        logger.debug("   Saving uploaded file temporarily...")
        temp_input = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=Path(image_file.filename or 'image').suffix or '.jpg', 
            dir=uploads_dir
        )
        # Read file content
        content = await image_file.read()
        temp_input.write(content)
        temp_input.close()
        temp_files.append(temp_input.name)
        logger.debug(f"   Saved to: {temp_input.name}")
        
        # Reset file pointer for potential retry
        await image_file.seek(0)
        
        input_path = temp_input.name
        base_filename = f"img_{image_index}_{Path(image_file.filename).stem}"
        
        # Preprocess image
        logger.info(f"   🔧 Step 1/5: Preprocessing image...")
        try:
            processed_image_path = preprocess_image(input_path)
            temp_files.append(processed_image_path)
            logger.info(f"   ✅ Image preprocessed: {Path(processed_image_path).name}")
        except Exception as e:
            logger.warning(f"   ❌ Image {image_index + 1} preprocessing failed: {e}")
            logger.info(f"   🔄 FALLBACK: Using original image without preprocessing...")
            # CRITICAL: Use original image if preprocessing fails - NEVER return None
            processed_image_path = input_path
        
        # Detect layout
        logger.info(f"   📐 Step 2/5: Detecting layout regions...")
        layout_regions = []
        detection_attempted = False
        detection_succeeded = False
        
        try:
            layout_regions = detect_layout(processed_image_path)
            detection_attempted = True
            if layout_regions:
                detection_succeeded = True
                logger.info(f"   ✅ Detected {len(layout_regions)} regions")
            else:
                logger.warning(f"   ⚠️  Image {image_index + 1}: No regions detected by layout detection")
                logger.info(f"   🔄 FALLBACK: Creating full-image region for OCR...")
                # CRITICAL: Create a single region covering the entire image
                # NEVER exit - always create full-image region
                h, w = processed_image.shape[:2]
                layout_regions = [{
                    'type': 'paragraph',
                    'bbox': [0, 0, w, h]
                }]
                logger.info(f"   ✅ Created full-image region for OCR (detection found 0 regions)")
        except Exception as e:
            logger.warning(f"   ❌ Image {image_index + 1} layout detection failed: {e}")
            logger.info(f"   🔄 FALLBACK: Creating full-image region for OCR...")
            # CRITICAL: Even if detection crashes, create full-image region
            # NEVER exit - always create full-image region
            h, w = processed_image.shape[:2]
            layout_regions = [{
                'type': 'paragraph',
                'bbox': [0, 0, w, h]
            }]
            logger.info(f"   ✅ Created full-image region for OCR (detection crashed)")
        
        # CRITICAL: Ensure we ALWAYS have at least one region
        # If somehow layout_regions is still empty, create full-image region
        if not layout_regions:
            logger.error(f"   🚨 CRITICAL: layout_regions is empty - creating full-image region")
            h, w = processed_image.shape[:2]
            layout_regions = [{
                'type': 'paragraph',
                'bbox': [0, 0, w, h]
            }]
        
        # THREE-LAYER SYSTEM IMPLEMENTATION
        logger.info("=" * 60)
        logger.info(f"   🏗️  THREE-LAYER EXTRACTION SYSTEM")
        logger.info("=" * 60)
        
        processed_image = cv2.imread(processed_image_path)
        if processed_image is None:
            logger.error(f"Image {image_index + 1}: Failed to load processed image")
            logger.info(f"   🔄 FALLBACK: Attempting to load original image...")
            # Try loading original image as fallback
            try:
                processed_image = cv2.imread(input_path)
                if processed_image is None:
                    logger.error(f"   ❌ Both processed and original images failed to load")
                    # Create empty document instead of returning None
                    return {
                        'structured_json': [{
                            'type': 'paragraph',
                            'text': '[OCR pipeline executed but no readable text was extracted]',
                            'source': 'image_load_failed'
                        }],
                        'diagram_dir': None,
                        'image_index': image_index
                    }
            except Exception as e:
                logger.error(f"   ❌ Fallback image load failed: {e}")
                # Create empty document instead of returning None
                return {
                    'structured_json': [{'type': 'paragraph', 'text': '', 'source': 'image_load_failed'}],
                    'diagram_dir': None,
                    'image_index': image_index
                }
        
        # ============================================================
        # OCR EXTRACTION - Qwen-VL-OCR (Primary and Exclusive)
        # NO FALLBACK: If Qwen-VL-OCR fails, processing HALTS
        # ============================================================
        logger.info(f"   🔍 Starting OCR extraction (Qwen-VL-OCR)...")
        final_text = ""
        diagram_regions = []
        structured_json = []
        use_qwen_ocr = False
        
        # PRIMARY: Qwen-VL-OCR (exclusive OCR engine)
        if USE_QWEN_VL_OCR:
            try:
                qwen_ocr = get_qwen_vl_ocr()
                if not qwen_ocr.available:
                    # CRITICAL: Qwen-VL-OCR is required - HALT processing
                    error_msg = qwen_ocr.initialization_error or "Qwen-VL-OCR is not available"
                    logger.error(f"   ❌ CRITICAL: {error_msg}")
                    raise RuntimeError(f"OCR HALTED: {error_msg}")
                
                logger.info(f"   🔮 Using Qwen-VL-OCR (primary and exclusive OCR)...")
                
                # Get image dimensions for diagram position estimation
                h, w = processed_image.shape[:2]
                
                # Extract text with diagram detection
                final_text, diagram_regions = qwen_ocr.extract_text_from_image(
                    processed_image_path,
                    image_width=w,
                    image_height=h
                )
                
                if final_text and final_text.strip():
                    use_qwen_ocr = True
                    logger.info(f"   ✅ Qwen-VL-OCR extracted {len(final_text)} characters")
                    if diagram_regions:
                        logger.info(f"   📊 Detected {len(diagram_regions)} diagram regions")
                else:
                    # Qwen-VL-OCR returned empty but no error - create fallback
                    logger.warning(f"   ⚠️  Qwen-VL-OCR extracted no text from image")
                    final_text = ""
                    
            except RuntimeError as e:
                # CRITICAL: Qwen-VL-OCR failure - HALT processing (no silent fallback)
                logger.error(f"   ❌ OCR HALTED: {e}")
                # Return structured error message instead of failing silently
                return {
                    'structured_json': [{
                        'type': 'paragraph',
                        'text': f'[OCR ERROR: {str(e)}]',
                        'source': 'qwen_ocr_error'
                    }],
                    'diagram_dir': None,
                    'image_index': image_index
                }
            except Exception as e:
                # CRITICAL: Unexpected error - HALT processing (no silent fallback)
                logger.error(f"   ❌ Unexpected OCR error: {e}")
                return {
                    'structured_json': [{
                        'type': 'paragraph',
                        'text': f'[OCR ERROR: Unexpected failure - {str(e)}]',
                        'source': 'qwen_ocr_error'
                    }],
                    'diagram_dir': None,
                    'image_index': image_index
                }
        else:
            # USE_QWEN_VL_OCR is False - this should not happen in production
            logger.error(f"   ❌ CRITICAL: USE_QWEN_VL_OCR is disabled - no OCR engine available")
            return {
                'structured_json': [{
                    'type': 'paragraph',
                    'text': '[OCR ERROR: No OCR engine enabled]',
                    'source': 'no_ocr_enabled'
                }],
                'diagram_dir': None,
                'image_index': image_index
            }
        
        # ============================================================
        # DOCUMENT STRUCTURE CREATION - Parse Extracted Text
        # ============================================================
        logger.info(f"   📄 Creating document structure...")
        
        if final_text and final_text.strip():
            # Parse the text into structured elements
            lines = final_text.split('\n')
            diagram_idx = 0
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                # Check for diagram placeholders
                if line_stripped.startswith('[[DIAGRAM') and ']]' in line_stripped:
                    # This is a diagram placeholder - crop and add diagram
                    logger.info(f"   🔍 Found diagram marker: {line_stripped}")
                    
                    if diagram_idx < len(diagram_regions):
                        region = diagram_regions[diagram_idx]
                        bbox = region.get('estimated_bbox')
                        logger.info(f"   📊 Processing diagram {diagram_idx}: bbox={bbox}")
                        
                        if bbox and bbox.get('x2') and bbox.get('y2'):
                            # Crop diagram region from original image
                            try:
                                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                                h, w = processed_image.shape[:2]
                                # Ensure bounds are valid
                                x1 = max(0, min(x1, w))
                                x2 = max(0, min(x2, w))
                                y1 = max(0, min(y1, h))
                                y2 = max(0, min(y2, h))
                                
                                logger.info(f"   ✂️  Cropping diagram: x={x1}-{x2}, y={y1}-{y2} (Image: {w}x{h})")
                                
                                if x2 > x1 and y2 > y1:
                                    cropped_diagram = processed_image[y1:y2, x1:x2]
                                    diagram_filename = f"diagram_{image_index}_{diagram_idx}.png"
                                    diagram_path = outputs_dir / diagram_filename
                                    cv2.imwrite(str(diagram_path), cropped_diagram)
                                    
                                    structured_json.append({
                                        'type': 'diagram',
                                        'image_path': str(diagram_path)
                                    })
                                    logger.info(f"   📸 Cropped diagram {diagram_idx} saved to {diagram_path}")
                                else:
                                    logger.warning(f"   ⚠️  Invalid crop dimensions: {x2-x1}x{y2-y1}")
                                    structured_json.append({
                                        'type': 'paragraph',
                                        'text': '[Diagram - Invalid dimensions]'
                                    })
                            except Exception as e:
                                logger.warning(f"   ⚠️  Failed to crop diagram {diagram_idx}: {e}")
                                structured_json.append({
                                    'type': 'paragraph',
                                    'text': '[Diagram - Could not be extracted]'
                                })
                        else:
                            # No bbox available - just add placeholder
                            logger.warning(f"   ⚠️  No bbox for diagram {diagram_idx}")
                            structured_json.append({
                                'type': 'paragraph',
                                'text': '[Diagram]'
                            })
                    else:
                        logger.warning(f"   ⚠️  Diagram marker found but no corresponding region (idx={diagram_idx}, regions={len(diagram_regions)})")
                        structured_json.append({
                            'type': 'paragraph',
                            'text': '[Diagram]'
                        })
                        
                    diagram_idx += 1
                    continue
                
                # Check for headings (## or ### markers from Gemini)
                if line_stripped.startswith('## '):
                    structured_json.append({
                        'type': 'heading',
                        'text': line_stripped[3:].strip()
                    })
                elif line_stripped.startswith('### '):
                    structured_json.append({
                        'type': 'heading',
                        'text': line_stripped[4:].strip()
                    })
                # Check for equations ($ markers)
                elif line_stripped.startswith('$$') and line_stripped.endswith('$$'):
                    structured_json.append({
                        'type': 'equation',
                        'latex': line_stripped[2:-2].strip()
                    })
                else:
                    # Regular paragraph
                    structured_json.append({
                        'type': 'paragraph',
                        'text': line_stripped,
                        'source': 'qwen_ocr' if use_qwen_ocr else 'ocr'
                    })
            
            logger.info(f"   ✅ Created {len(structured_json)} elements from text")
        
        # CRITICAL: If final_text is empty, create fallback paragraph
        # ABSOLUTE RULE: Blank documents are FORBIDDEN
        if not structured_json:
            logger.error(f"   🚨 CRITICAL: final_text is empty - creating fallback paragraph")
            structured_json = [{
                'type': 'paragraph',
                'text': '[OCR pipeline executed but no readable text was extracted]',
                'source': 'mandatory_fallback'
            }]
            
        # ============================================================
        # DIAGRAM FALLBACK: Check if specialized diagram extractor found diagrams that Qwen missed
        # ============================================================
        from app.services.diagram_extractor import diagram_extractor
        
        qwen_diagram_count = sum(1 for e in structured_json if e.get('type') == 'diagram')
        
        # Run dedicated diagram extraction
        logger.info(f"   🎨 Running dedicated diagram extraction...")
        detected_diagrams = diagram_extractor.extract_diagrams(str(processed_image_path))
        
        if qwen_diagram_count == 0 and detected_diagrams:
            logger.info(f"   ⚠️  Qwen missed {len(detected_diagrams)} diagrams detected by dedicated extractor - inserting as fallback")
            
            for i, region in enumerate(detected_diagrams):
                try:
                    bbox = region.get('bbox') # [x1, y1, x2, y2]
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        h, w = processed_image.shape[:2]
                        
                        # Ensure bounds are valid
                        x1 = max(0, min(x1, w))
                        x2 = max(0, min(x2, w))
                        y1 = max(0, min(y1, h))
                        y2 = max(0, min(y2, h))
                        
                        if x2 > x1 and y2 > y1:
                            cropped_diagram = processed_image[y1:y2, x1:x2]
                            diagram_filename = f"diagram_extracted_{image_index}_{i}.png"
                            diagram_path = outputs_dir / diagram_filename
                            cv2.imwrite(str(diagram_path), cropped_diagram)
                            
                            # Calculate insertion position based on vertical location
                            # This places the diagram roughly where it appears in the original image
                            y_center = (y1 + y2) / 2
                            relative_pos = y_center / h if h > 0 else 1.0
                            insert_idx = int(relative_pos * len(structured_json))
                            # Ensure index is within bounds
                            insert_idx = max(0, min(insert_idx, len(structured_json)))
                            
                            diagram_element = {
                                'type': 'diagram',
                                'image_path': str(diagram_path)
                            }
                            
                            structured_json.insert(insert_idx, diagram_element)
                            logger.info(f"   📸 Fallback: Cropped diagram {i} saved to {diagram_path} (inserted at {insert_idx}/{len(structured_json)})")
                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to crop fallback diagram {i}: {e}")
        
        # Build final document
        logger.info(f"   📄 Step 4/5: Building structured document...")
        try:
            # Use fused elements directly (already in correct format)
            logger.info(f"   ✅ Document structure ready with {len(structured_json)} elements")
            
            return {
                'structured_json': structured_json,
                'diagram_dir': None,  # No diagram directory - no images extracted
                'image_index': image_index
            }
        except Exception as e:
            logger.error(f"Image {image_index + 1} document building failed: {e}")
            logger.info(f"   🔄 FALLBACK: Creating fallback document...")
            # CRITICAL: NEVER return None - create document with fallback text instead
            # ABSOLUTE RULE: Blank documents are FORBIDDEN
            return {
                'structured_json': [{
                    'type': 'paragraph',
                    'text': '[OCR pipeline executed but no readable text was extracted]',
                    'source': 'build_failed'
                }],
                'diagram_dir': None,
                'image_index': image_index
            }
        
    except Exception as e:
        logger.error(f"Image {image_index + 1} processing failed: {e}")
        logger.info(f"   🔄 FALLBACK: Creating empty document instead of skipping...")
        # CRITICAL: NEVER return None - always return a document
        # Even if processing completely fails, return empty document
        return {
            'structured_json': [{'type': 'paragraph', 'text': '', 'source': 'processing_failed'}],
            'diagram_dir': None,
            'image_index': image_index
        }
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass


@router.post("/convert")
async def convert_image_to_word(images: List[UploadFile] = File(...)):
    """
    Convert handwritten notes images to editable Word document.
    Supports multiple images - all merged into one Word file.
    
    Full pipeline per image:
    1. Preprocess image
    2. Detect layout
    3. Extract text (OCR)
    4. Recognize equations (Math OCR)
    5. Extract diagrams
    6. Build structured document
    
    Final step:
    7. Merge all results and generate Word file
    
    Args:
        images: One or more uploaded image files
        
    Returns:
        Word document file with all images merged
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    all_structured_json = []
    outputs_dir = None
    
    try:
        logger.info("=" * 60)
        logger.info(f"🚀 Starting conversion for {len(images)} image(s)")
        logger.info("=" * 60)
        
        # Get file manager
        file_manager = get_file_manager()
        uploads_dir = file_manager.get_uploads_dir()
        outputs_dir = file_manager.get_outputs_dir()
        
        # Process each image
        successful_images = 0
        for idx, image_file in enumerate(images):
            try:
                # Reset file pointer (in case it was read before)
                await image_file.seek(0)
                
                result = await process_single_image(
                    image_file,
                    idx,
                    len(images),
                    uploads_dir,
                    outputs_dir
                )
                
                if result:
                    structured_json = result['structured_json']
                    
                    # Add page break marker before each new image (except first)
                    if idx > 0 and structured_json:
                        all_structured_json.append({"type": "page_break"})
                    
                    # Add all elements from this image
                    all_structured_json.extend(structured_json)
                    successful_images += 1
                
            except Exception as e:
                logger.warning(f"Failed to process image {idx + 1}: {e}, continuing with others")
                continue
        
        # CRITICAL: NEVER raise HTTPException for "No content extracted"
        # Even if all images failed, return a document with fallback text instead of error
        # ABSOLUTE RULE: Blank documents are FORBIDDEN
        if not all_structured_json:
            logger.warning(f"⚠️  No content extracted from any image - creating fallback document")
            logger.error(f"🚨 CRITICAL: All images produced empty output - adding mandatory fallback paragraph")
            # Create document with fallback text - this is better than a blank document
            all_structured_json = [{
                'type': 'paragraph',
                'text': '[OCR pipeline executed but no readable text was extracted]',
                'source': 'empty_fallback'
            }]
        
        if successful_images == 0:
            logger.warning(f"⚠️  All images failed to process - creating fallback document")
            logger.error(f"🚨 CRITICAL: All images failed - adding mandatory fallback paragraph")
            # ABSOLUTE RULE: Blank documents are FORBIDDEN
            if not all_structured_json:
                all_structured_json = [{
                    'type': 'paragraph',
                    'text': '[OCR pipeline executed but no readable text was extracted]',
                    'source': 'empty_fallback'
                }]
        
        logger.info("=" * 60)
        logger.info(f"✅ Successfully processed {successful_images}/{len(images)} images")
        logger.info(f"📊 Total elements in merged document: {len(all_structured_json)}")
        logger.info("=" * 60)
        
        # Generate Word document from merged results
        logger.info("📝 Step 5/5: Generating merged Word document...")
        try:
            base_filename = f"merged_{len(images)}_images"
            output_filename = f"{base_filename}_converted.docx"
            output_path = outputs_dir / output_filename
            
            docx_generator = DOCXGenerator()
            # CRITICAL: diagram_dir is ignored - no images are inserted into documents
            # Only text extracted via OCR is included
            # Document is validated to ensure NO images, shapes, or drawings
            docx_generator.generate_document(
                structured_json=all_structured_json,
                diagram_dir=None,  # No images should be inserted
                output_path=output_path
            )
            
            logger.info(f"Word document generated: {output_path}")
            logger.info("Document validated: Text-only, no images, shapes, or drawings")
        except Exception as e:
            logger.error(f"Word generation failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Word document generation failed: {str(e)}"
            )
        
        # Return Word document
        if not output_path or not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Word document was not generated successfully"
            )
        
        logger.info(f"Conversion complete. Returning file: {output_path.name}")
        return FileResponse(
            str(output_path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=output_path.name,
            headers={"Content-Disposition": f'attachment; filename="{output_path.name}"'}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )
