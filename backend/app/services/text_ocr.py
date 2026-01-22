"""
Handwritten OCR module using HYBRID OCR + C-RNN pipeline.

ARCHITECTURE:
- OCR Engine (PaddleOCR DB) → Text detection and line segmentation ONLY
- C-RNN (PaddleOCR CRNN) → Handwritten text recognition at LINE level
- CTC decoding for sequence-level reading

CRITICAL RULES:
- NEVER discard text due to low confidence
- Process at LINE level, not word level
- Preserve raw OCR output
- Text presence > Text perfection
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class TextOCR:
    """
    HYBRID OCR engine for handwritten text recognition.
    
    Pipeline:
    1. Light preprocessing (grayscale, no binarization)
    2. Text detection (PaddleOCR DB detector)
    3. Line segmentation (extract full lines, not words)
    4. C-RNN recognition (PaddleOCR CRNN with CTC decoding)
    5. Preserve all output regardless of confidence
    """
    
    def __init__(self):
        """Initialize OCR models."""
        logger.info("🔧 Initializing HYBRID OCR + C-RNN pipeline...")
        self.device = 'cuda' if self._check_cuda() else 'cpu'
        logger.info(f"💻 Using device: {self.device.upper()}")
        
        # Initialize PaddleOCR with HYBRID configuration
        # Detection: DB (Differentiable Binarization)
        # Recognition: CRNN (CNN + RNN + CTC) - optimized for handwriting
        logger.info("📦 Loading PaddleOCR HYBRID pipeline (DB detection + CRNN recognition)...")
        self.paddleocr_available = False
        self.paddleocr_detector = None
        self.paddleocr_recognizer = None
        
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR for HYBRID pipeline
            # We'll use the same instance but call it with det=True/rec=False for detection
            # and det=False/rec=True for recognition
            logger.info("   → Initializing PaddleOCR HYBRID pipeline...")
            self.paddleocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self.device == 'cuda',
                show_log=False,
                det_algorithm='DB',  # DB detection
                det_db_thresh=0.3,  # Lower threshold for handwriting
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6,
                det_limit_side_len=960,
                det_limit_type='max',
                rec_algorithm='CRNN',  # CRNN recognition (C-RNN)
                use_dilation=False,  # Preserve character shapes
            )
            
            # Use the same reader for both detection and recognition
            # We'll control it via the ocr() method parameters
            self.paddleocr_detector = self.paddleocr_reader
            self.paddleocr_recognizer = self.paddleocr_reader
            
            self.paddleocr_available = True
            logger.info(f"✅ HYBRID pipeline initialized successfully on {self.device.upper()}")
            logger.info("   Architecture: DB Detection → Line Segmentation → CRNN Recognition")
            
        except ImportError:
            logger.warning("⚠️  PaddleOCR not available (install with: pip install paddlepaddle paddleocr)")
            self.paddleocr_detector = None
            self.paddleocr_recognizer = None
        except Exception as e:
            logger.warning(f"⚠️  PaddleOCR initialization failed: {e}")
            self.paddleocr_detector = None
            self.paddleocr_recognizer = None
        
        # Fallback: EasyOCR (if PaddleOCR not available)
        logger.info("📦 Loading EasyOCR fallback...")
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(
                ['en'],
                gpu=self.device == 'cuda',
                verbose=False
            )
            logger.info(f"✅ EasyOCR fallback available on {self.device.upper()}")
            self.easyocr_available = True
        except Exception as e:
            logger.warning(f"⚠️  EasyOCR initialization failed: {e}")
            self.easyocr_reader = None
            self.easyocr_available = False
        
        if self.paddleocr_available:
            logger.info("✅ HYBRID OCR + C-RNN pipeline ready")
        elif self.easyocr_available:
            logger.warning("⚠️  Using EasyOCR fallback (HYBRID pipeline not available)")
        else:
            logger.error("❌ No OCR engines available! Please check installation.")
    
    def extract_text(
        self,
        image: np.ndarray,
        region_type: str = 'paragraph'
    ) -> Tuple[str, float]:
        """
        Extract text using HYBRID OCR + C-RNN pipeline.
        
        Pipeline:
        1. Light preprocessing (grayscale, no binarization)
        2. Text detection (DB detector) → find text regions
        3. Line segmentation → extract full lines
        4. C-RNN recognition → recognize each line with CTC decoding
        5. Combine lines preserving order
        
        CRITICAL RULES:
        - NEVER discard text due to low confidence
        - Process at LINE level (not word level)
        - Preserve raw OCR output
        - Return ALL recognized text, even if low confidence
        
        Args:
            image: Cropped image region (numpy array)
            region_type: Type of region ('heading', 'paragraph', 'equation', 'diagram')
            
        Returns:
            Tuple of (extracted_text, confidence_score)
            - Always returns actual OCR text (never empty unless no text detected)
            - Confidence is for logging only, NEVER used to filter
        """
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            logger.warning(f"🚨 [OCR DEBUG] Invalid image type: {type(image)}")
            return "", 0.0
        
        # Light preprocessing ONLY (no binarization, no thresholding)
        processed_image = self._light_preprocess(image, region_type)
        
        # Try HYBRID pipeline first (DB detection + CRNN recognition)
        if self.paddleocr_available and self.paddleocr_detector and self.paddleocr_recognizer:
            try:
                text, confidence = self._extract_with_hybrid_pipeline(processed_image, region_type)
                # CRITICAL: If ANY text detected, use it (regardless of confidence)
                if text and text.strip():  # Check for non-empty text
                    logger.info(f"✅ [OCR DEBUG] HYBRID pipeline result: text='{text}' (len={len(text)}), confidence={confidence:.4f}")
                    return text, confidence
                else:
                    logger.info(f"⚠️  [OCR DEBUG] HYBRID pipeline returned empty, trying fallback...")
            except Exception as e:
                logger.warning(f"⚠️  [OCR DEBUG] HYBRID pipeline error: {e}, trying fallback...")
        
        # Fallback: EasyOCR (if HYBRID pipeline fails or unavailable)
        if self.easyocr_available:
            try:
                text, confidence = self._extract_with_easyocr(processed_image, region_type)
                if text:  # If ANY text detected, use it
                    logger.info(f"✅ [OCR DEBUG] EasyOCR fallback result: text='{text}' (len={len(text)}), confidence={confidence:.4f}")
                    return text, confidence
            except Exception as e:
                logger.warning(f"⚠️  [OCR DEBUG] EasyOCR fallback error: {e}")
        
        # CRITICAL: If ALL OCR methods failed, run full-image OCR before giving up
        # OCR MUST always attempt full-image extraction before giving up
        logger.warning(f"⚠️  [OCR DEBUG] All OCR methods failed - attempting full-image OCR")
        logger.info(f"   🔄 Running full-image OCR (final fallback)...")
        return self._run_full_image_ocr(processed_image)
    
    def _extract_with_hybrid_pipeline(
        self,
        image: np.ndarray,
        region_type: str
    ) -> Tuple[str, float]:
        """
        HYBRID pipeline: DB Detection → Line Segmentation → CRNN Recognition.
        
        This is the core of the handwriting recognition system.
        Processes at LINE level for better context-aware recognition.
        """
        logger.info(f"🔍 [OCR DEBUG] Starting HYBRID pipeline (region_type: {region_type})")
        
        # Step 1: Text detection using DB detector
        # This finds text regions and segments them into lines
        logger.debug("   → Step 1: DB Detection (finding text lines)...")
        try:
            # Use detector to find text boxes (these are typically lines)
            # det=True, rec=False means detection only, no recognition
            det_result = self.paddleocr_detector.ocr(image, det=True, rec=False, cls=False)
            
            # CRITICAL: Fix numpy array comparison issue
            # Check if det_result is None or empty
            # CRITICAL: If no regions detected, DO NOT FAIL - run full-image OCR
            if det_result is None:
                logger.warning("   ⚠️  DB detector returned None - falling back to full-image OCR")
                logger.info("   🔄 Running full-image OCR (detection returned None)...")
                # Run full-image OCR instead of returning empty
                return self._run_full_image_ocr(image)
            
            # Check if first page result exists and is not empty
            # Use len() and try-except to avoid numpy array ambiguity
            try:
                if len(det_result) == 0:
                    logger.warning("   ⚠️  DB detector returned empty result - falling back to full-image OCR")
                    logger.info("   🔄 Running full-image OCR (detection returned empty)...")
                    # Run full-image OCR instead of returning empty
                    return self._run_full_image_ocr(image)
                
                first_page = det_result[0]
                # Check if first_page is None or empty list
                if first_page is None:
                    logger.warning("   ⚠️  DB detector first_page is None - falling back to full-image OCR")
                    logger.info("   🔄 Running full-image OCR (first_page is None)...")
                    # Run full-image OCR instead of returning empty
                    return self._run_full_image_ocr(image)
                
                # Use len() to check if list/tuple is empty (avoids numpy array boolean ambiguity)
                # CRITICAL: Check for NumPy array first, then list/tuple
                if hasattr(first_page, 'shape'):
                    # NumPy array - check size explicitly
                    if first_page.size == 0:
                        logger.warning("   ⚠️  DB detector first_page is empty NumPy array - falling back to full-image OCR")
                        logger.info("   🔄 Running full-image OCR (first_page is empty)...")
                        # Run full-image OCR instead of returning empty
                        return self._run_full_image_ocr(image)
                elif isinstance(first_page, (list, tuple)) and len(first_page) == 0:
                    logger.warning("   ⚠️  DB detector first_page is empty list/tuple - falling back to full-image OCR")
                    logger.info("   🔄 Running full-image OCR (first_page is empty)...")
                    # Run full-image OCR instead of returning empty
                    return self._run_full_image_ocr(image)
            except (IndexError, TypeError) as e:
                logger.warning(f"   ⚠️  DB detector invalid result structure: {e} - falling back to full-image OCR")
                logger.info("   🔄 Running full-image OCR (invalid result structure)...")
                # Run full-image OCR instead of returning empty
                return self._run_full_image_ocr(image)
            
            # Extract line boxes from detection results
            # det_result structure: [[[box_coords], ...], ...]
            # CRITICAL: Use explicit checks - never use NumPy arrays in boolean context
            line_boxes = []
            for page_result in det_result:
                # Explicit check: page_result is not None and has elements
                if page_result is not None:
                    # Check if it's a list/tuple with length > 0, or NumPy array with size > 0
                    if isinstance(page_result, (list, tuple)):
                        if len(page_result) > 0:
                            for line_info in page_result:
                                # Explicit check: line_info is not None and has elements
                                if line_info is not None:
                                    if isinstance(line_info, (list, tuple)) and len(line_info) > 0:
                                        # line_info[0] contains box coordinates (4 points)
                                        box = line_info[0]
                                        # Explicit check: box is not None and has at least 4 points
                                        if box is not None:
                                            # Check if box is list/tuple with len >= 4, or NumPy array with shape[0] >= 4
                                            if isinstance(box, (list, tuple)):
                                                if len(box) >= 4:
                                                    line_boxes.append(box)
                                            elif hasattr(box, 'shape'):  # NumPy array
                                                if box.shape[0] >= 4:
                                                    line_boxes.append(box)
                                    elif hasattr(line_info, 'shape'):  # NumPy array
                                        # If line_info is a NumPy array, check its shape
                                        if line_info.size > 0:
                                            box = line_info[0] if line_info.shape[0] > 0 else None
                                            if box is not None:
                                                if hasattr(box, 'shape') and box.shape[0] >= 4:
                                                    line_boxes.append(box)
                                                elif isinstance(box, (list, tuple)) and len(box) >= 4:
                                                    line_boxes.append(box)
                    elif hasattr(page_result, 'shape'):  # NumPy array
                        # If page_result is a NumPy array, iterate differently
                        if page_result.size > 0:
                            for i in range(page_result.shape[0]):
                                line_info = page_result[i]
                                if line_info is not None:
                                    if hasattr(line_info, 'shape') and line_info.size > 0:
                                        box = line_info[0] if line_info.shape[0] > 0 else None
                                        if box is not None:
                                            if hasattr(box, 'shape') and box.shape[0] >= 4:
                                                line_boxes.append(box)
                                            elif isinstance(box, (list, tuple)) and len(box) >= 4:
                                                line_boxes.append(box)
            
            # Explicit check: line_boxes is a list, use len() not boolean
            # CRITICAL: If no boxes detected, DO NOT FAIL - run full-image OCR
            if len(line_boxes) == 0:
                logger.warning("   ⚠️  No regions detected by DB detector - falling back to full-image OCR")
                logger.info("   🔄 Running full-image OCR (no regions detected)...")
                # CRITICAL: Run OCR on full image instead of returning empty
                # This ensures we ALWAYS attempt text extraction
                try:
                    # Use recognition-only mode on full image
                    full_image_result = self.paddleocr_recognizer.ocr(image, det=False, rec=True, cls=True)
                    if full_image_result is not None:
                        try:
                            first_rec = full_image_result[0]
                            if first_rec is not None:
                                # Extract text from full-image recognition
                                recognized_texts = []
                                if isinstance(first_rec, (list, tuple)) and len(first_rec) > 0:
                                    for rec_item in first_rec:
                                        if rec_item is not None and isinstance(rec_item, (list, tuple)) and len(rec_item) > 1:
                                            text_info = rec_item[1]
                                            if isinstance(text_info, tuple) and len(text_info) >= 1:
                                                text = text_info[0]
                                                if isinstance(text, str) and text.strip():
                                                    recognized_texts.append(text.strip())
                                elif hasattr(first_rec, 'shape') and first_rec.size > 0:
                                    # Handle NumPy array structure
                                    for i in range(first_rec.shape[0]):
                                        rec_item = first_rec[i]
                                        if rec_item is not None:
                                            try:
                                                if hasattr(rec_item, 'shape') and rec_item.size > 1:
                                                    if rec_item.shape[0] > 1:
                                                        text_info = rec_item[1]
                                                        if isinstance(text_info, (str, tuple)):
                                                            if isinstance(text_info, tuple) and len(text_info) >= 1:
                                                                text = text_info[0]
                                                                if isinstance(text, str) and text.strip():
                                                                    recognized_texts.append(text.strip())
                                            except (IndexError, TypeError, AttributeError):
                                                continue
                                
                                if recognized_texts:
                                    combined_text = '\n'.join(recognized_texts)
                                    logger.info(f"   ✅ Full-image OCR extracted {len(combined_text)} characters")
                                    return combined_text, 0.5  # Lower confidence for full-image OCR
                        except (IndexError, TypeError, AttributeError) as e:
                            logger.debug(f"   → Full-image OCR result processing error: {e}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Full-image OCR fallback error: {e}")
                
                # If full-image OCR also fails, return empty (caller will handle further fallback)
                logger.warning("   ⚠️  Full-image OCR also returned empty")
                return "", 0.0
            
            logger.info(f"   → Detected {len(line_boxes)} text lines")
            
        except Exception as e:
            # CRITICAL: Catch ALL OCR exceptions and continue
            # DO NOT FAIL - run full-image OCR instead
            logger.warning(f"   ⚠️  DB Detection error: {e} - falling back to full-image OCR")
            logger.info("   🔄 Running full-image OCR (detection error)...")
            import traceback
            logger.debug(f"   → Traceback: {traceback.format_exc()}")
            # Run full-image OCR instead of returning empty
            return self._run_full_image_ocr(image)
        
        # Step 2: Extract line images and recognize with CRNN
        logger.debug("   → Step 2: Line segmentation and CRNN recognition...")
        recognized_lines = []
        total_confidence = 0.0
        valid_lines = 0
        
        for line_idx, box in enumerate(line_boxes):
            try:
                # Extract line image from box coordinates
                line_image = self._extract_line_image(image, box)
                if line_image is None or line_image.size == 0:
                    logger.debug(f"   → Line {line_idx}: Invalid line image, skipping")
                    continue
                
                # Step 3: Recognize line with CRNN (CTC decoding)
                logger.debug(f"   → Line {line_idx}: Recognizing with CRNN...")
                # det=False, rec=True means recognition only, no detection
                rec_result = self.paddleocr_recognizer.ocr(line_image, det=False, rec=True, cls=True)
                
                # CRITICAL: Use explicit checks - never use NumPy arrays in boolean context
                if rec_result is not None:
                    try:
                        first_rec = rec_result[0]
                        if first_rec is not None:
                            # Check if it's a list/tuple with length > 0, or NumPy array with size > 0
                            if isinstance(first_rec, (list, tuple)) and len(first_rec) > 0:
                                # Extract text and confidence from recognition result
                                # Structure: [[[box], (text, confidence)], ...]
                                for rec_line in first_rec:
                                    # Explicit check: rec_line is not None and has at least 2 elements
                                    if rec_line is not None:
                                        if isinstance(rec_line, (list, tuple)) and len(rec_line) > 1:
                                            text_info = rec_line[1]
                                            if isinstance(text_info, tuple) and len(text_info) >= 2:
                                                line_text = text_info[0]
                                                line_confidence = text_info[1]
                                                
                                                # CRITICAL: Include ALL text, regardless of confidence
                                                # Even if confidence is 0.0, include the text
                                                if line_text is not None and isinstance(line_text, str) and line_text.strip():
                                                    recognized_lines.append(line_text.strip())
                                                    total_confidence += line_confidence
                                                    valid_lines += 1
                                                    logger.debug(f"   → Line {line_idx}: '{line_text}' (confidence: {line_confidence:.4f})")
                                                else:
                                                    logger.debug(f"   → Line {line_idx}: Empty text, skipping")
                                            elif isinstance(text_info, str):
                                                # Sometimes PaddleOCR returns just text string
                                                if text_info.strip():
                                                    recognized_lines.append(text_info.strip())
                                                    valid_lines += 1
                                                    logger.debug(f"   → Line {line_idx}: '{text_info}' (no confidence score)")
                            elif hasattr(first_rec, 'shape'):  # NumPy array
                                # Handle NumPy array case
                                if first_rec.size > 0:
                                    for i in range(first_rec.shape[0]):
                                        rec_line = first_rec[i]
                                        if rec_line is not None:
                                            try:
                                                if hasattr(rec_line, 'shape') and rec_line.size > 1:
                                                    if rec_line.shape[0] > 1:
                                                        text_info = rec_line[1]
                                                        if isinstance(text_info, (str, tuple)):
                                                            if isinstance(text_info, tuple) and len(text_info) >= 1:
                                                                line_text = text_info[0]
                                                                if isinstance(line_text, str) and line_text.strip():
                                                                    recognized_lines.append(line_text.strip())
                                                                    valid_lines += 1
                                                                    logger.debug(f"   → Line {line_idx}: '{line_text}' (from NumPy array)")
                                                elif isinstance(rec_line, (list, tuple)) and len(rec_line) > 1:
                                                    text_info = rec_line[1]
                                                    if isinstance(text_info, (str, tuple)):
                                                        if isinstance(text_info, tuple) and len(text_info) >= 1:
                                                            line_text = text_info[0]
                                                            if isinstance(line_text, str) and line_text.strip():
                                                                recognized_lines.append(line_text.strip())
                                                                valid_lines += 1
                                                                logger.debug(f"   → Line {line_idx}: '{line_text}' (from mixed structure)")
                                            except (IndexError, TypeError, AttributeError) as e:
                                                logger.debug(f"   → Line {line_idx}: Error extracting from rec_line: {e}")
                                                continue
                    except (IndexError, TypeError, AttributeError) as e:
                        logger.debug(f"   → Line {line_idx}: Error accessing rec_result[0]: {e}")
                        continue
                
            except Exception as e:
                logger.warning(f"   → Line {line_idx} recognition error: {e}")
                continue
        
        # Step 4: Combine lines preserving order
        # CRITICAL: If ALL regions return empty, run full-image OCR
        if recognized_lines:
            # Join lines with newline to preserve line breaks
            combined_text = '\n'.join(recognized_lines)
            avg_confidence = total_confidence / valid_lines if valid_lines > 0 else 0.0
            
            logger.info(f"✅ [OCR DEBUG] HYBRID pipeline: {len(recognized_lines)} lines recognized, avg confidence: {avg_confidence:.4f}")
            return combined_text, avg_confidence
        else:
            # CRITICAL: If no text recognized from lines, DO NOT FAIL - run full-image OCR
            logger.warning("   ⚠️  No text recognized from lines - falling back to full-image OCR")
            logger.info("   🔄 Running full-image OCR (all regions returned empty)...")
            return self._run_full_image_ocr(image)
    
    def _extract_line_image(self, image: np.ndarray, box: List) -> Optional[np.ndarray]:
        """
        Extract line image from bounding box coordinates.
        
        Args:
            image: Source image
            box: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            Extracted line image (cropped and rotated if needed)
        """
        try:
            # Convert box to numpy array
            box = np.array(box, dtype=np.float32)
            
            # Get bounding rectangle
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
            
            # Add padding
            pad = 5
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(image.shape[1], x_max + pad)
            y_max = min(image.shape[0], y_max + pad)
            
            # Extract region
            if len(image.shape) == 3:
                line_image = image[y_min:y_max, x_min:x_max]
            else:
                line_image = image[y_min:y_max, x_min:x_max]
            
            # If box is rotated, we could apply perspective transform here
            # For now, use axis-aligned bounding box (good enough for most cases)
            
            return line_image
            
        except Exception as e:
            logger.warning(f"   → Line extraction error: {e}")
            return None
    
    def _light_preprocess(
        self,
        image: np.ndarray,
        region_type: str
    ) -> np.ndarray:
        """
        Light preprocessing ONLY:
        - Convert to grayscale
        - NO binarization
        - NO thresholding
        - NO morphological ops that break strokes
        
        Preserves original character shapes for handwriting recognition.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ensure minimum size (avoid resizing that distorts characters)
        h, w = gray.shape
        min_size = 32
        
        if w < min_size or h < min_size:
            scale = max(min_size / w, min_size / h) * 1.2
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Optional: Very light denoising (only if very noisy)
        # Most handwritten notes don't need this
        # gray = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
        
        return gray
    
    def _run_full_image_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on full image as fallback when region detection fails.
        
        CRITICAL: This is called when:
        - DB detection finds no regions
        - All region OCR returns empty
        - Detection errors occur
        
        This ensures OCR ALWAYS attempts full-image extraction before giving up.
        
        Args:
            image: Full image as numpy array
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        logger.info("   🔄 [Full-Image OCR] Attempting OCR on full image...")
        
        # Try PaddleOCR recognition on full image
        if self.paddleocr_available and self.paddleocr_recognizer:
            try:
                # Use recognition-only mode (det=False, rec=True)
                full_result = self.paddleocr_recognizer.ocr(image, det=False, rec=True, cls=True)
                if full_result is not None:
                    try:
                        first_rec = full_result[0]
                        if first_rec is not None:
                            recognized_texts = []
                            # Extract text from recognition result
                            if isinstance(first_rec, (list, tuple)) and len(first_rec) > 0:
                                for rec_item in first_rec:
                                    if rec_item is not None and isinstance(rec_item, (list, tuple)) and len(rec_item) > 1:
                                        text_info = rec_item[1]
                                        if isinstance(text_info, tuple) and len(text_info) >= 1:
                                            text = text_info[0]
                                            if isinstance(text, str) and text.strip():
                                                recognized_texts.append(text.strip())
                            elif hasattr(first_rec, 'shape') and first_rec.size > 0:
                                # Handle NumPy array structure
                                for i in range(first_rec.shape[0]):
                                    rec_item = first_rec[i]
                                    if rec_item is not None:
                                        try:
                                            if hasattr(rec_item, 'shape') and rec_item.size > 1:
                                                if rec_item.shape[0] > 1:
                                                    text_info = rec_item[1]
                                                    if isinstance(text_info, (str, tuple)):
                                                        if isinstance(text_info, tuple) and len(text_info) >= 1:
                                                            text = text_info[0]
                                                            if isinstance(text, str) and text.strip():
                                                                recognized_texts.append(text.strip())
                                        except (IndexError, TypeError, AttributeError):
                                            continue
                            
                            if recognized_texts:
                                combined_text = '\n'.join(recognized_texts)
                                logger.info(f"   ✅ [Full-Image OCR] Extracted {len(combined_text)} characters")
                                return combined_text, 0.5  # Lower confidence for full-image OCR
                    except (IndexError, TypeError, AttributeError) as e:
                        logger.debug(f"   → Full-image OCR result processing error: {e}")
            except Exception as e:
                logger.warning(f"   ⚠️  [Full-Image OCR] PaddleOCR error: {e}")
        
        # Fallback: EasyOCR on full image
        if self.easyocr_available:
            try:
                results = self.easyocr_reader.readtext(image)
                if results is not None and len(results) > 0:
                    texts = []
                    for result in results:
                        if len(result) >= 2:
                            text = result[1]
                            if isinstance(text, str) and text.strip():
                                texts.append(text.strip())
                    if texts:
                        combined_text = ' '.join(texts)
                        logger.info(f"   ✅ [Full-Image OCR] EasyOCR extracted {len(combined_text)} characters")
                        return combined_text, 0.4  # Lower confidence for EasyOCR full-image
            except Exception as e:
                logger.warning(f"   ⚠️  [Full-Image OCR] EasyOCR error: {e}")
        
        # If all full-image OCR methods fail, return empty
        logger.warning("   ⚠️  [Full-Image OCR] All methods failed - returning empty")
        return "", 0.0
    
    def _extract_with_easyocr(
        self,
        image: np.ndarray,
        region_type: str
    ) -> Tuple[str, float]:
        """
        Fallback: EasyOCR recognition.
        Used when HYBRID pipeline is unavailable.
        """
        try:
            results = self.easyocr_reader.readtext(image)
            
            if not results:
                return "", 0.0
            
            # Combine all detected text
            texts = []
            total_confidence = 0.0
            valid_results = 0
            
            for result in results:
                if len(result) >= 2:
                    text = result[1]
                    confidence = result[2] if len(result) > 2 else 0.5
                    
                    # CRITICAL: Include ALL text, regardless of confidence
                    if text and isinstance(text, str):
                        texts.append(text)
                        total_confidence += confidence
                        valid_results += 1
            
            if texts:
                combined_text = ' '.join(texts)  # EasyOCR is word-level, join with space
                avg_confidence = total_confidence / valid_results if valid_results > 0 else 0.0
                return combined_text, avg_confidence
            
            return "", 0.0
            
        except AttributeError as e:
            # Handle PIL.Image.ANTIALIAS deprecation in EasyOCR
            if 'ANTIALIAS' in str(e):
                logger.warning(f"EasyOCR extraction error: PIL.Image.ANTIALIAS deprecated (Pillow 10+). EasyOCR needs update.")
                logger.warning(f"   → Workaround: Try downgrading Pillow: pip install 'Pillow<10.0.0'")
                logger.warning(f"   → Or wait for EasyOCR to update their code")
            else:
                logger.warning(f"EasyOCR extraction error: {e}")
            return "", 0.0
        except Exception as e:
            logger.warning(f"EasyOCR extraction error: {e}")
            import traceback
            logger.debug(f"EasyOCR traceback: {traceback.format_exc()}")
            return "", 0.0
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"   ✅ GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"   CUDA Version: {torch.version.cuda}")
                logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                return True
            else:
                logger.warning("   ⚠️  No GPU detected - using CPU (slower)")
                return False
        except ImportError:
            logger.warning("   ⚠️  PyTorch not available - GPU check skipped")
            return False
        except Exception as e:
            logger.warning(f"   ⚠️  GPU check failed: {e}")
            return False
