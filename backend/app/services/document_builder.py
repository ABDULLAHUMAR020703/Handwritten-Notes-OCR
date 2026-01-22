"""
Document builder that combines all extracted elements into structured JSON.
Preserves reading order and formats output according to specification.
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DocumentBuilder:
    """
    Build structured JSON document from extracted elements.
    Combines: headings, paragraphs, equations, and diagrams.
    """
    
    def __init__(self):
        """Initialize document builder."""
        pass
    
    def build_document(
        self,
        layout_regions: List[Dict],
        text_extractions: Dict[str, Any],  # region_id -> text or (text, confidence)
        equation_extractions: Dict[str, Any],  # region_id -> latex or dict with latex
        diagram_extractions: List[Dict]  # List of diagram info dicts
    ) -> List[Dict[str, Any]]:
        """
        Combine all extracted elements into structured JSON.
        
        Args:
            layout_regions: List of layout regions with type and bbox
            text_extractions: Dictionary mapping region indices to extracted text/content
            equation_extractions: Dictionary mapping region indices to LaTeX content
            diagram_extractions: List of diagram extraction results
            
        Returns:
            List of document elements in reading order
        """
        logger.info(f"📄 Building structured document from {len(layout_regions)} regions")
        logger.debug(f"   Text extractions: {len(text_extractions)}")
        logger.debug(f"   Equation extractions: {len(equation_extractions)}")
        logger.debug(f"   Diagram extractions: {len(diagram_extractions)}")
        
        # Create a unified list of all elements
        all_elements = []
        
        # Process text regions (headings, paragraphs, equations)
        for idx, region in enumerate(layout_regions):
            region_type = region.get('type', 'paragraph')
            bbox = region.get('bbox', [0, 0, 0, 0])
            
            # Handle diagrams - they have text placeholders, not images
            # Diagrams are detected and skipped during OCR, but text placeholders are added
            # Process diagram text placeholders like regular text regions
            # if region_type == 'diagram':
            #     continue  # Now handled as text with placeholder
            
            # Get extracted text for this region
            # Handle both string and tuple formats
            text_content = text_extractions.get(str(idx)) or text_extractions.get(idx)
            text = ""
            confidence = 0.0
            original_text = None
            
            if isinstance(text_content, tuple):
                text, confidence = text_content  # (text, confidence)
                original_text = text
            elif isinstance(text_content, dict):
                text = text_content.get('text', '')
                confidence = text_content.get('confidence', 0.0)
                original_text = text
            elif isinstance(text_content, str):
                text = text_content
                original_text = text
            
            # OCR DEBUG: Log extracted text for this region
            logger.info(f"📄 [OCR DEBUG] Region {idx} ({region_type}): raw_text='{original_text}' (len={len(original_text) if original_text else 0}), confidence={confidence:.4f}")
            
            # Ensure text is a string (not an image path or other type)
            if not isinstance(text, str):
                logger.warning(f"🚨 [OCR DEBUG] BUG: Region {idx} text is not string: {type(text)} = {text}, converting to string")
                text = str(text) if text else ""
            
            # CRITICAL: Never allow image paths in text fields
            # Remove any image path strings that might have leaked in
            if text and any(text.strip().lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']):
                logger.warning(f"🚨 [OCR DEBUG] BUG: Region {idx} text is image path, DISCARDED: '{text}'")
                text = ""
            
            # CRITICAL: If text is empty, log warning but DO NOT add placeholder
            # ABSOLUTE RULE: Placeholder messages must NEVER reach user output
            if not text or not text.strip():
                # No text extracted - log warning but skip region (no placeholder)
                logger.warning(f"🚨 [OCR DEBUG] Region {idx} ({region_type}) text empty - skipping (no placeholder added)")
                # DO NOT add placeholder text - skip this region entirely
                # Placeholder messages are FORBIDDEN in final output
                continue  # Skip this region - don't add placeholder
            
            # OCR DEBUG: Log that text is being included
            logger.info(f"✅ [OCR DEBUG] Region {idx} ({region_type}) INCLUDED: text='{text}' (len={len(text)}), confidence={confidence:.4f}")
            
            # Get equation LaTeX or text if it's an equation
            if region_type == 'equation':
                eq_content = equation_extractions.get(str(idx)) or equation_extractions.get(idx)
                latex = ""
                eq_text = ""
                confidence = 0.0
                is_complex = False
                
                if isinstance(eq_content, dict):
                    latex = eq_content.get('latex', '')
                    eq_text = eq_content.get('text', '')
                    confidence = eq_content.get('confidence', 0.0)
                    is_complex = eq_content.get('is_complex', False)
                elif isinstance(eq_content, str):
                    # Legacy format - treat as LaTeX
                    latex = eq_content
                    confidence = 0.5  # Default confidence
                
                # REMOVED: Confidence threshold filtering
                # CRITICAL: Include ALL OCR results, regardless of confidence
                # Do NOT filter LaTeX based on confidence threshold
                # If LaTeX exists, use it (even if low confidence)
                if latex and latex.strip():
                    # Use LaTeX regardless of confidence
                    element = {
                        "type": "equation",
                        "latex": latex.strip()
                    }
                elif eq_text and eq_text.strip():
                    # Low confidence or text form - use readable text
                    # Tag complex equations
                    if is_complex and "[complex equation]" not in eq_text:
                        final_text = f"[complex equation] {eq_text}"
                    else:
                        final_text = eq_text
                    
                    element = {
                        "type": "paragraph",
                        "text": final_text.strip()
                    }
                elif latex and latex.strip():
                    # LaTeX exists but low confidence - convert to readable text
                    readable_text = _latex_to_readable_text(latex)
                    if is_complex:
                        readable_text = f"[complex equation] {readable_text}"
                    element = {
                        "type": "paragraph",
                        "text": readable_text.strip()
                    }
                else:
                    # No equation content - use text OCR result
                    # CRITICAL: If text is empty, log warning but DO NOT add placeholder
                    if text and text.strip():
                        element = {
                            "type": "paragraph",
                            "text": text.strip()
                        }
                    else:
                        # No text available - log warning but DO NOT add placeholder
                        # CRITICAL: Placeholder messages must NEVER reach user output
                        logger.warning(f"🚨 [OCR DEBUG] Region {idx} equation: No text available - skipping (no placeholder added)")
                        # DO NOT add placeholder text - skip this region
                        continue  # Skip this region - don't add placeholder
            elif region_type == 'heading':
                # Always include headings, even if unclear
                element = {
                    "type": "heading",
                    "text": text.strip()
                }
            else:  # paragraph or other text
                # Always include paragraphs, even if unclear
                element = {
                    "type": "paragraph",
                    "text": text.strip()
                }
            
            # Add position info for sorting
            element['_bbox'] = bbox
            element['_y'] = bbox[1] if len(bbox) >= 2 else 0
            element['_index'] = idx
            element['_type'] = region_type
            
            all_elements.append(element)
        
        # CRITICAL: Diagrams are SKIPPED - no images should be inserted into documents
        # Only text extracted via OCR should be included
        # If OCR fails for a diagram region, it is skipped entirely
        # Process diagrams
        # for diag in diagram_extractions:
        #     image_path = diag.get('image_path', '')
        #     if not image_path:
        #         continue
        #     
        #     # Get just the filename
        #     diagram_file = Path(image_path).name
        #     
        #     element = {
        #         "type": "diagram",
        #         "file": diagram_file
        #     }
        #     
        #     # Add position info for sorting
        #     bbox = diag.get('bbox', [0, 0, 0, 0])
        #     element['_bbox'] = bbox
        #     element['_y'] = bbox[1] if len(bbox) >= 2 else 0
        #     element['_type'] = 'diagram'
        #     
        #     all_elements.append(element)
        
        # Sort by reading order (y-coordinate first, then x-coordinate)
        sorted_elements = self._sort_by_reading_order(all_elements)
        
        # Remove internal sorting fields
        final_document = []
        for elem in sorted_elements:
            clean_elem = {k: v for k, v in elem.items() if not k.startswith('_')}
            final_document.append(clean_elem)
        
        # CRITICAL: NEVER return empty document
        # If no elements were extracted, add fallback paragraph
        if not final_document:
            logger.error(f"🚨 CRITICAL: Document builder produced EMPTY document!")
            logger.error(f"🚨 Adding mandatory fallback paragraph to prevent blank document")
            final_document.append({
                "type": "paragraph",
                "text": "[OCR pipeline executed but no readable text was extracted]"
            })
        
        return final_document
    
    def _sort_by_reading_order(self, elements: List[Dict]) -> List[Dict]:
        """
        Sort elements by reading order (top-to-bottom, left-to-right).
        Uses y-coordinate primarily, then x-coordinate.
        
        Args:
            elements: List of elements with _y and _bbox
            
        Returns:
            Sorted list of elements
        """
        if not elements:
            return []
        
        # Group elements by approximate row (similar y-coordinates)
        # Elements within a threshold are considered on the same row
        y_threshold = 30  # pixels
        
        # Sort all elements by y-coordinate
        sorted_by_y = sorted(elements, key=lambda e: e.get('_y', 0))
        
        # Group into rows
        rows = []
        current_row = [sorted_by_y[0]]
        
        for elem in sorted_by_y[1:]:
            if not current_row:
                current_row = [elem]
                continue
            
            # Calculate average y of current row
            avg_y = np.mean([e.get('_y', 0) for e in current_row])
            elem_y = elem.get('_y', 0)
            
            # Check if element is on the same row
            if abs(elem_y - avg_y) < y_threshold:
                current_row.append(elem)
            else:
                # Start new row
                rows.append(current_row)
                current_row = [elem]
        
        # Don't forget the last row
        if current_row:
            rows.append(current_row)
        
        # Sort each row by x-coordinate (left-to-right)
        final_order = []
        for row in rows:
            row_sorted = sorted(row, key=lambda e: e.get('_bbox', [0])[0])
            final_order.extend(row_sorted)
        
        return final_order
    
    def build_from_extractions(
        self,
        layout_results: List[Dict],
        ocr_results: Dict[str, Any],
        equation_results: Dict[str, Any],
        diagram_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Convenience method that takes extraction results and builds document.
        
        Args:
            layout_results: Layout detection results (from layout.py)
            ocr_results: Text OCR results (from text_ocr.py)
            equation_results: Equation recognition results (from math_ocr.py)
            diagram_results: Diagram extraction results (from diagram.py)
            
        Returns:
            Structured JSON document
        """
        # Map layout regions to extracted content
        text_extractions = {}
        equation_extractions = {}
        
        # Process OCR results
        # Assuming OCR results map region indices to text
        for key, value in ocr_results.items():
            if isinstance(key, (int, str)):
                text_extractions[str(key)] = value
        
        # Process equation results
        # Assuming equation results map region indices to LaTeX
        for key, value in equation_results.items():
            if isinstance(key, (int, str)):
                if isinstance(value, dict) and 'latex' in value:
                    equation_extractions[str(key)] = value['latex']
                elif isinstance(value, str):
                    equation_extractions[str(key)] = value
        
        # Build document
        return self.build_document(
            layout_regions=layout_results,
            text_extractions=text_extractions,
            equation_extractions=equation_extractions,
            diagram_extractions=diagram_results
        )
    
    def to_json(
        self,
        document: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
        indent: int = 2
    ) -> str:
        """
        Convert document to JSON string or save to file.
        
        Args:
            document: Structured document list
            output_path: Optional path to save JSON file
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        json_string = json.dumps(document, indent=indent, ensure_ascii=False)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_string)
        
        return json_string


def build_document(
    layout_regions: List[Dict],
    text_extractions: Dict[str, str],
    equation_extractions: Dict[str, str],
    diagram_extractions: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Convenience function to build structured document.
    
    Args:
        layout_regions: List of layout regions
        text_extractions: Dictionary of region_id -> text
        equation_extractions: Dictionary of region_id -> latex
        diagram_extractions: List of diagram extraction results
        
    Returns:
        Structured JSON document
    """
    builder = DocumentBuilder()
    return builder.build_document(
        layout_regions,
        text_extractions,
        equation_extractions,
        diagram_extractions
    )


def _latex_to_readable_text(latex: str) -> str:
    """
    Convert LaTeX to readable text form.
    Example: "x^{2} + y^{2} = r^{2}" -> "x squared plus y squared equals r squared"
    Used for low-confidence equations.
    """
    import re
    
    # Remove LaTeX commands and convert to readable form
    readable = latex
    
    # Convert superscripts
    readable = re.sub(r'\^\{(\d+)\}', r' to the power of \1', readable)
    readable = re.sub(r'\^(\d+)', r' to the power of \1', readable)
    readable = re.sub(r'\^\{2\}', ' squared', readable)
    readable = re.sub(r'\^2', ' squared', readable)
    readable = re.sub(r'\^\{3\}', ' cubed', readable)
    readable = re.sub(r'\^3', ' cubed', readable)
    
    # Convert fractions
    readable = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 divided by \2', readable)
    
    # Convert square roots
    readable = re.sub(r'\\sqrt\{([^}]+)\}', r'square root of \1', readable)
    
    # Convert operators
    readable = readable.replace('+', ' plus ')
    readable = readable.replace('-', ' minus ')
    readable = readable.replace('=', ' equals ')
    readable = readable.replace('/', ' divided by ')
    readable = readable.replace('\\cdot', ' times ')
    readable = readable.replace('\\times', ' times ')
    readable = readable.replace('∑', ' sum of ')
    readable = readable.replace('∫', ' integral of ')
    
    # Remove LaTeX braces
    readable = readable.replace('{', '').replace('}', '')
    
    # PRESERVE RAW OCR OUTPUT - do NOT collapse spaces
    # Handwritten notes may have irregular spacing - preserve it
    # Only normalize line endings
    readable = readable.replace('\r\n', '\n').replace('\r', '\n')
    
    return readable
