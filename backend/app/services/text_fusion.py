"""
Text Fusion Layer - Merges Gemini Vision and OCR/CRNN outputs.

CRITICAL RULES:
- Prefer text PRESENCE over confidence
- Include BOTH outputs if they differ
- Maximum information, not minimal noise
- NEVER discard text
"""
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class TextFusion:
    """
    THREE-LAYER SYSTEM - Layer 3: Text Fusion
    
    Merges outputs from:
    - Layer 1: Gemini Vision (full image extraction)
    - Layer 2: OCR/CRNN (line-level recognition)
    
    Strategy:
    - If both outputs exist → include BOTH
    - If outputs differ → include BOTH
    - If one is empty → use the other
    - NEVER suppress duplicates aggressively
    - NEVER choose "better" output based on confidence
    """
    
    def fuse_texts(
        self,
        gemini_text: str,
        ocr_texts: Dict[str, Tuple[str, float]],  # region_id -> (text, confidence)
        preserve_order: bool = True
    ) -> List[Dict[str, any]]:
        """
        Fuse Gemini Vision output with OCR/CRNN outputs.
        
        Args:
            gemini_text: Full text extracted by Gemini Vision (Layer 1)
            ocr_texts: Dictionary of OCR outputs per region (Layer 2)
            preserve_order: Whether to preserve reading order
            
        Returns:
            List of text elements (paragraphs) with fused content
        """
        fused_elements = []
        
        # CRITICAL: Include Gemini output FIRST (it's full-image extraction)
        # NEVER discard empty Gemini text - it's optional, not required
        if gemini_text and gemini_text.strip():
            logger.info(f"📝 [Text Fusion] Gemini text: {len(gemini_text)} characters")
            
            # Split Gemini text into lines/paragraphs
            gemini_lines = gemini_text.split('\n')
            for line in gemini_lines:
                line = line.strip()
                if line:  # Only add non-empty lines
                    fused_elements.append({
                        'type': 'paragraph',
                        'text': line,
                        'source': 'gemini',
                        'confidence': 0.8  # Assume good for Gemini
                    })
        else:
            logger.info(f"📝 [Text Fusion] Gemini text: empty (optional, continuing)")
        
        # CRITICAL: Include ALL OCR outputs (Layer 2 verification)
        # NEVER discard empty OCR text - use fallback token instead
        ocr_count = 0
        for region_id, ocr_content in ocr_texts.items():
            # Handle both tuple and string formats
            if isinstance(ocr_content, tuple):
                ocr_text, confidence = ocr_content
            elif isinstance(ocr_content, str):
                ocr_text = ocr_content
                confidence = 0.0
            else:
                ocr_text = str(ocr_content) if ocr_content else ""
                confidence = 0.0
            
            if ocr_text and ocr_text.strip():
                ocr_count += 1
                logger.info(f"📝 [Text Fusion] OCR region {region_id}: {len(ocr_text)} characters (confidence: {confidence:.4f})")
                
                # Split OCR text into lines
                ocr_lines = ocr_text.split('\n')
                for line in ocr_lines:
                    line = line.strip()
                    if line:  # Only add non-empty lines
                        fused_elements.append({
                            'type': 'paragraph',
                            'text': line,
                            'source': 'ocr',
                            'confidence': confidence
                        })
            else:
                # Empty OCR text - log warning but DO NOT add placeholder to output
                # CRITICAL: Placeholder messages must NEVER reach user output
                logger.warning(f"📝 [Text Fusion] OCR region {region_id}: empty - skipping (no placeholder added)")
                # DO NOT add placeholder text - skip this region
                # Placeholder messages are FORBIDDEN in final output
        
        logger.info(f"📝 [Text Fusion] OCR regions: {ocr_count} with text")
        
        # CRITICAL: NEVER return empty list - always include fallback
        # ABSOLUTE RULE: Blank documents are FORBIDDEN
        if not fused_elements:
            logger.error(f"🚨 [Text Fusion] CRITICAL: Both Gemini and OCR returned empty!")
            logger.error(f"🚨 [Text Fusion] Adding mandatory fallback paragraph")
            fused_elements.append({
                'type': 'paragraph',
                'text': '[OCR pipeline executed but no readable text was extracted]',
                'source': 'fallback',
                'confidence': 0.0
            })
        
        logger.info(f"✅ [Text Fusion] Fused {len(fused_elements)} text elements")
        return fused_elements
    
    def merge_region_outputs(
        self,
        gemini_text: str,
        region_ocr_outputs: Dict[str, Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Alternative fusion method: Merge Gemini with region-specific OCR outputs.
        
        Args:
            gemini_text: Full text from Gemini Vision
            region_ocr_outputs: Dictionary mapping region_id to OCR output dict
            
        Returns:
            List of fused text elements
        """
        fused_elements = []
        
        # Add Gemini output
        if gemini_text and gemini_text.strip():
            gemini_lines = gemini_text.split('\n')
            for line in gemini_lines:
                line = line.strip()
                if line:
                    fused_elements.append({
                        'type': 'paragraph',
                        'text': line,
                        'source': 'gemini'
                    })
        
        # Add OCR outputs
        for region_id, ocr_output in region_ocr_outputs.items():
            text = ocr_output.get('text', '')
            if text and text.strip():
                text_lines = text.split('\n')
                for line in text_lines:
                    line = line.strip()
                    if line:
                        fused_elements.append({
                            'type': 'paragraph',
                            'text': line,
                            'source': f'ocr_region_{region_id}'
                        })
        
        return fused_elements
