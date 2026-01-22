"""
Handwritten equation recognition using FREE open-source tools.
Converts equation images to LaTeX format.
Uses pix2text or other free math-OCR tools.
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any
from PIL import Image
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MathOCR:
    """
    Math OCR engine for converting handwritten equations to LaTeX.
    """
    
    def __init__(self):
        """Initialize math OCR models."""
        logger.info("🔧 Initializing Math OCR models...")
        
        # Try pix2text (free and open-source)
        self.pix2text_available = False
        self.pix2text_model = None
        logger.info("📦 Loading pix2text model...")
        try:
            from pix2text import Pix2Text
            self.pix2text_model = Pix2Text(analyzer_config=dict(model_name='mfd'))
            self.pix2text_available = True
            logger.info("✅ pix2text initialized successfully")
        except ImportError:
            logger.warning("⚠️  pix2text not available (install with: pip install pix2text)")
        except Exception as e:
            logger.warning(f"⚠️  pix2text initialization failed: {e}")
        
        # Try mathpix-free alternative (if available)
        # Note: We can't use actual Mathpix API as it's paid
        # But we can use alternative free tools
        
        # Fallback: Use EasyOCR with LaTeX conversion heuristics
        logger.info("📦 Loading EasyOCR fallback for math OCR...")
        self.easyocr_fallback = False
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            self.easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            self.easyocr_fallback = True
            device_str = "GPU" if use_gpu else "CPU"
            logger.info(f"✅ EasyOCR fallback available for math OCR (using {device_str})")
        except Exception as e:
            logger.warning(f"⚠️  EasyOCR fallback not available: {e}")
            self.easyocr_reader = None
        
        if self.pix2text_available or self.easyocr_fallback:
            logger.info("✅ Math OCR models ready for equation recognition")
        else:
            logger.error("❌ No Math OCR engines available!")
    
    def recognize_equation(self, equation_image: np.ndarray) -> Dict[str, Any]:
        """
        Recognize equation from image using standard text symbols.
        Preserves mathematical symbols: +, -, =, /, x, ∑, ∫
        Converts low-confidence equations to readable text form.
        NEVER inserts equation images - always returns text.
        
        Args:
            equation_image: Image containing the equation (numpy array)
            
        Returns:
            Dictionary with:
            - 'latex': LaTeX string (if high confidence)
            - 'text': Readable text form (for low confidence or as fallback)
            - 'confidence': Confidence score (0.0 to 1.0)
            - 'is_complex': Boolean indicating if equation is complex
        """
        # Preprocess equation image for better recognition
        processed_image = self._preprocess_equation_image(equation_image)
        
        best_latex = ""
        best_text = ""
        best_confidence = 0.0
        is_complex = False
        
        # OCR DEBUG: Log equation recognition start
        logger.info(f"🔢 [OCR DEBUG] Starting equation recognition")
        
        # Try pix2text first (best for math OCR)
        if self.pix2text_available and self.pix2text_model:
            try:
                latex, confidence = self._recognize_with_pix2text(processed_image)
                logger.info(f"📝 [OCR DEBUG] pix2text result: latex='{latex}' (len={len(latex) if latex else 0}), confidence={confidence:.4f}")
                
                if latex and latex.strip():
                    # Preserve mathematical symbols
                    latex = self._preserve_math_symbols(latex)
                    if confidence > best_confidence:
                        best_latex = latex.strip()
                        best_confidence = confidence
                        # Check if equation is complex
                        is_complex = self._is_complex_equation(latex)
                        logger.info(f"✅ [OCR DEBUG] pix2text selected: latex='{best_latex}', confidence={best_confidence:.4f}")
                    else:
                        logger.debug(f"⚠️  [OCR DEBUG] pix2text NOT selected (confidence {confidence:.4f} <= {best_confidence:.4f})")
                else:
                    logger.info(f"⚠️  [OCR DEBUG] pix2text returned empty/invalid: latex={latex}")
            except Exception as e:
                logger.warning(f"⚠️  [OCR DEBUG] pix2text recognition error: {e}")
        
        # Fallback to OCR with symbol preservation
        if self.easyocr_fallback and self.easyocr_reader:
            try:
                text_result = self._recognize_with_ocr_fallback(processed_image)
                if text_result:
                    # Extract text and confidence
                    ocr_text = text_result.get('text', '')
                    ocr_confidence = text_result.get('confidence', 0.0)
                    logger.info(f"📝 [OCR DEBUG] EasyOCR fallback result: text='{ocr_text}' (len={len(ocr_text)}), confidence={ocr_confidence:.4f}")
                    
                    if ocr_text:
                        # Preserve mathematical symbols
                        preserved_text = self._preserve_math_symbols(ocr_text)
                        
                        # REMOVED: Confidence threshold filtering
                        # CRITICAL: Try LaTeX conversion regardless of confidence
                        # Do NOT filter based on confidence threshold
                        latex = self._convert_text_to_latex(preserved_text)
                        if latex and latex.strip():
                            if ocr_confidence > best_confidence:
                                best_latex = latex.strip()
                                best_confidence = ocr_confidence
                                is_complex = self._is_complex_equation(latex)
                                logger.info(f"✅ [OCR DEBUG] EasyOCR LaTeX selected: latex='{best_latex}', confidence={best_confidence:.4f}")
                        
                        # Also try readable text form (regardless of confidence)
                        readable_text = self._convert_to_readable_text(preserved_text)
                        if ocr_confidence > best_confidence:
                            best_text = readable_text
                            best_confidence = ocr_confidence
                            logger.info(f"✅ [OCR DEBUG] EasyOCR text selected: text='{best_text}', confidence={best_confidence:.4f}")
                    else:
                        logger.info(f"⚠️  [OCR DEBUG] EasyOCR fallback returned empty text")
            except Exception as e:
                logger.warning(f"⚠️  [OCR DEBUG] OCR fallback error: {e}")
        
        # REMOVED: Confidence-based filtering
        # CRITICAL: Include ALL OCR results, regardless of confidence
        # Do NOT discard LaTeX due to low confidence - preserve raw OCR output
        # Keep both LaTeX and text if available, don't filter by confidence
        
        # If no result, try direct OCR as text
        if not best_latex and not best_text:
            try:
                # Direct OCR attempt with symbol preservation
                direct_text = self._direct_ocr_with_symbols(processed_image)
                logger.info(f"📝 [OCR DEBUG] Direct OCR result: text='{direct_text}' (len={len(direct_text) if direct_text else 0})")
                
                if direct_text:
                    best_text = direct_text
                    best_confidence = 0.2  # Low confidence
                    logger.info(f"✅ [OCR DEBUG] Direct OCR selected: text='{best_text}', confidence={best_confidence:.4f}")
            except Exception as e:
                logger.warning(f"⚠️  [OCR DEBUG] Direct OCR error: {e}")
        
        # CRITICAL: NEVER use placeholders - always return actual OCR text
        # If no text detected, return empty string (not placeholder)
        # Text is ALWAYS better than nothing, but we have no text here
        
        # Return result
        result = {
            "confidence": best_confidence,
            "is_complex": is_complex
        }
        
        # REMOVED: Confidence threshold filtering
        # CRITICAL: Include ALL OCR results, regardless of confidence
        # Do NOT filter LaTeX based on confidence threshold
        # If LaTeX exists, use it (even if low confidence)
        if best_latex:
            result["latex"] = best_latex
        # Always include text if available (regardless of confidence)
        if best_text:
            result["text"] = best_text
        # If neither exists, return empty (not placeholder)
        
        # OCR DEBUG: Log final equation result
        logger.info(f"✅ [OCR DEBUG] FINAL equation result: latex='{best_latex}' (len={len(best_latex)}), text='{best_text}' (len={len(best_text)}), confidence={best_confidence:.4f}, is_complex={is_complex}")
        
        return result
    
    def _preprocess_equation_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess equation image for optimal OCR accuracy.
        Equations need special handling: high contrast, denoising, resizing.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Enhance contrast aggressively (equations need high contrast)
        # Increased clip limit for better symbol visibility in handwritten equations
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply additional sharpening for equation symbols
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Step 2: Apply adaptive thresholding for cleaner binary image
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Step 3: Remove small noise while preserving equation symbols
        # Use morphological operations carefully to preserve symbols like +, -, =, etc.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Step 4: Invert back to normal (black text on white)
        denoised = 255 - denoised
        
        # Step 5: Resize to optimal size (math OCR works better with larger images)
        h, w = denoised.shape
        min_height = 80  # Minimum height for good recognition
        min_width = 200  # Minimum width for good recognition
        
        if h < min_height or w < min_width:
            scale_h = max(1.0, min_height / h)
            scale_w = max(1.0, min_width / w)
            scale = max(scale_h, scale_w) * 1.5  # Extra scaling for better quality
            
            new_h = int(h * scale)
            new_w = int(w * scale)
            resized = cv2.resize(
                denoised, (new_w, new_h),
                interpolation=cv2.INTER_CUBIC
            )
        else:
            resized = denoised.copy()
        
        # Step 6: Add padding for better recognition
        pad = 20
        padded = cv2.copyMakeBorder(
            resized, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=255  # White background
        )
        
        return padded
    
    def _recognize_with_pix2text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize equation using pix2text (best for math OCR).
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # pix2text can work with PIL Image or file path
        try:
            # Save to temp file (pix2text typically works with file paths)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            pil_image.save(temp_file.name, 'PNG')
            temp_path = temp_file.name
            
            try:
                # Use pix2text to recognize (it returns text by default, but we want LaTeX)
                result = self.pix2text_model(temp_path)
                
                # pix2text returns text or structured result
                confidence = 0.7  # Default confidence for pix2text
                latex = ""
                
                if isinstance(result, str):
                    # If it's already LaTeX or text, return it
                    latex = result
                elif isinstance(result, dict):
                    # Check for LaTeX in result
                    if 'latex' in result:
                        latex = result['latex']
                        confidence = result.get('confidence', 0.7)
                    elif 'text' in result:
                        # Convert text to LaTeX
                        latex = self._convert_text_to_latex(result['text'])
                        confidence = result.get('confidence', 0.6)
                elif isinstance(result, list):
                    # Multiple results, extract LaTeX from each
                    latex_parts = []
                    confidences = []
                    for item in result:
                        if isinstance(item, dict):
                            if 'latex' in item:
                                latex_parts.append(item['latex'])
                                confidences.append(item.get('confidence', 0.7))
                            elif 'text' in item:
                                latex_parts.append(self._convert_text_to_latex(item['text']))
                                confidences.append(item.get('confidence', 0.6))
                        elif isinstance(item, str):
                            latex_parts.append(item)
                            confidences.append(0.7)
                    if latex_parts:
                        latex = " ".join(latex_parts)
                        confidence = sum(confidences) / len(confidences) if confidences else 0.7
                
                if latex:
                    return latex, confidence
            finally:
                Path(temp_path).unlink()
        except Exception as e:
            print(f"pix2text recognition error: {e}")
        
        return "", 0.0
    
    def _recognize_with_ocr_fallback(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Fallback method: Use EasyOCR with symbol preservation.
        Returns text with confidence score.
        """
        # Convert to BGR for EasyOCR
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image
        
        # Perform OCR
        results = self.easyocr_reader.readtext(image_bgr)
        
        if not results:
            return {"text": "", "confidence": 0.0}
        
        # Extract text with bounding boxes - accept lower confidence for handwriting
        texts = []
        confidences = []
        # REMOVED: Confidence threshold filtering
        # CRITICAL: Include ALL OCR results, regardless of confidence
        # Do NOT filter based on confidence threshold
        for (bbox, text, confidence) in results:
            text_clean = text.strip()
            if text_clean:
                # Include all text, regardless of confidence
                # No confidence threshold filtering
                    texts.append(text_clean)
                    confidences.append(confidence)
        
        if not texts:
            return {"text": "", "confidence": 0.0}
        
        # Combine text
        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {"text": combined_text, "confidence": avg_confidence}
    
    def _direct_ocr_with_symbols(self, image: np.ndarray) -> str:
        """
        Direct OCR attempt preserving mathematical symbols.
        Used as last resort fallback.
        """
        if not self.easyocr_fallback or not self.easyocr_reader:
            return ""
        
        # Convert to BGR for EasyOCR
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image
        
        # Perform OCR
        results = self.easyocr_reader.readtext(image_bgr)
        
        if not results:
            return ""
        
        # Extract all text, preserving symbols
        texts = []
        for (bbox, text, confidence) in results:
            if text.strip():
                texts.append(text.strip())
        
        if not texts:
            return ""
        
        combined_text = " ".join(texts)
        # Preserve mathematical symbols
        return self._preserve_math_symbols(combined_text)
    
    def _preserve_math_symbols(self, text: str) -> str:
        """
        Preserve mathematical symbols: +, -, =, /, x, ∑, ∫
        Ensures these symbols are not corrupted during OCR.
        
        CRITICAL: Preserve RAW OCR output for handwritten notes.
        - DO NOT auto-correct symbols aggressively
        - DO NOT replace characters
        - Preserve original OCR output
        
        Priority: RAW OCR OUTPUT > CLEAN OUTPUT
        
        Args:
            text: Raw OCR text
            
        Returns:
            Text as-is (no corrections)
        """
        # Return raw OCR output - no symbol corrections
        # Handwritten notes may have irregular symbols - preserve them
        return text
    
    def _convert_text_to_latex(self, text: str) -> str:
        """
        Convert OCR text to LaTeX format using heuristics.
        This is a simple converter - not perfect but better than raw OCR.
        """
        import re
        
        # PRESERVE RAW OCR OUTPUT - do NOT collapse spaces
        # Handwritten notes may have irregular spacing - preserve it
        # Only normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Convert common operators
        conversions = {
            '->': r'\rightarrow',
            '→': r'\rightarrow',
            '=>': r'\Rightarrow',
            '⇒': r'\Rightarrow',
            '==': '=',
            '+-': r'\pm',
            '±': r'\pm',
            '**': '^',
            '^': '^',
            '_': '_',
            '/': r' / ',
            '*': r' \cdot ',
        }
        
        for old, new in conversions.items():
            text = text.replace(old, new)
        
        # Convert superscripts (numbers after letters or closing parens)
        # Pattern: X2 -> X^{2}
        text = re.sub(r'([A-Za-z\)])(\d+)', r'\1^{\2}', text)
        
        # Convert subscripts (numbers before letters)
        # Pattern: 2X -> _{2}X (simplified)
        
        # Handle fractions: a/b -> \frac{a}{b}
        text = re.sub(r'(\w+)\s*/\s*(\w+)', r'\\frac{\1}{\2}', text)
        
        # Handle square roots: sqrt(X) -> \sqrt{X}
        text = re.sub(r'sqrt\s*\(([^)]+)\)', r'\\sqrt{\1}', text, flags=re.IGNORECASE)
        
        # Handle powers: X^Y -> X^{Y}
        text = re.sub(r'\^(\w+)', r'^{\1}', text)
        
        # Wrap chemical formulas and common math expressions
        # This is basic - can be improved
        
        return text
    
    def _convert_to_readable_text(self, text: str) -> str:
        """
        Convert mathematical expression to readable text form.
        Example: "x^2 + y^2 = r^2" -> "x squared plus y squared equals r squared"
        Used for low-confidence equations.
        """
        import re
        
        # Preserve symbols first
        text = self._preserve_math_symbols(text)
        
        # Convert powers to "squared", "cubed", etc.
        text = re.sub(r'(\w+)\^2\b', r'\1 squared', text)
        text = re.sub(r'(\w+)\^3\b', r'\1 cubed', text)
        text = re.sub(r'(\w+)\^(\d+)', r'\1 to the power of \2', text)
        
        # Convert operators to words
        replacements = {
            '+': ' plus ',
            '-': ' minus ',
            '=': ' equals ',
            '/': ' divided by ',
            'x': ' times ',
            '*': ' times ',
            '∑': ' sum of ',
            '∫': ' integral of ',
        }
        
        for symbol, word in replacements.items():
            text = text.replace(symbol, word)
        
        # PRESERVE RAW OCR OUTPUT - do NOT collapse spaces
        # Handwritten notes may have irregular spacing - preserve it
        # Only normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text
    
    def _latex_to_readable_text(self, latex: str) -> str:
        """
        Convert LaTeX to readable text form.
        Example: "x^{2} + y^{2} = r^{2}" -> "x squared plus y squared equals r squared"
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
        # Only normalize line endings, but preserve spacing within text
        readable = readable.replace('\r\n', '\n').replace('\r', '\n')
        
        return readable
    
    def _is_complex_equation(self, latex_or_text: str) -> bool:
        """
        Determine if an equation is complex (should be tagged as [complex equation]).
        Complex equations include integrals, summations, fractions, etc.
        """
        complex_indicators = [
            '\\int', '∫',  # Integrals
            '\\sum', '∑',  # Summations
            '\\frac',  # Fractions
            '\\sqrt',  # Square roots
            '\\lim',  # Limits
            '\\partial',  # Partial derivatives
            '\\nabla',  # Gradient
            '\\infty',  # Infinity
        ]
        
        for indicator in complex_indicators:
            if indicator in latex_or_text:
                return True
        
        return False


def recognize_equation(equation_image: np.ndarray) -> Dict[str, str]:
    """
    Convenience function to recognize equation from image.
    
    Args:
        equation_image: Image containing the equation
        
    Returns:
        Dictionary with 'latex' key containing LaTeX string
    """
    # Lazy initialization
    if not hasattr(recognize_equation, '_math_ocr'):
        recognize_equation._math_ocr = MathOCR()
    
    return recognize_equation._math_ocr.recognize_equation(equation_image)
