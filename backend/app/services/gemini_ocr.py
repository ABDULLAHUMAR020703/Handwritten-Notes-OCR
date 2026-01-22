"""
Gemini API for POST-OCR text cleanup and enhancement.
Used AFTER OCR extraction to improve text quality.

CRITICAL RULES:
- Input: Extracted OCR text (string) - NEVER images
- Task: Clean up OCR text (spelling, readability)
- Output: Cleaned text (text-only)
- Preserve technical terms and original meaning
- Do NOT replace OCR - only enhance OCR output

MANDATORY MODEL POLICY:
- Use ONLY models that are ACTUALLY available via list_models()
- Default: models/gemini-2.5-pro (if available)
- Fallback: models/gemini-2.5-flash (if 2.5-pro not available)
- Disable Gemini if neither model is available

PIPELINE:
raw_text = full_image_ocr(image)
cleaned_text = gemini_cleanup(raw_text)  # Only if raw_text is not empty
"""
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

# PREFERRED GEMINI MODELS (in order of preference)
# These will be checked against actual availability
PREFERRED_MODELS = [
    'models/gemini-2.5-pro',    # Primary choice
    'models/gemini-2.5-flash',  # Fallback choice
]

# DEFAULT MODEL (will be set dynamically based on availability)
DEFAULT_GEMINI_MODEL = 'models/gemini-2.5-pro'  # Preferred, but will validate

# FORBIDDEN MODEL PATTERNS
FORBIDDEN_PATTERNS = [
    '-latest',
    'gemini-pro',   # without version
    'gemini-flash', # without version
]


class GeminiTextCleanup:
    """
    Gemini API for POST-OCR text cleanup and enhancement.
    Used AFTER OCR extraction to improve text quality.
    
    MANDATORY: Uses only officially supported, versioned Gemini models.
    Validates model availability at startup.
    
    ROLE: Text cleanup only - NEVER performs OCR or sees images.
    """
    
    def __init__(self):
        """Initialize Gemini text cleanup service with startup validation."""
        self.available = False
        self.client = None
        self.model_name = None
        self.genai = None
        
        try:
            import google.generativeai as genai
            self.genai = genai
            
            # Get API key from environment variable
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.info("⚠️  GEMINI_API_KEY not set - Gemini text cleanup unavailable")
                logger.info("   Set GEMINI_API_KEY environment variable to enable Gemini text cleanup")
                return
            
            # Configure API
            genai.configure(api_key=api_key)
            
            # CRITICAL: Select model based on ACTUAL availability
            # No hardcoded assumptions - check what's actually available
            selected_model = self._select_available_model()
            
            if not selected_model:
                logger.error(f"❌ No available Gemini models found")
                logger.error(f"   Checked models: {PREFERRED_MODELS}")
                logger.error(f"   ⚠️  Gemini will be disabled - text cleanup will be skipped")
                self.available = False
                return
            
            # CRITICAL: Validate model name - check for forbidden patterns
            # CRITICAL: Gemini is OPTIONAL - validation failures should NOT crash server
            if self._is_forbidden_model(selected_model):
                logger.error(f"❌ FORBIDDEN MODEL NAME DETECTED: {selected_model}")
                logger.error(f"   Forbidden patterns: {FORBIDDEN_PATTERNS}")
                logger.error(f"   ⚠️  Gemini will be disabled - text cleanup will be skipped")
                self.available = False
                return
            
            # CRITICAL: Validate model exists and supports generateContent at startup
            # CRITICAL: Validation failures should NOT crash server - Gemini is optional
            try:
                if not self._validate_model_at_startup(selected_model):
                    logger.error(f"❌ Model validation failed: {selected_model}")
                    logger.error(f"   Model does not exist or does not support generateContent")
                    logger.error(f"   ⚠️  Gemini will be disabled - text cleanup will be skipped")
                    self.available = False
                    return
            except Exception as validation_error:
                # CRITICAL: Even validation errors should not crash - Gemini is optional
                logger.error(f"❌ Model validation error: {validation_error}")
                logger.error(f"   ⚠️  Gemini will be disabled - text cleanup will be skipped")
                self.available = False
                return
            
            # Initialize model
            # CRITICAL: Gemini is OPTIONAL - initialization failures should NOT crash server
            try:
                self.client = genai.GenerativeModel(selected_model)
                self.model_name = selected_model
                self.available = True
                logger.info(f"✅ Gemini text cleanup initialized - using {selected_model}")
            except Exception as e:
                # CRITICAL: Never re-raise - Gemini is optional, OCR pipeline must continue
                logger.error(f"❌ [Gemini] Failed to initialize model '{selected_model}': {e}")
                logger.error(f"   ⚠️  Gemini will be disabled - text cleanup will be skipped")
                self.available = False
                # DO NOT re-raise - allow server to start without Gemini
            
        except ImportError:
            logger.info("⚠️  google-generativeai not installed - Gemini text cleanup unavailable")
            logger.info("   Install with: pip install google-generativeai")
            self.available = False
        except Exception as e:
            # CRITICAL: ALL Gemini errors must be survivable - never crash server
            # Gemini is OPTIONAL - OCR pipeline is PRIMARY
            logger.error(f"❌ [Gemini] Initialization error: {e}")
            logger.error(f"   ⚠️  Gemini will be disabled - text cleanup will be skipped")
            self.available = False
            # DO NOT re-raise - allow server to start without Gemini
    
    def _select_available_model(self) -> Optional[str]:
        """
        Select Gemini model based on ACTUAL availability.
        
        Selection logic:
        1. Try models/gemini-2.5-pro (preferred)
        2. If not available, try models/gemini-2.5-flash (fallback)
        3. If neither available, return None (disable Gemini)
        
        Returns:
            Model name if available, None otherwise
        """
        try:
            if not self.genai:
                return None
            
            # Get list of available models
            try:
                models_generator = self.genai.list_models()
                available_models = list(models_generator)
            except Exception as e:
                logger.error(f"   Failed to list models: {e}")
                return None
            
            if not available_models:
                logger.error(f"   No models available from API")
                return None
            
            # Extract available model names
            available_model_names = [getattr(m, 'name', '') for m in available_models if getattr(m, 'name', None)]
            
            logger.info(f"   Checking model availability from {len(available_model_names)} available models...")
            
            # Try preferred models in order
            for preferred_model in PREFERRED_MODELS:
                # Check if model exists in available models
                for model_obj in available_models:
                    model_name = getattr(model_obj, 'name', '')
                    if not model_name:
                        continue
                    
                    # Normalize names for comparison
                    preferred_normalized = preferred_model.replace('models/', '')
                    model_normalized = model_name.replace('models/', '')
                    
                    # Check if this is the model we want
                    if (preferred_model == model_name or 
                        preferred_normalized == model_normalized or
                        model_name.endswith(preferred_normalized) or
                        preferred_normalized in model_name):
                        
                        # Check if it supports generateContent
                        supported_methods = getattr(model_obj, 'supported_generation_methods', [])
                        if 'generateContent' in supported_methods:
                            logger.info(f"   ✅ Found available model: {model_name}")
                            return preferred_model  # Return the preferred name format
                        else:
                            logger.warning(f"   Model '{model_name}' found but does not support generateContent")
            
            # No preferred models available
            logger.error(f"   None of the preferred models are available: {PREFERRED_MODELS}")
            logger.error(f"   Available models (first 10): {available_model_names[:10]}")
            return None
            
        except Exception as e:
            logger.error(f"   Model selection error: {e}")
            import traceback
            logger.debug(f"   Selection traceback: {traceback.format_exc()}")
            return None
    
    def _is_forbidden_model(self, model_name: str) -> bool:
        """
        Check if model name matches any forbidden patterns.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model is forbidden, False otherwise
        """
        model_lower = model_name.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in model_lower:
                return True
        return False
    
    def _validate_model_at_startup(self, model_name: str) -> bool:
        """
        Validate that model exists and supports generateContent.
        Uses list_models() to verify at startup.
        
        CRITICAL: list_models() returns a generator, not a list.
        Must convert to list before indexing or slicing.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if model is valid and supports generateContent, False otherwise
        """
        try:
            if not self.genai:
                return False
            
            # CRITICAL: list_models() returns a generator, not a list
            # Convert to list to avoid "'generator' object is not subscriptable" error
            try:
                models_generator = self.genai.list_models()
                # Convert generator to list safely
                available_models = list(models_generator)
            except Exception as e:
                logger.error(f"   Failed to list models: {e}")
                return False
            
            if not available_models:
                logger.error(f"   No models available from API")
                return False
            
            # Extract model names as strings for comparison
            available_model_names = []
            model_found = False
            supports_generate_content = False
            
            for model in available_models:
                # Model name might be just the name or full path
                model_display_name = getattr(model, 'name', '')
                if model_display_name:
                    available_model_names.append(model_display_name)
                    
                    # Extract base name (last part after /)
                    model_base_name = model_display_name.split('/')[-1] if '/' in model_display_name else model_display_name
                    
                    # Check if this is the model we're looking for
                    # Explicit matching - handle both "models/gemini-2.5-pro" and "gemini-2.5-pro" formats
                    model_name_normalized = model_name.replace('models/', '')
                    model_display_normalized = model_display_name.replace('models/', '')
                    
                    # Match exact name or normalized name
                    if (model_name == model_display_name or 
                        model_name_normalized == model_display_normalized or
                        model_name.endswith(model_base_name) or 
                        model_base_name in model_name or
                        model_display_normalized.endswith(model_name_normalized)):
                        model_found = True
                        # Check if it supports generateContent
                        supported_methods = getattr(model, 'supported_generation_methods', [])
                        if 'generateContent' in supported_methods:
                            supports_generate_content = True
                            logger.info(f"   ✅ Found model '{model_display_name}' (matches '{model_name}')")
                            break
                        else:
                            logger.warning(f"   Model '{model_display_name}' found but does not support generateContent")
            
            if not model_found:
                logger.error(f"   Model '{model_name}' not found in available models")
                logger.error(f"   Available models (first 10): {available_model_names[:10]}")
                logger.error(f"   Total available models: {len(available_model_names)}")
                return False
            
            if not supports_generate_content:
                logger.error(f"   Model '{model_name}' does not support generateContent")
                return False
            
            logger.info(f"   ✅ Model '{model_name}' validated: exists and supports generateContent")
            return True
            
        except Exception as e:
            logger.error(f"   Model validation error: {e}")
            import traceback
            logger.debug(f"   Validation traceback: {traceback.format_exc()}")
            return False
    
    def cleanup_text(
        self,
        raw_text: str
    ) -> str:
        """
        Clean up OCR-extracted text using Gemini API.
        
        CRITICAL RULES:
        - Input: Extracted OCR text (string) - NEVER images
        - Task: Improve text quality (spelling, readability)
        - Preserve technical terms and original meaning
        - Return cleaned text (text-only)
        - If raw_text is empty, return empty string (skip cleanup)
        
        Args:
            raw_text: OCR-extracted text to clean up
            
        Returns:
            Cleaned text string, or empty string if input is empty or cleanup fails
        """
        # CRITICAL: If raw_text is empty, skip Gemini cleanup
        if not raw_text or not raw_text.strip():
            logger.info("   ⚠️  [Gemini Cleanup] Raw text is empty - skipping cleanup")
            return ""
        
        if not self.available or not self.client:
            logger.info("   ⚠️  [Gemini Cleanup] Gemini not available - returning raw text")
            return raw_text
        
        try:
            # Prepare prompt - CRITICAL: TEXT CLEANUP mode
            # Gemini receives OCR text, not images
            prompt = """You are cleaning up OCR-extracted text from handwritten notes.

TASK: Improve the text quality while preserving original meaning.

CRITICAL REQUIREMENTS (ABSOLUTE):
- Fix spelling errors and typos
- Improve readability and clarity
- Preserve ALL technical terms exactly (formulas, equations, scientific terms)
- Preserve mathematical expressions and symbols
- Preserve proper nouns (names, places, technical terms)
- Do NOT change the meaning or content
- Do NOT add explanations or summaries
- Do NOT remove any content
- Preserve line breaks and paragraph structure
- Return ONLY the cleaned text (no markdown, no explanations)

OUTPUT FORMAT:
- Plain text only
- One line per original line
- No markdown, no formatting, no explanations
- Cleaned version of the input text

Example:
Input: "th3 qu1ck br0wn f0x jumprs ov3r th3 l4zy d0g"
Output: "the quick brown fox jumps over the lazy dog"

Remember: CLEAN UP the text, do NOT rewrite or summarize."""

            # Call Gemini API with text only (NO images)
            # CRITICAL: Wrap ALL Gemini API calls in try/except - never re-raise
            logger.info(f"   🔮 [Gemini Cleanup] Cleaning up {len(raw_text)} characters of OCR text...")
            try:
                response = self.client.generate_content([
                    prompt,
                    f"\n\nOCR-extracted text to clean up:\n\n{raw_text}"
                ])
            except Exception as api_error:
                # CRITICAL: API call errors must NEVER interrupt pipeline
                logger.error(f"   ❌ [Gemini Cleanup] API call failed: {api_error}")
                logger.error(f"   ⚠️  [Gemini Cleanup] Returning raw OCR text (cleanup failed)")
                return raw_text  # Return original text if cleanup fails
            
            # Extract cleaned text from response
            # CRITICAL: Wrap response processing in try/except - never re-raise
            try:
                if response and response.text:
                    cleaned_text = response.text.strip()
                    
                    # Remove any markdown formatting if present
                    cleaned_text = cleaned_text.replace('```', '').strip()
                    
                    # Remove prompt artifacts if present
                    if "OCR-extracted text to clean up:" in cleaned_text:
                        cleaned_text = cleaned_text.split("OCR-extracted text to clean up:")[-1].strip()
                    
                    if cleaned_text and cleaned_text.strip():
                        logger.info(f"   ✅ [Gemini Cleanup] Cleaned text: {len(cleaned_text)} characters")
                        return cleaned_text
                    else:
                        logger.warning(f"   ⚠️  [Gemini Cleanup] Cleaned text is empty - returning raw text")
                        return raw_text  # Return original if cleanup produced empty
                else:
                    logger.warning("   ⚠️  [Gemini Cleanup] No text in response - returning raw text")
                    return raw_text  # Return original if no response
            except Exception as response_error:
                # CRITICAL: Response processing errors must NEVER interrupt pipeline
                logger.error(f"   ❌ [Gemini Cleanup] Response processing error: {response_error}")
                logger.error(f"   ⚠️  [Gemini Cleanup] Returning raw OCR text (processing failed)")
                return raw_text  # Return original text if processing fails
                
        except Exception as e:
            # CRITICAL: Gemini errors MUST NEVER block OCR pipeline
            # Log error and return original text
            # NEVER re-raise exceptions - Gemini is optional
            logger.error(f"   ❌ [Gemini Cleanup] Error: {e}")
            logger.error(f"   ⚠️  [Gemini Cleanup] Returning raw OCR text (cleanup failed)")
            return raw_text  # Return original text - pipeline continues


# Global instance
_gemini_cleanup = None


def get_gemini_cleanup() -> GeminiTextCleanup:
    """Get or create GeminiTextCleanup instance."""
    global _gemini_cleanup
    if _gemini_cleanup is None:
        _gemini_cleanup = GeminiTextCleanup()
    return _gemini_cleanup


# Backward compatibility alias (deprecated - use get_gemini_cleanup)
def get_gemini_ocr() -> GeminiTextCleanup:
    """Deprecated: Use get_gemini_cleanup() instead."""
    return get_gemini_cleanup()
