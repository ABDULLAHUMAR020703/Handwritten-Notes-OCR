"""
Enhanced Gemini Vision API integration for smart OCR with diagram detection.
Converts handwritten notes to text, preserving headings, equations, and diagrams.

FEATURES:
- Dynamic model discovery (checks both v1beta and v1 endpoints)
- Direct image-to-text OCR using Gemini Vision API
- Diagram detection with bounding box estimation
- Preserves document structure (headings, paragraphs, equations)
"""
import os
import logging
import requests
import json
import base64
import re
from typing import Optional, Tuple, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class SmartGeminiOCR:
    """
    Enhanced Gemini Vision API integration for smart OCR.
    
    Key capabilities:
    1. Dynamic model discovery for optimal performance
    2. Smart diagram detection with position markers
    3. Enhanced OCR for handwritten text
    4. Preserves headings, equations, and document structure
    """
    
    def __init__(self):
        """Initialize enhanced Gemini OCR service with dynamic model discovery."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.available = False
        self.model_name = None
        self.api_base_url = None
        
        if not self.api_key:
            logger.info("ℹ️ GEMINI_API_KEY not found - Smart Gemini OCR disabled")
            return
        
        # Dynamic model discovery
        self.model_name = self._discover_best_model()
        if self.model_name:
            self.available = True
            logger.info(f"✅ Smart Gemini OCR initialized with model: {self.model_name}")
        else:
            logger.warning("⚠️ No compatible Gemini models found")
            self.available = False
    
    def _discover_best_model(self) -> Optional[str]:
        """
        Discover the best available Gemini model for OCR tasks.
        Checks both v1beta and v1 endpoints for available models.
        
        Returns:
            Model name string (without 'models/' prefix) or None if no suitable model found
        """
        logger.info("🔍 Checking Gemini API Key permissions...")
        
        # Try both Beta and Stable endpoints
        endpoints = [
            "https://generativelanguage.googleapis.com/v1beta/models",
            "https://generativelanguage.googleapis.com/v1/models"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{endpoint}?key={self.api_key}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"   ✅ Connected to {endpoint}")
                    
                    # Filter for models that support generating content
                    for model in data.get('models', []):
                        name = model.get('name', '')  # Comes as "models/gemini-..."
                        methods = model.get('supportedGenerationMethods', [])
                        
                        if 'generateContent' in methods:
                            # Prioritize Vision models (Flash/Pro)
                            if 'flash' in name or 'pro' in name or 'vision' in name:
                                # Store the API base URL that worked
                                self.api_base_url = endpoint.replace('/models', '')
                                clean_name = name.replace('models/', '')
                                logger.info(f"   🎯 Found compatible model: {clean_name}")
                                return clean_name
                                
            except requests.exceptions.RequestException as e:
                logger.debug(f"   ⚠️ Connection check failed for {endpoint}: {e}")
                continue
        
        logger.error("   ❌ No compatible Gemini models found")
        return None
    
    def extract_text_from_image(
        self,
        image_path: str,
        image_width: int = None,
        image_height: int = None
    ) -> Tuple[str, List[Dict]]:
        """
        Extract text from an image using Gemini Vision API.
        
        Args:
            image_path: Path to the image file
            image_width: Width of the image (for estimating diagram positions)
            image_height: Height of the image (for estimating diagram positions)
            
        Returns:
            Tuple of (extracted_text, diagram_regions)
            - extracted_text: The transcribed text with [[DIAGRAM_n]] placeholders
            - diagram_regions: List of estimated diagram region info
        """
        if not self.available:
            logger.warning("⚠️ Smart Gemini OCR not available")
            return "", []
        
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine MIME type
            ext = Path(image_path).suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # Build API URL
            url = f"{self.api_base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
            headers = {'Content-Type': 'application/json'}
            
            # Get the OCR prompt
            prompt_text = self._get_ocr_prompt()
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_image
                            }
                        }
                    ]
                }]
            }
            
            logger.info(f"   📤 Sending image to Gemini Vision API...")
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            
            if response.status_code != 200:
                logger.error(f"   ⚠️ API Error {response.status_code}: {response.text}")
                return "", []
            
            result_json = response.json()
            if 'candidates' in result_json and result_json['candidates']:
                full_text = result_json['candidates'][0]['content']['parts'][0]['text']
                logger.info(f"   ✅ Extracted {len(full_text)} characters from image")
                
                # Parse and normalize the text, extract diagram positions
                clean_text, diagram_regions = self._parse_response(full_text, image_width, image_height)
                
                return clean_text, diagram_regions
            else:
                logger.warning("   ⚠️ AI returned no text. Image might be blocked.")
                return "", []
                
        except Exception as e:
            logger.error(f"   ❌ Error in Gemini Vision OCR: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return "", []
    
    def _get_ocr_prompt(self) -> str:
        """Get OCR prompt for handwritten notes conversion."""
        return """You are a smart document transcriber specializing in handwritten notes.

TASK: Convert these handwritten notes into a clean, structured text format.

CRITICAL INSTRUCTIONS:

1. TEXT STRUCTURE:
   - Use "## " prefix for main headings/titles
   - Use "### " prefix for subheadings
   - Preserve paragraph breaks
   - Keep bullet points and numbered lists as-is

2. EQUATIONS & FORMULAS:
   - Write math formulas in LaTeX format enclosed in $..$ for inline or $$...$$ for display
   - Examples: $E = mc^2$, $\\frac{x}{y}$, $\\alpha + \\beta$
   - For simple expressions, plain text is fine: "x = 5"

3. DIAGRAMS, CHARTS, GRAPHS, DRAWINGS:
   - When you see a diagram, chart, graph, or any visual element:
   - Insert the marker: [[DIAGRAM]] on its own line
   - Also provide position hint: [[DIAGRAM:position=X%]] where X is vertical % position in document
   - Continue transcribing text after the diagram

4. PRESERVE EVERYTHING:
   - Don't skip any content
   - Fix obvious spelling errors
   - Keep technical terms exactly as written

OUTPUT FORMAT:
- Clean structured text
- Headings with ## or ###
- Math in $...$ or $$...$$
- [[DIAGRAM:position=X%]] markers for visuals
"""
    
    def _parse_response(
        self,
        text: str,
        image_width: int = None,
        image_height: int = None
    ) -> Tuple[str, List[Dict]]:
        """
        Parse Gemini response to extract clean text and diagram regions.
        
        Args:
            text: Raw text from Gemini API
            image_width: Width of source image
            image_height: Height of source image
            
        Returns:
            Tuple of (clean_text, diagram_regions)
        """
        diagram_regions = []
        diagram_idx = 0
        
        # Find all diagram markers
        pattern = r'\[\[DIAGRAM(?::position=(\d+)%)?\]\]'
        
        def replace_diagram(match):
            nonlocal diagram_idx
            position_pct = match.group(1)
            position = int(position_pct) if position_pct else 50  # Default to middle
            
            # Estimate bounding box if we have image dimensions
            if image_height:
                # Estimate vertical position
                y_start = int((position / 100.0) * image_height * 0.8)  # Leave some margin
                y_end = min(y_start + int(image_height * 0.3), image_height)  # ~30% height for diagram
            else:
                y_start = 0
                y_end = 0
            
            diagram_regions.append({
                'index': diagram_idx,
                'position_percent': position,
                'estimated_bbox': {
                    'x1': 0,
                    'y1': y_start,
                    'x2': image_width or 0,
                    'y2': y_end
                } if image_height else None
            })
            
            placeholder = f"[[DIAGRAM_{diagram_idx}]]"
            diagram_idx += 1
            return placeholder
        
        clean_text = re.sub(pattern, replace_diagram, text)
        
        # Clean up any remaining formatting issues
        clean_text = clean_text.strip()
        
        return clean_text, diagram_regions


# Global instance
_smart_gemini_ocr = None


def get_smart_gemini_ocr() -> SmartGeminiOCR:
    """Get or create SmartGeminiOCR instance."""
    global _smart_gemini_ocr
    if _smart_gemini_ocr is None:
        _smart_gemini_ocr = SmartGeminiOCR()
    return _smart_gemini_ocr