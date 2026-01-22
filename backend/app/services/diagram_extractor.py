"""
Dedicated Diagram Extraction Service.
Focuses exclusively on detecting and extracting visual elements (charts, graphs, drawings)
from document images, separate from text layout analysis.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DiagramExtractor:
    """
    Specialized computer vision pipeline for detecting diagrams in handwritten notes.
    """
    
    def __init__(self):
        pass
        
    def extract_diagrams(self, image_path: str) -> List[Dict]:
        """
        Detect diagram regions in the image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of diagram dictionaries with 'bbox' [x1, y1, x2, y2] and 'confidence'
        """
        logger.info(f"🎨 Starting dedicated diagram extraction for: {Path(image_path).name}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"❌ Could not load image: {image_path}")
            return []
            
        h, w = img.shape[:2]
        
        # 1. Preprocessing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # 2. Edge Detection (Canny)
        # Finds all structural edges (text strokes + diagram lines)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # 3. Text Removal Mask
        # We want to remove things that look like lines of text
        # Adaptive threshold to find all ink
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to identify text lines
        # Long horizontal kernel to connect characters into lines
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
        text_lines = cv2.dilate(binary, kernel_text, iterations=1)
        
        # Subtract text lines from edges
        # This leaves "non-linear" or "vertical" or "complex" structures
        # Note: This is a heuristic. Some diagram labels will be removed, which is fine.
        # We want the SHAPE of the diagram.
        
        # Dilate edges to make them solid
        kernel_edges = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel_edges, iterations=2)
        
        # 4. Find Diagram Candidates
        # Combine edges and exclude known text line areas
        # (We invert text_lines because we want to KEEP non-text areas)
        # Actually, let's look for "dense edge regions" that are NOT just horizontal lines
        
        # Alternative approach: Connected Components on Edges
        # Diagrams are usually large connected components of edges
        
        # Morphological closing to connect diagram parts
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours on the "closed" edge map
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        diagrams = []
        
        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            area = w_rect * h_rect
            
            # Filter 1: Size
            # Must be significant size (> 5% of page)
            if area < (w * h) * 0.05:
                continue
                
            # Filter 2: Background
            # Must not be the whole page (> 90%)
            if area > (w * h) * 0.90:
                continue
                
            # Filter 3: Aspect Ratio
            # Diagrams are usually somewhat square-ish (0.3 to 3.0)
            aspect_ratio = w_rect / h_rect
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Filter 4: Density Check
            # Check the original binary image within this bbox
            roi = binary[y:y+h_rect, x:x+w_rect]
            density = cv2.countNonZero(roi) / area
            
            # Text blocks are usually dense (0.3-0.6).
            # Diagrams can be sparse (line drawings) or dense (filled charts).
            # Very low density (< 0.05) is likely noise/empty.
            # Very high density (> 0.8) is likely a photo or dark block.
            if density < 0.02:
                continue
                
            # Filter 5: Text Line Structure
            # Check if the region is composed mainly of horizontal lines (text paragraph)
            roi_text_lines = text_lines[y:y+h_rect, x:x+w_rect]
            text_line_density = cv2.countNonZero(roi_text_lines) / area
            
            # If the region is almost entirely covered by "text line" detection, it's likely a paragraph
            # Diagrams will have vertical lines, circles, etc. that don't match the horizontal kernel perfectly
            # But this is risky if the diagram has many labels.
            
            # Instead, let's accept it as a diagram candidate
            
            diagrams.append({
                'bbox': [x, y, x + w_rect, y + h_rect],
                'confidence': 0.8, # Placeholder
                'type': 'diagram'
            })
            logger.info(f"   🎨 Detected diagram candidate: {w_rect}x{h_rect} at ({x},{y})")
            
        # 5. Merge Overlapping Candidates
        merged_diagrams = self._merge_overlaps(diagrams)
        
        logger.info(f"   ✅ Found {len(merged_diagrams)} diagram regions")
        return merged_diagrams
        
    def _merge_overlaps(self, diagrams: List[Dict]) -> List[Dict]:
        """Merge overlapping diagram regions."""
        if not diagrams:
            return []
            
        # Sort by area (largest first)
        sorted_diagrams = sorted(diagrams, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]), reverse=True)
        
        merged = []
        
        for current in sorted_diagrams:
            is_overlap = False
            cx1, cy1, cx2, cy2 = current['bbox']
            c_area = (cx2-cx1) * (cy2-cy1)
            
            for existing in merged:
                ex1, ey1, ex2, ey2 = existing['bbox']
                
                # Calculate intersection
                ix1 = max(cx1, ex1)
                iy1 = max(cy1, ey1)
                ix2 = min(cx2, ex2)
                iy2 = min(cy2, ey2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    # If intersection is significant (> 50% of smaller rect), merge
                    if intersection > 0.5 * min(c_area, (ex2-ex1)*(ey2-ey1)):
                        # Merge into existing (expand existing bbox)
                        existing['bbox'] = [
                            min(cx1, ex1),
                            min(cy1, ey1),
                            max(cx2, ex2),
                            max(cy2, ey2)
                        ]
                        is_overlap = True
                        break
            
            if not is_overlap:
                merged.append(current)
                
        return merged

# Singleton instance
diagram_extractor = DiagramExtractor()
