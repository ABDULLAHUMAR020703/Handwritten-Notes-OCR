import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class TextRegion:
    bbox: Tuple[int, int, int, int]
    region_type: str
    confidence: float = 0.0

@dataclass
class DiagramRegion:
    bbox: Tuple[int, int, int, int]
    image: np.ndarray

@dataclass
class LayoutResult:
    image: np.ndarray
    text_regions: List[TextRegion]
    diagram_regions: List[DiagramRegion]
    reading_order: List[int]

class LayoutAnalyzer:
    def __init__(self):
        self.min_text_region_area = 500
        self.min_diagram_area = 2000
    
    def analyze(self, image: np.ndarray) -> LayoutResult:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        text_regions = self._detect_text_regions(gray)
        diagram_regions = self._detect_diagram_regions(image, text_regions)
        reading_order = self._determine_reading_order(text_regions)
        
        return LayoutResult(
            image=image,
            text_regions=text_regions,
            diagram_regions=diagram_regions,
            reading_order=reading_order
        )
    
    def _detect_text_regions(self, gray: np.ndarray) -> List[TextRegion]:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > self.min_text_region_area:
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 10 and 20 < h < gray.shape[0] * 0.8:
                    region_type = self._classify_text_region(w, h, y)
                    text_regions.append(TextRegion(
                        bbox=(x, y, w, h),
                        region_type=region_type,
                        confidence=0.8
                    ))
        
        return text_regions
    
    def _classify_text_region(self, width: int, height: int, y_pos: int) -> str:
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 3 and height > 30:
            return 'heading'
        elif aspect_ratio > 5:
            return 'equation'
        else:
            return 'text'
    
    def _detect_diagram_regions(self, image: np.ndarray, text_regions: List[TextRegion]) -> List[DiagramRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        text_mask = np.zeros(gray.shape, dtype=np.uint8)
        for region in text_regions:
            x, y, w, h = region.bbox
            text_mask[y:y+h, x:x+w] = 255
        
        non_text_mask = 255 - text_mask
        contours, _ = cv2.findContours(non_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        diagram_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > self.min_diagram_area:
                pad = 10
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                diagram_image = image[y1:y2, x1:x2]
                
                diagram_regions.append(DiagramRegion(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    image=diagram_image
                ))
        
        return diagram_regions
    
    def _determine_reading_order(self, text_regions: List[TextRegion]) -> List[int]:
        indexed_regions = [(i, region) for i, region in enumerate(text_regions)]
        indexed_regions.sort(key=lambda x: (x[1].bbox[1], x[1].bbox[0]))
        return [i for i, _ in indexed_regions]
