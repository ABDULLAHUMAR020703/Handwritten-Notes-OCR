"""
Free open-source layout detection system for document analysis.
Uses computer vision techniques to detect and classify regions:
- heading, paragraph, equation, diagram
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


def detect_layout(image_path: str) -> List[Dict[str, Any]]:
    """
    Detect and classify document regions with preserved reading order.
    
    Args:
        image_path: Path to preprocessed image
        
    Returns:
        List of region dictionaries with type and bbox [x1, y1, x2, y2]
    """
    logger.info(f"📐 Starting layout detection for: {Path(image_path).name}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"❌ Could not load image: {image_path}")
        raise ValueError(f"Could not load image: {image_path}")
    
    logger.debug(f"   Image dimensions: {img.shape[1]}x{img.shape[0]}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Detect all regions
    logger.debug("🔍 Detecting regions...")
    regions = _detect_regions(gray, img)
    logger.info(f"   Found {len(regions)} regions")
    
    # Classify regions
    logger.debug("🏷️  Classifying regions...")
    classified_regions = _classify_regions(regions, gray, img)
    
    # Count by type
    type_counts = {}
    for r in classified_regions:
        type_counts[r['type']] = type_counts.get(r['type'], 0) + 1
    logger.info(f"   Region types: {type_counts}")
    
    # Sort by reading order (top-to-bottom, left-to-right)
    logger.debug("📋 Sorting regions by reading order...")
    sorted_regions = _sort_reading_order(classified_regions)
    
    # Format output
    output = []
    for region in sorted_regions:
        x1, y1, x2, y2 = region['bbox']
        output.append({
            "type": region['type'],
            "bbox": [x1, y1, x2, y2]
        })
    
    return output


def _detect_regions(gray: np.ndarray, img: np.ndarray) -> List[Dict]:
    """
    Detect all document regions using contour analysis.
    """
    regions = []
    
    # Apply adaptive thresholding for better text detection
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to connect text components
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Horizontal dilation for text lines
    dilated_h = cv2.dilate(binary, kernel_horizontal, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = gray.shape
    # RELAXED thresholds: Much lower minimum area (0.01% instead of 0.1%)
    # This allows detection of smaller text regions
    min_area = (w * h) * 0.0001  # 0.01% of image area (was 0.1%)
    max_area = (w * h) * 0.95   # 95% of image area
    
    for contour in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        area = w_rect * h_rect
        
        # Filter by size - RELAXED thresholds
        if min_area < area < max_area:
            # Filter by aspect ratio and dimensions - RELAXED
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            
            # RELAXED: Lower height threshold (5 instead of 10), wider aspect ratio range
            # Valid text regions should have reasonable aspect ratios
            if 0.05 < aspect_ratio < 100 and h_rect > 5:  # Was: 0.1 < aspect_ratio < 50 and h_rect > 10
                x1, y1 = x, y
                x2, y2 = x + w_rect, y + h_rect
                
                regions.append({
                    'bbox': (x1, y1, x2, y2),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'width': w_rect,
                    'height': h_rect,
                    'x': x,
                    'y': y
                })
    
    # Detect diagram regions (non-text areas)
    diagram_regions = _detect_diagram_regions(gray, regions)
    regions.extend(diagram_regions)
    
    return regions


def _detect_diagram_regions(gray: np.ndarray, text_regions: List[Dict]) -> List[Dict]:
    """
    Detect diagram/image regions by finding large non-text areas.
    
    CRITICAL: This function is EXTREMELY conservative.
    - Only detects regions with NO readable characters
    - Much stricter size thresholds
    - If in doubt, returns empty list (treat as text)
    """
    h, w = gray.shape
    text_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark text regions
    for region in text_regions:
        x1, y1, x2, y2 = region['bbox']
        text_mask[y1:y2, x1:x2] = 255
    
    # Invert to find non-text regions
    non_text = 255 - text_mask
    
    # Find large connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        non_text, connectivity=8
    )
    
    diagram_regions = []
    # Much stricter threshold - only very large regions (5% instead of 2%)
    min_diagram_area = (w * h) * 0.05  # 5% of image area
    
    # Try to import OCR for character detection
    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        ocr_available = True
        if use_gpu:
            logger.debug("   Using GPU for diagram detection OCR check")
    except:
        ocr_reader = None
        ocr_available = False
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_diagram_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_rect = stats[i, cv2.CC_STAT_WIDTH]
            h_rect = stats[i, cv2.CC_STAT_HEIGHT]
            
            x1, y1 = x, y
            x2, y2 = x + w_rect, y + h_rect
            
            # Avoid very elongated regions (likely margins/gaps)
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            
            # Much stricter aspect ratio - only square-ish regions
            if 0.7 < aspect_ratio < 1.5:  # Very square (was 0.2-5)
                # CRITICAL: Check for readable characters before adding as diagram
                region_img = gray[y1:y2, x1:x2]
                has_characters = False
                
                if ocr_available and ocr_reader and region_img.size > 0:
                    try:
                        # Quick OCR check
                        check_img = region_img.copy()
                        if check_img.shape[0] > 200 or check_img.shape[1] > 200:
                            scale = min(200 / check_img.shape[0], 200 / check_img.shape[1])
                            new_h = int(check_img.shape[0] * scale)
                            new_w = int(check_img.shape[1] * scale)
                            check_img = cv2.resize(check_img, (new_w, new_h))
                        
                        results = ocr_reader.readtext(check_img, detail=0)
                        for text in results:
                            if text and text.strip():
                                text_clean = text.strip()
                                if any(c.isalnum() for c in text_clean) or any(c in '+-=×÷∑∫√≤≥≠≈' for c in text_clean):
                                    has_characters = True
                                    break
                    except:
                        # If check fails, assume text (conservative)
                        has_characters = True
                
                # Only add as diagram if NO characters detected
                if not has_characters:
                    diagram_regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'width': w_rect,
                        'height': h_rect,
                        'x': x,
                        'y': y
                    })
                    logger.debug(f"   Detected diagram region (no characters, area={area/(w*h)*100:.1f}%)")
                else:
                    logger.debug(f"   Skipped potential diagram (contains characters) → treating as text")
    
    logger.info(f"   Detected {len(diagram_regions)} diagram regions (conservative detection)")
    return diagram_regions


def _classify_regions(regions: List[Dict], gray: np.ndarray, img: np.ndarray) -> List[Dict]:
    """
    Classify regions into: heading, paragraph, equation, diagram.
    
    CRITICAL: Diagram detection is EXTREMELY conservative.
    - Only classify as diagram if NO readable characters detected
    - If region contains letters, numbers, or math symbols → treat as TEXT
    - Handwritten text irregularity does NOT mean diagram
    - If in doubt → treat as TEXT, not diagram
    """
    classified = []
    h, w = gray.shape
    
    # Try to import OCR for character detection (optional, fail gracefully)
    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        ocr_available = True
        if use_gpu:
            logger.debug("   Using GPU for region classification OCR check")
    except:
        ocr_reader = None
        ocr_available = False
        logger.debug("   OCR not available for diagram detection - will use conservative heuristics only")
    
    for region in regions:
        x1, y1, x2, y2 = region['bbox']
        
        # Extract region properties
        region_height = region['height']
        region_width = region['width']
        aspect_ratio = region['aspect_ratio']
        area = region['area']
        y_position = region['y']
        
        # Extract region image for analysis
        region_img = gray[y1:y2, x1:x2]
        
        # CRITICAL: Check for readable characters BEFORE classifying as diagram
        # If ANY characters detected → treat as TEXT, not diagram
        has_readable_characters = False
        
        if ocr_available and ocr_reader and region_img.size > 0:
            try:
                # Quick OCR check for any readable characters
                # Use a small sample to avoid performance issues
                # Resize if too large for faster processing
                check_img = region_img.copy()
                if check_img.shape[0] > 200 or check_img.shape[1] > 200:
                    scale = min(200 / check_img.shape[0], 200 / check_img.shape[1])
                    new_h = int(check_img.shape[0] * scale)
                    new_w = int(check_img.shape[1] * scale)
                    check_img = cv2.resize(check_img, (new_w, new_h))
                
                # Run OCR to check for characters
                results = ocr_reader.readtext(check_img, detail=0)
                
                # Check if any readable text was found
                for text in results:
                    if text and text.strip():
                        # Check if it contains letters, numbers, or common math symbols
                        text_clean = text.strip()
                        # Look for alphanumeric characters or math symbols
                        if any(c.isalnum() for c in text_clean) or any(c in '+-=×÷∑∫√≤≥≠≈' for c in text_clean):
                            has_readable_characters = True
                            logger.debug(f"   Region contains readable characters: '{text_clean[:30]}...' → treating as TEXT")
                            break
            except Exception as e:
                # If OCR check fails, assume text (conservative)
                logger.debug(f"   OCR check failed for region, defaulting to TEXT: {e}")
                has_readable_characters = True  # Default to text if check fails
        
        # Calculate text density (using horizontal projections)
        horizontal_projection = np.sum(region_img < 128, axis=1)
        text_density = np.mean(horizontal_projection > region_width * 0.1)
        
        # Calculate vertical line density (for equations/diagrams)
        vertical_projection = np.sum(region_img < 128, axis=0)
        vertical_density = np.mean(vertical_projection > region_height * 0.1)
        
        # Calculate compactness (for diagrams)
        region_area = region_height * region_width
        compactness = area / region_area if region_area > 0 else 0
        
        # EXTREMELY CONSERVATIVE diagram detection
        # Only classify as diagram if:
        # 1. NO readable characters detected (checked above)
        # 2. AND meets strict size/shape criteria
        # 3. AND very low text density
        
        # Much stricter thresholds for diagram classification
        is_diagram = False
        if not has_readable_characters:  # CRITICAL: Only if NO characters detected
            # Very strict criteria - only large, square-ish regions with very low text density
            is_diagram = (
                area > (w * h) * 0.10 and  # Much larger threshold (10% instead of 5%)
                0.7 < aspect_ratio < 1.5 and  # Very square (not elongated)
                compactness < 0.15 and  # Very low text density (was 0.3)
                text_density < 0.1  # Additional check: very low text density
            )
            
            if is_diagram:
                logger.debug(f"   Classified as diagram (no characters, area={area/(w*h)*100:.1f}%, compactness={compactness:.2f})")
        
        # If in doubt, treat as TEXT (not diagram)
        if has_readable_characters:
            is_diagram = False  # Force text if characters detected
        
        if is_diagram:
            region['type'] = 'diagram'
        else:
            # Classify text regions
            # Headings are typically:
            # - Wider than tall (high aspect ratio)
            # - At top of document or start of new sections
            # - Taller than regular text (but not too tall)
            # - Have high text density
            # - Often centered or have special formatting
            y_percentile = y_position / h if h > 0 else 1.0
            
            is_heading = (
                aspect_ratio > 2.5 and  # Lowered threshold for better detection
                region_height > 20 and region_height < 80 and  # More flexible height range
                (y_percentile < 0.4 or (y_percentile > 0.3 and aspect_ratio > 4.0)) and  # Top section or very wide
                text_density > 0.4  # Lowered threshold
            )
            
            # Additional check: very wide regions with medium height are likely headings
            if not is_heading and aspect_ratio > 4.0 and 20 < region_height < 60:
                is_heading = True
            
            # Equations are typically:
            # - Wide (high aspect ratio)
            # - Short (low height)
            # - Have special symbols/characters
            # - Medium text density
            is_equation = (
                aspect_ratio > 5.0 and
                region_height < 60 and
                region_height > 15 and
                0.3 < text_density < 0.7
            )
            
            if is_heading:
                region['type'] = 'heading'
            elif is_equation:
                region['type'] = 'equation'
            else:
                region['type'] = 'paragraph'
        
        classified.append(region)
    
    return classified


def _sort_reading_order(regions: List[Dict]) -> List[Dict]:
    """
    Sort regions by reading order: top-to-bottom, left-to-right.
    Uses a sophisticated algorithm to handle multi-column layouts.
    """
    if not regions:
        return []
    
    # Group regions by approximate row (y-coordinate)
    # Regions with similar y-coordinates are in the same row
    y_threshold = 30  # Pixels
    
    # Sort by y-coordinate first
    sorted_by_y = sorted(regions, key=lambda r: r['y'])
    
    # Group into rows
    rows = []
    current_row = [sorted_by_y[0]]
    
    for region in sorted_by_y[1:]:
        # Check if region is in the same row (similar y-coordinate)
        avg_y_current = np.mean([r['y'] + r['height']/2 for r in current_row])
        region_center_y = region['y'] + region['height']/2
        
        if abs(region_center_y - avg_y_current) < y_threshold:
            current_row.append(region)
        else:
            # Start new row
            rows.append(current_row)
            current_row = [region]
    
    # Don't forget the last row
    if current_row:
        rows.append(current_row)
    
    # Sort each row by x-coordinate (left-to-right)
    final_order = []
    for row in rows:
        row_sorted = sorted(row, key=lambda r: r['x'])
        final_order.extend(row_sorted)
    
    return final_order


def detect_layout_from_array(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect layout from numpy array image.
    Useful when image is already loaded in memory.
    """
    # Save temporarily
    import tempfile
    from pathlib import Path
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    cv2.imwrite(temp_file.name, image)
    temp_path = temp_file.name
    
    try:
        result = detect_layout(temp_path)
    finally:
        # Clean up temp file
        Path(temp_path).unlink()
    
    return result
