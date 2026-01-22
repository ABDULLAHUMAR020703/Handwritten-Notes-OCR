"""
Image preprocessing module optimized for handwritten notes.
Handles: grayscale conversion, denoising, contrast enhancement, deskewing, resizing.
"""
import cv2
import numpy as np
from pathlib import Path
import tempfile
from skimage import exposure
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> str:
    """
    Preprocess notebook image for optimal OCR performance.
    
    Args:
        image_path: Path to input image (JPEG/PNG)
        
    Returns:
        Path to processed image (temporarily saved)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Remove noise
    denoised = _remove_noise(gray)
    
    # Step 3: Increase contrast
    contrasted = _increase_contrast(denoised)
    
    # Step 4: Correct skew (deskew) - gracefully handle failures
    deskewed = _deskew_image(contrasted)
    
    # Step 5: Resize to optimal OCR resolution (preserving aspect ratio)
    resized = _resize_for_ocr(deskewed)
    
    # Save processed image temporarily
    processed_path = _save_temp_image(resized, image_path)
    
    return processed_path


def _remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Remove noise while preserving text edges.
    Uses adaptive filtering suitable for handwritten text.
    """
    try:
        # Non-local means denoising - better for preserving text details
        denoised = cv2.fastNlMeansDenoising(
            image, None, h=10, templateWindowSize=7, searchWindowSize=21
        )
        # Additional bilateral filter for handwritten notes (reduces noise, preserves edges)
        denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
        return denoised
    except Exception as e:
        logger.warning(f"Noise removal failed: {e}, using original image")
        return image


def _increase_contrast(image: np.ndarray) -> np.ndarray:
    """
    Increase contrast using adaptive histogram equalization.
    Optimized for varying lighting conditions in notebook photos.
    """
    try:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Increased clip limit for better contrast in handwritten notes
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Additional contrast stretching
        enhanced = exposure.rescale_intensity(enhanced, in_range='image', out_range=(0, 255))
        enhanced = enhanced.astype(np.uint8)
        
        return enhanced
    except Exception as e:
        logger.warning(f"Contrast enhancement failed: {e}, using original image")
        return image


def _deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Correct image rotation/skew using Hough line detection.
    Critical for handwritten notes which often have rotation.
    Gracefully handles failures by returning original image.
    """
    try:
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        # Handle OpenCV version differences safely
        try:
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        except Exception as e:
            logger.warning(f"HoughLines failed: {e}, skipping deskew")
            return image
        
        if lines is None or len(lines) == 0:
            # No lines detected, return original image
            return image
        
        # Safely extract line data
        try:
            # Calculate angles from detected lines
            angles = []
            lines_to_check = min(50, len(lines))  # Limit to first 50 lines
            
            for i in range(lines_to_check):
                try:
                    line = lines[i]
                    # Handle both OpenCV 3.x (array) and 4.x (ndarray) formats
                    if isinstance(line, np.ndarray):
                        if line.ndim == 1:
                            rho, theta = float(line[0]), float(line[1])
                        else:
                            rho, theta = float(line[0][0]), float(line[0][1])
                    else:
                        # Fallback for other formats
                        rho, theta = float(line[0]), float(line[1])
                    
                    angle = theta * 180 / np.pi - 90
                    # Only consider reasonable angles
                    if -45 < angle < 45:
                        angles.append(angle)
                except (IndexError, ValueError, TypeError) as e:
                    logger.debug(f"Error processing line {i}: {e}, skipping")
                    continue
            
            if not angles or len(angles) == 0:
                return image
            
            # Use median angle for robustness against outliers
            median_angle = np.median(angles)
            
            # Only rotate if skew is significant (> 0.3 degrees) - more sensitive for handwritten notes
            if abs(median_angle) <= 0.3:
                return image
            
            # Get image center
            h, w = image.shape
            center = (w // 2, h // 2)
            
            # Create rotation matrix
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            
            # Calculate new dimensions to fit rotated image
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust rotation matrix for new dimensions
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Apply rotation
            rotated = cv2.warpAffine(
                image, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return rotated
            
        except Exception as e:
            logger.warning(f"Deskew angle calculation failed: {e}, returning original image")
            return image
            
    except Exception as e:
        logger.warning(f"Deskew operation failed: {e}, returning original image")
        return image


def _resize_for_ocr(image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
    """
    Resize image to optimal OCR resolution while preserving aspect ratio.
    
    OCR models typically work best at 300 DPI or ~2400 pixels on the long side.
    This preserves readability while keeping processing efficient.
    """
    try:
        # Safely get image dimensions
        if image.ndim == 2:
            h, w = image.shape
        elif image.ndim == 3:
            h, w = image.shape[:2]
        else:
            logger.warning(f"Unexpected image dimensions: {image.ndim}, returning original")
            return image
        
        long_side = max(h, w)
        
        # Optimal long side for OCR: 2400 pixels (300 DPI for 8-inch page)
        optimal_long_side = 2400
        
        # Resize based on current size
        if long_side != optimal_long_side:
            # Calculate scale factor to reach optimal size
            scale = optimal_long_side / long_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Validate dimensions
            if new_w <= 0 or new_h <= 0:
                logger.warning(f"Invalid resize dimensions: {new_w}x{new_h}, returning original")
                return image
            
            # Choose interpolation based on upscaling or downscaling
            if scale > 1:
                # Upscaling - use cubic for better quality
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                # Downscaling - use area for better quality
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return resized
        else:
            # Already at optimal size
            return image.copy()
            
    except Exception as e:
        logger.warning(f"Resize operation failed: {e}, returning original image")
        return image


def _save_temp_image(image: np.ndarray, original_path: str) -> str:
    """
    Save processed image to temporary file.
    Preserves original filename with '_processed' suffix.
    """
    try:
        # Validate image
        if image is None or image.size == 0:
            raise ValueError("Invalid image to save")
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        original_path_obj = Path(original_path)
        temp_dir = Path(tempfile.gettempdir()) / "ocr_preprocessing"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Create processed filename
        processed_filename = f"{original_path_obj.stem}_processed.png"
        processed_path = temp_dir / processed_filename
        
        # Save as PNG (lossless)
        success = cv2.imwrite(str(processed_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        if not success:
            raise ValueError(f"Failed to save image to {processed_path}")
        
        if not processed_path.exists():
            raise FileNotFoundError(f"Image file was not created at {processed_path}")
        
        return str(processed_path)
        
    except Exception as e:
        logger.error(f"Failed to save processed image: {e}")
        # Fallback: save to same directory as original with a different name
        try:
            original_path_obj = Path(original_path)
            fallback_path = original_path_obj.parent / f"{original_path_obj.stem}_processed_fallback.png"
            cv2.imwrite(str(fallback_path), image)
            return str(fallback_path)
        except Exception as e2:
            logger.error(f"Fallback save also failed: {e2}")
            # Last resort: return original path
            logger.warning("Returning original image path as last resort")
            return original_path
