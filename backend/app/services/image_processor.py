import cv2
import numpy as np
from skimage import exposure

class ImageProcessor:
    def __init__(self):
        self.processed_cache = {}
    
    def preprocess(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._deskew(img)
        img = self._correct_lighting(img)
        img = self._enhance_contrast(img)
        img = self._denoise(img)
        
        return img
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:20]:
                angle = theta * 180 / np.pi - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                avg_angle = np.median(angles)
                if abs(avg_angle) > 0.5:
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    image = cv2.warpAffine(
                        image, M, (image.shape[1], image.shape[0]),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
        
        return image
    
    def _correct_lighting(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        image = exposure.rescale_intensity(image, in_range='image', out_range=(0, 255))
        image = image.astype(np.uint8)
        return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )
        return denoised
