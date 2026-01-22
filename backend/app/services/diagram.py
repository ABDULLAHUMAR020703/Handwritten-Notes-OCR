"""
Diagram extraction and processing module.
Extracts diagram regions, saves as images, and optionally converts to SVG.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import tempfile
import os
from PIL import Image


class DiagramExtractor:
    """
    Extract and process diagram regions from documents.
    Diagrams are saved as images (not converted to text).
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize diagram extractor.
        
        Args:
            output_dir: Directory to save extracted diagrams
        """
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "diagrams"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if Potrace is available for SVG conversion
        self.potrace_available = self._check_potrace()
        if self.potrace_available:
            print("Potrace available for SVG conversion")
        else:
            print("Potrace not available (SVG conversion disabled)")
    
    def extract_diagrams(
        self,
        image: np.ndarray,
        diagram_regions: List[Dict],
        base_filename: str = "diagram"
    ) -> List[Dict]:
        """
        Extract diagram regions from image and save them.
        
        Args:
            image: Full document image
            diagram_regions: List of diagram regions with bbox [x1, y1, x2, y2]
            base_filename: Base filename for saved diagrams
            
        Returns:
            List of dictionaries with diagram info:
            {
                "image_path": str,
                "svg_path": Optional[str],
                "bbox": [x1, y1, x2, y2],
                "position": {"x": int, "y": int}
            }
        """
        extracted_diagrams = []
        
        for idx, region in enumerate(diagram_regions):
            bbox = region.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Ensure valid coordinates
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1, min(x2, w))
            y2 = max(y1, min(y2, h))
            
            # Crop diagram region
            diagram_image = image[y1:y2, x1:x2]
            
            if diagram_image.size == 0:
                continue
            
            # Process and save diagram
            diagram_info = self._process_diagram(
                diagram_image,
                bbox,
                base_filename,
                idx
            )
            
            extracted_diagrams.append(diagram_info)
        
        return extracted_diagrams
    
    def _process_diagram(
        self,
        diagram_image: np.ndarray,
        bbox: List[int],
        base_filename: str,
        index: int
    ) -> Dict:
        """
        Process a single diagram: save as image and optionally convert to SVG.
        
        Args:
            diagram_image: Cropped diagram image
            bbox: Bounding box [x1, y1, x2, y2]
            base_filename: Base filename
            index: Diagram index
            
        Returns:
            Dictionary with diagram information
        """
        x1, y1, x2, y2 = bbox
        
        # Enhance diagram image quality
        enhanced_image = self._enhance_diagram_image(diagram_image)
        
        # Generate filename
        diagram_filename = f"{base_filename}_diagram_{index:03d}.png"
        diagram_path = self.output_dir / diagram_filename
        
        # Save as PNG (lossless)
        cv2.imwrite(str(diagram_path), enhanced_image)
        
        # Prepare result
        result = {
            "image_path": str(diagram_path),
            "svg_path": None,
            "bbox": bbox,
            "position": {
                "x": x1,
                "y": y1
            },
            "width": x2 - x1,
            "height": y2 - y1
        }
        
        # Optional: Convert to SVG using Potrace
        if self.potrace_available:
            try:
                svg_path = self._convert_to_svg(enhanced_image, base_filename, index)
                if svg_path and Path(svg_path).exists():
                    result["svg_path"] = svg_path
            except Exception as e:
                print(f"SVG conversion failed for diagram {index}: {e}")
        
        return result
    
    def _enhance_diagram_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance diagram image for better quality.
        
        Args:
            image: Original diagram image
            
        Returns:
            Enhanced image
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Convert to RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast slightly (but preserve colors)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only (preserves colors)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to BGR for saving
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr
    
    def _convert_to_svg(
        self,
        image: np.ndarray,
        base_filename: str,
        index: int
    ) -> Optional[str]:
        """
        Convert diagram image to SVG using Potrace.
        
        Args:
            image: Diagram image
            base_filename: Base filename
            index: Diagram index
            
        Returns:
            Path to SVG file or None if conversion failed
        """
        try:
            import subprocess
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Save as temporary PBM (Portable Bitmap) for Potrace
            pbm_filename = f"{base_filename}_diagram_{index:03d}.pbm"
            pbm_path = self.output_dir / pbm_filename
            
            # Convert to binary (black and white) for better vectorization
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Save as PBM
            # PBM format: P4 (binary PBM) - simpler for Potrace
            h, w = binary.shape
            with open(pbm_path, 'wb') as f:
                # Write PBM header
                f.write(f"P4\n{w} {h}\n".encode())
                # Write binary data
                # Invert (Potrace expects black on white)
                inverted = 255 - binary
                # Pack bits (PBM format uses 1 bit per pixel)
                packed = np.packbits(inverted.flatten(), bitorder='big')
                f.write(packed.tobytes())
            
            # Generate SVG filename
            svg_filename = f"{base_filename}_diagram_{index:03d}.svg"
            svg_path = self.output_dir / svg_filename
            
            # Run Potrace to convert PBM to SVG
            # Potrace options:
            # -s: SVG output
            # --tight: Remove redundant points
            # --flat: Suppress curve optimization (faster, simpler)
            result = subprocess.run(
                [
                    'potrace',
                    '-s',  # SVG output
                    '--tight',  # Tighten paths
                    '--flat',  # Flat curves (simpler)
                    '-o', str(svg_path),  # Output file
                    str(pbm_path)  # Input file
                ],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Clean up temporary PBM file
            if pbm_path.exists():
                pbm_path.unlink()
            
            if result.returncode == 0 and svg_path.exists():
                return str(svg_path)
            else:
                print(f"Potrace conversion failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            print("Potrace not found in PATH")
            return None
        except subprocess.TimeoutExpired:
            print(f"Potrace conversion timed out for diagram {index}")
            return None
        except Exception as e:
            print(f"SVG conversion error: {e}")
            return None
    
    def _check_potrace(self) -> bool:
        """
        Check if Potrace is available in the system.
        
        Returns:
            True if Potrace is available, False otherwise
        """
        try:
            import subprocess
            result = subprocess.run(
                ['potrace', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception:
            return False
    
    def cleanup(self):
        """
        Clean up temporary files (optional).
        Can be called after processing is complete.
        """
        # Optionally remove temporary files
        pass


def extract_diagrams(
    image: np.ndarray,
    diagram_regions: List[Dict],
    output_dir: Optional[Path] = None,
    base_filename: str = "diagram"
) -> List[Dict]:
    """
    Convenience function to extract diagrams from image.
    
    Args:
        image: Full document image
        diagram_regions: List of diagram regions with bbox [x1, y1, x2, y2]
        output_dir: Directory to save extracted diagrams
        base_filename: Base filename for saved diagrams
        
    Returns:
        List of dictionaries with diagram information
    """
    extractor = DiagramExtractor(output_dir=output_dir)
    return extractor.extract_diagrams(image, diagram_regions, base_filename)
