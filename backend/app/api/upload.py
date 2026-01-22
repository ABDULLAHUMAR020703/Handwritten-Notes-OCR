from fastapi import APIRouter, File, UploadFile, HTTPException
from pathlib import Path
import tempfile
import os

from app.services.image_processor import ImageProcessor
from app.services.layout_analyzer import LayoutAnalyzer
from app.services.ocr_engine import OCREngine
from app.services.document_generator import DocumentGenerator
from app.utils.file_manager import FileManager

router = APIRouter()

image_processor = ImageProcessor()
layout_analyzer = LayoutAnalyzer()
ocr_engine = OCREngine()
doc_generator = DocumentGenerator()
file_manager = FileManager()

@router.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        uploads_dir = file_manager.get_uploads_dir()
        temp_input = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(image.filename).suffix, dir=uploads_dir
        )
        content = await image.read()
        temp_input.write(content)
        temp_input.close()
        
        input_path = temp_input.name
        
        processed_image = image_processor.preprocess(input_path)
        layout_result = layout_analyzer.analyze(processed_image)
        ocr_result = ocr_engine.process(layout_result)
        
        outputs_dir = file_manager.get_outputs_dir()
        output_path = doc_generator.generate(
            ocr_result, 
            input_path,
            output_dir=outputs_dir
        )
        
        os.unlink(input_path)
        
        return {"status": "success", "output_path": output_path}
        
    except Exception as e:
        if 'input_path' in locals() and os.path.exists(input_path):
            os.unlink(input_path)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
