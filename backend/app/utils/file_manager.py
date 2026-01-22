from pathlib import Path

class FileManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.uploads_dir = self.base_dir / "uploads"
        self.outputs_dir = self.base_dir / "outputs"
        
        self.uploads_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
    
    def get_uploads_dir(self) -> Path:
        return self.uploads_dir
    
    def get_outputs_dir(self) -> Path:
        return self.outputs_dir
