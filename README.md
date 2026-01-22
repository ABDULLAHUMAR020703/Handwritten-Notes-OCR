# Handwritten Notes OCR to Word

A free, open-source AI-powered web application that converts handwritten notes images into editable Word (.docx) documents with preserved formatting.

## Features

- ✅ **HYBRID OCR + C-RNN Pipeline** - Advanced handwriting recognition using DB detection + CRNN recognition at line level
- ✅ **Handwriting Recognition** - Optimized for messy student notes using PaddleOCR (DB + CRNN) and EasyOCR fallback
- ✅ **Line-Level Processing** - Processes full text lines (not words) for better context-aware recognition
- ✅ **Zero Text Drop Policy** - ALL OCR text is preserved regardless of confidence (no silent discards)
- ✅ **Format Preservation** - Maintains headings, paragraphs, lists, and structure
- ✅ **Equation Detection** - Automatically detects and converts equations to readable text or LaTeX
- ✅ **Text-Only Output** - Generates 100% text-based Word documents (no images, shapes, or drawings)
- ✅ **Raw OCR Output** - Preserves original OCR text without aggressive cleaning or filtering
- ✅ **Mathematical Symbols** - Preserves symbols: +, -, =, /, x, ∑, ∫
- ✅ **Diagram Detection** - Extremely conservative detection; only uses placeholders if OCR returns empty
- ✅ **Light Preprocessing** - Grayscale conversion only (no binarization that breaks strokes)
- ✅ **Debug Logging** - Comprehensive logging tracks all OCR results and any text discards
- ✅ **100% Free** - Uses only free and open-source tools (no paid APIs)

## Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI** - Modern, fast web framework
- **PaddleOCR** - HYBRID pipeline:
  - **DB Detector** - Text detection and line segmentation
  - **CRNN Recognizer** - CNN + RNN + CTC for handwritten text recognition
- **EasyOCR** - Fallback OCR engine
- **pix2text** - Math equation recognition
- **OpenCV** - Light image preprocessing (grayscale only)
- **python-docx** - Word document generation (text-only, validated)

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Modern CSS** - Clean, responsive UI

## Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- npm or yarn

### Backend Setup

#### Step 1: Navigate to backend directory
```bash
cd backend
```

#### Step 2: Create virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verify activation:** You should see `(venv)` in your terminal prompt.

#### Step 3: Install dependencies

**Important:** Make sure your virtual environment is activated (you should see `(venv)` in your prompt).

```bash
pip install -r requirements.txt
```

**Note:** 
- First installation will download ML models (~500MB for EasyOCR, PaddleOCR, and pix2text)
- This may take 5-10 minutes depending on your internet connection
- Models are cached after first download for faster subsequent runs

**If you encounter dependency conflicts:**

**Windows - Delete and recreate virtual environment:**
```bash
# Delete venv
rmdir /s /q venv

# Recreate venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac - Delete and recreate virtual environment:**
```bash
# Delete venv
rm -rf venv

# Recreate venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**PowerShell (Windows) - Alternative deletion:**
```powershell
Remove-Item -Recurse -Force venv
```

#### Step 4: Create necessary directories
```bash
# Windows
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

# Linux/Mac
mkdir -p uploads outputs
```

#### Step 5: Run the server

**Option 1: Using start script (Recommended - Windows)**
```bash
start.bat
```

**Option 2: Using start script (Recommended - Linux/Mac)**
```bash
chmod +x start.sh
./start.sh
```

**Option 3: Using Python module directly**
```bash
# Make sure venv is activated
python -m app.main
```

**Option 4: Using uvicorn directly (with auto-reload for development)**
```bash
# Make sure venv is activated
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Verify server is running:**
- Open your browser and go to `http://localhost:8000`
- You should see: `{"status":"ok","message":"Handwritten Notes OCR API"}`
- API documentation: `http://localhost:8000/docs`

**Viewing logs:** The server will output real-time logs showing:
- Model initialization progress
- Image processing steps
- OCR recognition progress
- Document generation status

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. **Start the backend** (in one terminal):
```bash
cd backend
python -m app.main
```

2. **Start the frontend** (in another terminal):
```bash
cd frontend
npm run dev
```

3. **Open the application**:
   - Open your browser and go to `http://localhost:3000`
   - Upload an image of handwritten notes (JPG, PNG, JPEG)
   - Click "Convert to Word"
   - **Watch the backend terminal** for real-time processing logs:
     - 📸 Image upload and file size
     - 🔧 Image preprocessing (Step 1/5)
     - 📐 Layout detection - number of regions found (Step 2/5)
     - 🔍 OCR extraction progress (Step 3/5)
     - 📄 Document structure building (Step 4/5)
     - 📝 Word document generation (Step 5/5)
   - Wait for processing (may take 1-2 minutes depending on image size)
   - Word document will automatically download
   - **Note:** The document contains only text (no images). All text is selectable and copy-pasteable.

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── api/
│   │   │   ├── convert.py          # Main conversion endpoint
│   │   │   ├── upload.py           # Upload endpoint
│   │   │   └── download.py         # Download endpoint
│   │   ├── services/
│   │   │   ├── preprocessing.py    # Image preprocessing
│   │   │   ├── layout.py           # Layout detection
│   │   │   ├── text_ocr.py         # Text OCR
│   │   │   ├── math_ocr.py         # Equation recognition
│   │   │   ├── diagram.py          # Diagram extraction
│   │   │   ├── document_builder.py # JSON document builder
│   │   │   └── docx_generator.py   # Word document generator
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic models
│   │   └── utils/
│   │       └── file_manager.py     # File management utilities
│   ├── requirements.txt            # Python dependencies
│   ├── start.bat                   # Windows start script
│   ├── start.sh                    # Linux/Mac start script
│   ├── uploads/                    # Temporary upload directory (auto-created)
│   └── outputs/                    # Generated documents directory (auto-created)
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── index.tsx           # Main page
│   │   │   └── _app.tsx            # App wrapper
│   │   ├── components/
│   │   │   ├── ImageUploader.tsx   # File upload component
│   │   │   └── ProcessingStatus.tsx # Status display
│   │   ├── services/
│   │   │   └── api.ts              # API service
│   │   └── styles/
│   │       └── globals.css         # Global styles
│   ├── package.json
│   ├── tsconfig.json
│   └── start.bat                   # Windows start script
├── README.md
└── TESTING.md                      # Testing guide and limitations
```

## API Endpoints

### POST /api/convert
Convert handwritten notes image to Word document.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file (form field: `image`)

**Response:**
- Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document
- Body: Word document file (.docx)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/convert" \
  -F "image=@your_notes.jpg"
```

### GET /
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "Handwritten Notes OCR API"
}
```

## How It Works

### HYBRID OCR + C-RNN Pipeline Architecture

The system uses a **HYBRID approach** that separates detection from recognition:

1. **Light Preprocessing** - Converts to grayscale only (NO binarization, NO thresholding)
   - Preserves original character shapes
   - Avoids morphological operations that break strokes

2. **Text Detection (DB Detector)** - PaddleOCR DB algorithm finds text regions
   - Detects text lines (not words)
   - Conservative detection to avoid classifying handwriting as diagrams
   - If letters/numbers exist → treated as TEXT

3. **Line Segmentation** - Extracts full text lines
   - NOT word-level processing
   - NOT character-level processing
   - Full lines preserve context for better recognition

4. **C-RNN Recognition** - PaddleOCR CRNN with CTC decoding
   - Processes each line image
   - CNN extracts features
   - RNN handles sequence context
   - CTC decodes the sequence
   - Returns RAW recognized text

5. **Text Preservation** - CRITICAL RULES:
   - **ALL OCR text is preserved** regardless of confidence
   - **NO confidence-based filtering** - confidence is for logging only
   - **Raw OCR output** - no aggressive cleaning or character removal
   - **Text presence > Text perfection** - bad text is better than no text
   - Only empty OCR results are skipped (no placeholders for text regions)

6. **Layout Analysis** - Detects text regions vs diagrams vs equations

7. **Equation Recognition** - Detects and converts equations:
   - Uses pix2text for LaTeX conversion
   - Falls back to readable text if needed
   - Preserves mathematical symbols: +, -, =, /, x, ∑, ∫
   - Tags complex equations: `[complex equation]`

8. **Diagram Detection** - EXTREMELY conservative:
   - Only classifies as diagram if OCR returns EMPTY string
   - If even one character detected → treated as TEXT
   - Placeholders are LAST RESORT only: `[diagram]`

9. **Document Generation** - Creates formatted Word document:
   - Preserved headings and structure
   - Selectable, copy-pasteable text only
   - Line breaks preserved
   - **Validation ensures NO images, shapes, or drawings**
   - Equations as text (LaTeX notation or readable form)

### Debug Logging

The system includes comprehensive OCR debug logging:
- Logs raw OCR text for each region
- Logs confidence scores (for monitoring, not filtering)
- Logs when text is discarded (marked as BUG)
- Ensures OCR text is never silently dropped

## Performance

### Accuracy Estimates (on good quality images):
- **Text Recognition:** 80-90% accuracy (HYBRID DB + CRNN pipeline, line-level processing)
- **Heading Detection:** 85-95% accuracy
- **Equation Recognition:** 60-75% accuracy (converts to readable text if low confidence)
- **Diagram Detection:** 90-95% accuracy (extremely conservative - only if OCR returns empty)
- **Text Preservation:** 100% (all OCR text included, no confidence-based filtering)

### Processing Times:
- **Small images (< 2MB):** 1-2 minutes
- **Medium images (2-5 MB):** 2-3 minutes
- **Large images (> 5 MB):** 3-5 minutes

## Limitations

See [TESTING.md](TESTING.md) for detailed limitations and recommendations.

### Key Limitations:
- Messy handwriting reduces accuracy significantly (but ALL text is still included)
- Complex equations may convert to readable text form instead of LaTeX
- Multi-column layouts may have reading order issues
- Requires good image quality (800x600 minimum, good lighting)
- First run downloads ML models (~500MB for PaddleOCR, EasyOCR, pix2text)
- Diagrams are not extracted - only text placeholders added if OCR returns empty
- Documents are 100% text-based (no images, shapes, or drawings)
- Line-level processing may merge words that should be separate (handwriting limitation)

## Troubleshooting

### Backend won't start
- **Error:** `can't open file 'main.py'`
  - **Solution:** Make sure you're in the `backend` directory and run `python -m app.main`
  
- Ensure Python 3.10+ is installed: `python --version`
- Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
- Check dependencies: `pip install -r requirements.txt`
- Run from backend directory: `cd backend && python -m app.main`

### Models not loading
- First run downloads models automatically (~500MB)
- Ensure stable internet connection
- Check disk space (models need ~1GB)

### OCR accuracy is low
- Use better quality images (good lighting, sharp focus)
- Ensure text is clearly written
- Avoid severe skew (>15°)
- See [TESTING.md](TESTING.md) for optimization tips

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check CORS settings in `backend/app/main.py`
- Verify API URL in `frontend/src/services/api.ts`
- Check browser console for CORS errors

### Port already in use
- Backend default port: 8000
- Frontend default port: 3000
- Change ports if needed:
  - Backend: `uvicorn app.main:app --port 8001`
  - Frontend: `npm run dev -- -p 3001`

## Development

### Running in Development Mode

**Backend (with auto-reload):**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (with hot reload):**
```bash
cd frontend
npm run dev
```

### Environment Variables

Create `.env` files if needed:

**Backend (.env):**

**Gemini Vision API Configuration (Optional):**
```env
# Gemini Vision API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
# Replace 'your-gemini-api-key-here' with your actual API key

GEMINI_API_KEY=your-gemini-api-key-here

# Optional: Model selection (default: models/gemini-2.5-pro)
# The system automatically selects from available models:
# 1. Tries models/gemini-2.5-pro (preferred)
# 2. Falls back to models/gemini-2.5-flash if 2.5-pro not available
# 3. Disables Gemini if neither model is available
# Note: Model selection is dynamic based on actual API availability
GEMINI_MODEL=models/gemini-2.5-pro

# Optional: Enable fast mode (uses models/gemini-2.5-flash)
# Set to 'true' to prefer fast model, 'false' for primary model
# Note: Final selection still depends on actual model availability
GEMINI_FAST_MODE=false
```

**Important Notes:**
- Gemini is **optional** - the system works without it
- If Gemini API key is not set, the system uses OCR-only mode
- Model validation happens at startup - invalid models will cause startup failure
- Only officially supported, versioned models are allowed (no `-latest` suffixes)
- Model selection is **dynamic** - checks actual availability via `list_models()`
- Preferred models: `models/gemini-2.5-pro` (primary), `models/gemini-2.5-flash` (fallback)
- Forbidden models: `gemini-*-latest`, unversioned `gemini-pro`/`gemini-flash`
- If no preferred models are available, Gemini is automatically disabled (no crash)
```
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
LOG_LEVEL=INFO
```

**Frontend (.env.local):**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open-source and free to use.

## Notes

- First run will download ML models (~500MB for EasyOCR, PaddleOCR, pix2text)
- GPU acceleration is recommended but not required (CPU works, slower)
- For best results, use clear, well-lit photos with minimal skew
- Processing time depends on image size and complexity
- Models are cached after first download for faster subsequent runs
- **Output documents are 100% text-based** - All text is selectable and copy-pasteable
- **No images are embedded** - Diagrams are detected but replaced with text placeholders (only if OCR returns empty)
- **ALL OCR text is preserved** - No confidence-based filtering; all recognized text is included
- **HYBRID pipeline** - DB detection + CRNN recognition at line level for optimal handwriting recognition
- **Raw OCR output** - Text is preserved exactly as extracted (no aggressive cleaning)
- **Equations** are converted to readable text if OCR confidence is low
- **Debug logging** - Comprehensive logging tracks all OCR results and any text discards