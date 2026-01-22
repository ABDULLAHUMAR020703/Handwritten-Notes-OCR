@echo off
REM Start script for frontend (Windows)

echo Starting Handwritten Notes OCR Frontend...

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

REM Start the development server
echo Starting development server on http://localhost:3000
npm run dev

pause
