import { useState } from 'react'
import Head from 'next/head'
import ImageUploader from '../components/ImageUploader'
import ProcessingStatus from '../components/ProcessingStatus'
import { convertImagesToWord } from '../services/api'

export default function Home() {
  const [processing, setProcessing] = useState(false)
  const [status, setStatus] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [previewUrls, setPreviewUrls] = useState<string[]>([])

  const handleFilesSelect = (files: File[], previews: string[]) => {
    setSelectedFiles(files)
    setPreviewUrls(previews)
    setError('')
  }

  const handleConvert = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image')
      return
    }

    setProcessing(true)
    setStatus(`Uploading ${selectedFiles.length} image(s)...`)
    setError('')

    try {
      await convertImagesToWord(selectedFiles, (currentStatus) => {
        setStatus(currentStatus)
      })

      setStatus(`Complete! Document with ${selectedFiles.length} image(s) downloaded.`)
      
      // Reset after 3 seconds
      setTimeout(() => {
        setProcessing(false)
        setStatus('')
        setSelectedFiles([])
        setPreviewUrls([])
      }, 3000)
    } catch (err: any) {
      setError(err.message || 'An error occurred during processing')
      setProcessing(false)
      setStatus('')
    }
  }

  const handleReset = () => {
    setSelectedFiles([])
    setPreviewUrls([])
    setError('')
  }

  return (
    <>
      <Head>
        <title>Handwritten Notes OCR</title>
        <meta name="description" content="Convert handwritten notes images to editable Word documents" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      
      <main className="container">
        <header className="header">
          <h1>📝 Handwritten Notes OCR</h1>
          <p className="subtitle">Convert your handwritten notes to editable Word documents</p>
        </header>

        <div className="content">
          {!processing ? (
            <>
              {selectedFiles.length === 0 ? (
                <ImageUploader onFilesSelect={handleFilesSelect} />
              ) : (
                <div className="preview-section">
                  <div className="files-info">
                    <p className="files-count">{selectedFiles.length} image(s) selected</p>
                  </div>
                  <div className="image-previews">
                    {previewUrls.map((preview, idx) => (
                      <div key={idx} className="image-preview-item">
                        <img src={preview} alt={`Preview ${idx + 1}`} />
                        <p className="image-name">{selectedFiles[idx].name}</p>
                      </div>
                    ))}
                  </div>
                  <div className="preview-actions">
                    <button onClick={handleReset} className="btn btn-secondary">
                      Choose Different Images
                    </button>
                    <button onClick={handleConvert} className="btn btn-primary btn-large">
                      Convert {selectedFiles.length > 1 ? 'All Images' : 'Image'} to Word
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div className="error-message">
                  <p>❌ {error}</p>
                  <button onClick={() => setError('')} className="btn-close">
                    Close
                  </button>
                </div>
              )}
            </>
          ) : (
            <ProcessingStatus status={status} />
          )}
        </div>

        <div className="info-section">
          <h2>How it works</h2>
          <ol>
            <li>Upload one or more images of your handwritten notes</li>
            <li>Click "Convert to Word"</li>
            <li>Wait for processing (takes 1-2 minutes per image)</li>
            <li>Download your merged editable Word document</li>
          </ol>
        </div>
      </main>
    </>
  )
}
