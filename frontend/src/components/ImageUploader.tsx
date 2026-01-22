import { useState, useRef } from 'react'

interface ImageUploaderProps {
  onFilesSelect: (files: File[], previewUrls: string[]) => void
}

export default function ImageUploader({ onFilesSelect }: ImageUploaderProps) {
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files))
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files))
    }
  }

  const handleFiles = (files: File[]) => {
    // Filter image files
    const imageFiles = files.filter(file => file.type.startsWith('image/'))
    
    if (imageFiles.length === 0) {
      alert('Please upload image files (JPG, PNG, etc.)')
      return
    }
    
    // Validate file sizes (max 10MB each)
    const validFiles: File[] = []
    for (const file of imageFiles) {
      if (file.size > 10 * 1024 * 1024) {
        alert(`File "${file.name}" exceeds 10MB limit and will be skipped`)
        continue
      }
      validFiles.push(file)
    }
    
    if (validFiles.length === 0) {
      alert('No valid image files selected')
      return
    }
    
    // Create previews
    const previewPromises = validFiles.map(file => {
      return new Promise<string>((resolve) => {
        const reader = new FileReader()
        reader.onloadend = () => {
          resolve(reader.result as string)
        }
        reader.readAsDataURL(file)
      })
    })
    
    Promise.all(previewPromises).then(previewUrls => {
      onFilesSelect(validFiles, previewUrls)
    })
  }

  return (
    <div
      className={`upload-area ${dragActive ? 'drag-active' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <div className="upload-content">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <h3>Drop your images here</h3>
        <p>or</p>
        <label htmlFor="file-input" className="btn btn-secondary">
          Browse Files
        </label>
        <input
          ref={fileInputRef}
          id="file-input"
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileInput}
          className="file-input"
        />
        <p className="hint">Supported: JPG, PNG, JPEG (Max 10MB per file)</p>
        <p className="hint" style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>
          Multiple images will be merged into one Word document
        </p>
      </div>
    </div>
  )
}
