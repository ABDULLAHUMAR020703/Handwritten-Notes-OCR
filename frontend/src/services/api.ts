const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface StatusCallback {
  (status: string): void
}

export async function convertImagesToWord(
  files: File[],
  onStatusUpdate?: StatusCallback
): Promise<void> {
  const formData = new FormData()
  
  // Append all files with the same field name 'images'
  files.forEach((file) => {
    formData.append('images', file)
  })

  if (onStatusUpdate) {
    onStatusUpdate(`Processing ${files.length} image(s)...`)
  }

  const response = await fetch(`${API_BASE_URL}/api/convert`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let errorMessage = 'Processing failed'
    try {
      const errorData = await response.json()
      errorMessage = errorData.detail || errorMessage
    } catch {
      errorMessage = response.statusText || errorMessage
    }
    throw new Error(errorMessage)
  }

  if (onStatusUpdate) {
    onStatusUpdate('Generating Word document...')
  }

  // Get the blob
  const blob = await response.blob()
  
  // Create download link
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `converted_${files.length}_images.docx`
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

// Keep backward compatibility
export async function convertImageToWord(
  file: File,
  onStatusUpdate?: StatusCallback
): Promise<void> {
  return convertImagesToWord([file], onStatusUpdate)
}
