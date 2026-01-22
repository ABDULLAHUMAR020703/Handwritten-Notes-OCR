interface ProcessingStatusProps {
  status: string
}

export default function ProcessingStatus({ status }: ProcessingStatusProps) {
  return (
    <div className="processing-container">
      <div className="spinner"></div>
      <p className="processing-text">{status || 'Processing your image...'}</p>
      <p className="processing-hint">
        This may take 1-2 minutes depending on image size and complexity.
      </p>
    </div>
  )
}
