export default function LoadingSpinner() {
  return (
    <div className="flex justify-center py-10" role="status" aria-label="Caricamento">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-neutral-200 border-t-brand-600" />
    </div>
  )
}
