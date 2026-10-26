import api from '@/api/client'

/** Nome file dal Content-Disposition, con fallback se l'header non è esposto. */
function filenameFrom(disposition: string | undefined, fallback: string): string {
  const match = disposition?.match(/filename="?([^";]+)"?/)
  return match?.[1] ?? fallback
}

/** Scarica un .xlsx dall'endpoint indicato e fa partire il download nel browser. */
async function downloadXlsx(url: string, fallbackName: string): Promise<void> {
  const { data, headers } = await api.get<Blob>(url, { responseType: 'blob' })

  const objectUrl = URL.createObjectURL(data)
  const link = document.createElement('a')
  link.href = objectUrl
  link.download = filenameFrom(headers['content-disposition'], fallbackName)
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(objectUrl)
}

/** Classifica generale + speciali della stagione corrente (solo admin). */
export function downloadLeaderboardsXlsx(): Promise<void> {
  return downloadXlsx('/admin/export/leaderboards', 'Classifica_Generale.xlsx')
}

/** Classifica di una singola giornata (solo admin). */
export function downloadRoundXlsx(roundId: string): Promise<void> {
  return downloadXlsx(`/admin/export/rounds/${roundId}`, 'Giornata.xlsx')
}
