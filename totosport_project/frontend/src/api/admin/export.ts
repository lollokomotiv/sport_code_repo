import api from '@/api/client'

/** Nome file dal Content-Disposition, con fallback se l'header non è esposto. */
function filenameFrom(disposition: string | undefined, fallback: string): string {
  const match = disposition?.match(/filename="?([^";]+)"?/)
  return match?.[1] ?? fallback
}

/**
 * Scarica le classifiche in un unico .xlsx (solo admin) e fa partire il
 * download nel browser. `includeSchedine` aggiunge il foglio con le previsioni
 * grezze di tutta la stagione.
 */
export async function downloadLeaderboardsXlsx(includeSchedine = true): Promise<void> {
  const { data, headers } = await api.get<Blob>('/admin/export/leaderboards', {
    params: { include_schedine: includeSchedine },
    responseType: 'blob',
  })

  const url = URL.createObjectURL(data)
  const link = document.createElement('a')
  link.href = url
  link.download = filenameFrom(headers['content-disposition'], 'totosport_classifiche.xlsx')
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
