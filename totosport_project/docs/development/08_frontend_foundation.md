# Fase 8 — Frontend Foundation

Obiettivo: app React funzionante con autenticazione JWT, routing per ruolo, layout base, e client API configurato.

> **Stato: ✅ Completata.** `npm run build` verde (0 errori TS); login verificato
> end-to-end (CORS ok, token ricevuto). Include la chiusura del setup frontend
> rimasto aperto in Fase 1 (Tailwind v3 installato, `strict` on, Prettier, boilerplate rimosso).
> Scelte/deviazioni:
> - **Tailwind v3** (coerente col `tailwind.config.ts` esistente); stack su React 19 /
>   TS 6 / Vite 8 / react-router 7 / zustand 5 (più recenti del doc, ma compatibili).
> - **Entrambi i token in localStorage** via zustand `persist`: il backend restituisce
>   access E refresh nel body (niente cookie HttpOnly). Accettabile per app privata;
>   il refresh ruota i token a ogni uso. Alias `@/` → `src/` (senza `baseUrl`, deprecato in TS6).
> - Alle voci di menu non ancora pronte risponde un placeholder "ComingSoon" (Fase 9).

---

## Checklist

### Setup Vite + React + TypeScript
- [x] `vite.config.ts` con proxy per dev: `/api → http://localhost:8000`
- [x] `tsconfig.json` con `strict: true`, path aliases (`@/` → `src/`)
- [x] Tailwind CSS funzionante con design system base (colori custom nel `tailwind.config.ts`)
- [x] Font: Inter o Geist (ottima leggibilità per classifiche e numeri)

### API Client (`src/api/client.ts`)

```typescript
import axios from 'axios'
import { useAuthStore } from '@/store/authStore'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
})

// Inietta access token ad ogni request
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().accessToken
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Auto-refresh su 401
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true
      try {
        const newToken = await refreshAccessToken()
        useAuthStore.getState().setAccessToken(newToken)
        original.headers.Authorization = `Bearer ${newToken}`
        return api(original)
      } catch {
        useAuthStore.getState().logout()
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export default api
```

### Zustand Store (`src/store/authStore.ts`)

```typescript
interface AuthState {
  accessToken: string | null
  user: { id: string; username: string; role: 'admin' | 'player' } | null
  setAccessToken: (token: string) => void
  setUser: (user: AuthState['user']) => void
  logout: () => void
}
```

- Persisti `accessToken` in `localStorage` (attenzione: vedi nota sicurezza sotto)
- `refreshToken` in cookie HttpOnly (gestito lato backend) — **non accessibile da JS**

> **Nota sicurezza**: il refresh token deve essere un cookie HttpOnly per proteggerlo da XSS. L'access token in localStorage è accettabile per applicazioni non ad alto rischio come questa, ma valuta `sessionStorage` come alternativa.

### Routing (`src/router.tsx`)

```typescript
// React Router v6
const router = createBrowserRouter([
  { path: '/login', element: <Login /> },
  {
    path: '/',
    element: <ProtectedLayout />,  // redirecta a /login se non autenticato
    children: [
      { path: 'player', element: <PlayerLayout />, children: [...playerRoutes] },
      { path: 'admin', element: <AdminLayout requireRole="admin" />, children: [...adminRoutes] },
    ]
  }
])
```

### Componenti Layout
- [x] `ProtectedLayout` — controlla auth, redirecta se non loggato
- [x] `PlayerLayout` — navbar con: Giornate, Le mie previsioni, Classifica, Tabellone
- [x] `AdminLayout` — navbar con: Dashboard, Giornate, Fixture, Tabellone, Classifica, Giocatori
- [x] `LoadingSpinner` — usato da React Query durante il fetching
- [x] `ErrorBoundary` — cattura errori React
- [x] `PageTitle` — wrapper con `<h1>` e breadcrumb opzionale

### Tipi TypeScript (`src/types/`)

Crea interfacce che rispecchiano 1:1 i Pydantic response schemas del backend:

```typescript
// src/types/auth.ts
interface TokenPair { access_token: string; refresh_token: string; token_type: string }
interface UserOut { id: string; username: string; email: string; role: 'admin' | 'player' }

// src/types/round.ts
interface RoundOut { id: string; name: string; competition: string; deadline: string; status: RoundStatus; matches: MatchOut[] }

// src/types/match.ts
interface MatchOut { id: string; home_team: string; away_team: string; kickoff: string; actual_home_goals: number | null; actual_away_goals: number | null }

// src/types/prediction.ts
interface MatchPredictionOut { match_id: string; predicted_home_goals: number; predicted_away_goals: number; points_earned: number }

// src/types/leaderboard.ts
interface LeaderboardEntry { rank: number; player_id: string; username: string; total_points: number }
```

### Utilities (`src/utils/`)
- [x] `deriveSign(home: number, away: number): '1' | 'X' | '2'`
- [x] `formatDate(iso: string): string` — es. "Sab 12 Apr, 20:45"
- [x] `formatCompetition(comp: string): string` — es. "serie_a" → "Serie A"
- [x] `isDeadlinePassed(deadline: string): boolean`

---

## Design system base (Tailwind config)

```typescript
// tailwind.config.ts
colors: {
  brand: { 500: '#16a34a', 600: '#15803d' },  // verde calcio
  goal: '#f59e0b',    // giallo/oro per punti
  miss: '#ef4444',    // rosso per 0 punti
  neutral: colors.slate,
}
```

Usa sempre le utility classes, niente CSS custom salvo casi eccezionali.

---

## Test di accettazione fase 8

1. `npm run dev` senza errori TypeScript
2. `/login` → form login funzionante → redirect a `/player` o `/admin` in base al ruolo
3. Refresh pagina → utente rimane loggato (token in localStorage)
4. Token scaduto → auto-refresh trasparente
5. Accesso a `/admin/*` con account player → redirect a `/player`
6. `npm run build` senza errori TypeScript
