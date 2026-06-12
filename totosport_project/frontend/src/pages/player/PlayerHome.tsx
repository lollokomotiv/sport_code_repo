import { useAuthStore } from '@/store/authStore'

export default function PlayerHome() {
  const user = useAuthStore((s) => s.user)
  return (
    <div>
      <h1 className="mb-2 text-xl font-semibold">Ciao, {user?.username} 👋</h1>
      <p className="text-neutral-500">
        Benvenuto in TotoSport. Le giornate e le previsioni arriveranno nella prossima fase.
      </p>
    </div>
  )
}
