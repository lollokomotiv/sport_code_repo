import { useAuthStore } from '@/store/authStore'

export default function AdminHome() {
  const user = useAuthStore((s) => s.user)
  return (
    <div>
      <h1 className="mb-2 text-xl font-semibold">Dashboard admin — {user?.username}</h1>
      <p className="text-neutral-500">
        Da qui gestirai stagioni, giornate, risultati e tabellone. Pagine in arrivo.
      </p>
    </div>
  )
}
