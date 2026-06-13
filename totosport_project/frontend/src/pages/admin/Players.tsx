import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { disableUser, listUsers, registerUser } from '@/api/admin/users'
import LoadingSpinner from '@/components/LoadingSpinner'
import type { UserCreateInput } from '@/types/user'
import { errorMessage, formatDate } from '@/utils'

export default function AdminPlayers() {
  const queryClient = useQueryClient()
  const { data: users, isLoading, isError } = useQuery({ queryKey: ['users'], queryFn: listUsers })

  const [show, setShow] = useState(false)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['users'] })

  const createMut = useMutation({
    mutationFn: (payload: UserCreateInput) => registerUser(payload),
    onSuccess: () => {
      invalidate()
      setShow(false)
      setUsername('')
      setEmail('')
      setPassword('')
    },
  })
  const disableMut = useMutation({
    mutationFn: (userId: string) => disableUser(userId),
    onSuccess: invalidate,
  })

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">Giocatori</h1>
        <button
          onClick={() => setShow((s) => !s)}
          className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          {show ? 'Annulla' : 'Crea giocatore'}
        </button>
      </div>

      {show && (
        <div className="mb-4 rounded-xl border bg-white p-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-neutral-600">Username</span>
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="rounded-lg border px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-neutral-600">Email</span>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="rounded-lg border px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-neutral-600">Password iniziale</span>
              <input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="rounded-lg border px-3 py-2"
              />
            </label>
          </div>
          <p className="mt-2 text-xs text-neutral-500">
            L'account viene creato subito ma non viene inviata alcuna mail: comunica tu username e
            password iniziale al giocatore.
          </p>
          {createMut.isError && (
            <p className="mt-2 text-sm text-miss">{errorMessage(createMut.error)}</p>
          )}
          <div className="mt-3 flex justify-end">
            <button
              onClick={() =>
                createMut.mutate({
                  username: username.trim(),
                  email: email.trim(),
                  password,
                  role: 'player',
                })
              }
              disabled={
                createMut.isPending || !username.trim() || !email.trim() || password.length < 4
              }
              className="rounded-lg bg-brand-600 px-6 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
            >
              {createMut.isPending ? 'Creazione…' : 'Crea giocatore'}
            </button>
          </div>
        </div>
      )}

      {isLoading ? (
        <LoadingSpinner />
      ) : isError ? (
        <p className="text-miss">Errore nel caricamento.</p>
      ) : (
        <div className="overflow-x-auto rounded-xl border bg-white">
          <table className="w-full text-sm">
            <thead className="bg-neutral-50 text-left text-neutral-500">
              <tr>
                <th className="px-3 py-2">Username</th>
                <th className="px-3 py-2">Email</th>
                <th className="px-3 py-2">Ruolo</th>
                <th className="px-3 py-2">Registrato</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {users?.map((u) => (
                <tr key={u.id} className={`border-t ${u.is_active ? '' : 'text-neutral-400'}`}>
                  <td className="px-3 py-2 font-medium">{u.username}</td>
                  <td className="px-3 py-2">{u.email}</td>
                  <td className="px-3 py-2">{u.role}</td>
                  <td className="px-3 py-2 text-neutral-500">{formatDate(u.created_at)}</td>
                  <td className="px-3 py-2 text-right">
                    {u.role !== 'admin' &&
                      (u.is_active ? (
                        <button
                          onClick={() => {
                            if (window.confirm(`Disattivare ${u.username}?`))
                              disableMut.mutate(u.id)
                          }}
                          className="text-xs text-miss hover:underline"
                        >
                          Disattiva
                        </button>
                      ) : (
                        <span className="text-xs text-neutral-400">disattivato</span>
                      ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
