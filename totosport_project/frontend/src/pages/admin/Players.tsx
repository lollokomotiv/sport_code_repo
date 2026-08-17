import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import {
  deleteUser,
  listUsers,
  registerUser,
  updateUser,
  type UserUpdateInput,
} from '@/api/admin/users'
import LoadingSpinner from '@/components/LoadingSpinner'
import type { UserCreateInput, UserOut } from '@/types/user'
import { errorMessage } from '@/utils'

export default function AdminPlayers() {
  const queryClient = useQueryClient()
  const { data: users, isLoading, isError } = useQuery({ queryKey: ['users'], queryFn: listUsers })

  const [show, setShow] = useState(false)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)

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
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-lg border px-3 py-2 pr-16"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  className="absolute inset-y-0 right-0 px-3 text-xs text-neutral-500 hover:text-neutral-800"
                >
                  {showPassword ? 'Nascondi' : 'Mostra'}
                </button>
              </div>
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
                <th className="px-3 py-2">Stato</th>
                <th className="px-3 py-2 text-right">Azioni</th>
              </tr>
            </thead>
            <tbody>
              {users?.map((u) => (
                <PlayerRow key={u.id} user={u} onChanged={invalidate} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function PlayerRow({ user, onChanged }: { user: UserOut; onChanged: () => void }) {
  const [editing, setEditing] = useState(false)
  const [email, setEmail] = useState(user.email)
  const [password, setPassword] = useState('')

  const updateMut = useMutation({
    mutationFn: (payload: UserUpdateInput) => updateUser(user.id, payload),
    onSuccess: () => {
      onChanged()
      setEditing(false)
      setPassword('')
    },
  })
  const deleteMut = useMutation({
    mutationFn: () => deleteUser(user.id),
    onSuccess: onChanged,
  })

  const isAdmin = user.role === 'admin'
  const rowErr = updateMut.isError
    ? errorMessage(updateMut.error)
    : deleteMut.isError
      ? errorMessage(deleteMut.error)
      : null

  function saveEdit() {
    const payload: UserUpdateInput = {}
    if (email.trim() && email.trim() !== user.email) payload.email = email.trim()
    if (password) payload.password = password
    if (Object.keys(payload).length === 0) {
      setEditing(false)
      return
    }
    updateMut.mutate(payload)
  }

  return (
    <>
      <tr className={`border-t ${user.is_active ? '' : 'text-neutral-400'}`}>
        <td className="px-3 py-2 font-medium">{user.username}</td>
        <td className="px-3 py-2">{user.email}</td>
        <td className="px-3 py-2">{user.role}</td>
        <td className="px-3 py-2">
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
              user.is_active ? 'bg-brand-50 text-brand-700' : 'bg-neutral-200 text-neutral-600'
            }`}
          >
            {user.is_active ? 'attivo' : 'disattivato'}
          </span>
        </td>
        <td className="px-3 py-2">
          {!isAdmin && (
            <div className="flex justify-end gap-3 text-xs">
              <button
                onClick={() => setEditing((e) => !e)}
                className="text-brand-700 hover:underline"
              >
                {editing ? 'Chiudi' : 'Modifica'}
              </button>
              <button
                onClick={() => updateMut.mutate({ is_active: !user.is_active })}
                disabled={updateMut.isPending}
                className="text-neutral-600 hover:underline disabled:opacity-50"
              >
                {user.is_active ? 'Disattiva' : 'Attiva'}
              </button>
              <button
                onClick={() => {
                  if (
                    window.confirm(
                      `Eliminare definitivamente ${user.username}? Verranno rimossi anche i suoi dati (previsioni, punti).`,
                    )
                  )
                    deleteMut.mutate()
                }}
                disabled={deleteMut.isPending}
                className="text-miss hover:underline disabled:opacity-50"
              >
                Elimina
              </button>
            </div>
          )}
          {isAdmin && <span className="text-xs text-neutral-400">—</span>}
        </td>
      </tr>

      {editing && (
        <tr className="border-t bg-neutral-50">
          <td colSpan={5} className="px-3 py-3">
            <div className="grid max-w-xl gap-3 sm:grid-cols-2">
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
                <span className="text-neutral-600">Nuova password (vuoto = invariata)</span>
                <input
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="•••••"
                  className="rounded-lg border px-3 py-2"
                />
              </label>
            </div>
            {rowErr && <p className="mt-2 text-sm text-miss">{rowErr}</p>}
            <div className="mt-3 flex gap-2">
              <button
                onClick={saveEdit}
                disabled={updateMut.isPending}
                className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
              >
                {updateMut.isPending ? 'Salvataggio…' : 'Salva'}
              </button>
              <button
                onClick={() => {
                  setEditing(false)
                  setEmail(user.email)
                  setPassword('')
                }}
                className="rounded-lg border px-4 py-2 text-sm font-medium text-neutral-600 hover:bg-neutral-100"
              >
                Annulla
              </button>
            </div>
          </td>
        </tr>
      )}

      {rowErr && !editing && (
        <tr>
          <td colSpan={5} className="px-3 pb-2 text-xs text-miss">
            {rowErr}
          </td>
        </tr>
      )}
    </>
  )
}
