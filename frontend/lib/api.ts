import { getIdToken } from './auth'

const BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? '').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function authHeaders(): Promise<HeadersInit> {
  const token = await getIdToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function friendlyError(status: number, detail?: string): string {
  if (status === 401) return 'Session expired. Please sign in again.'
  if (status === 403) return 'You do not have permission to perform this action.'
  if (status >= 500) return 'Server error — please try again in a moment.'
  return detail ?? `Request failed (${status}).`
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) return res.json() as Promise<T>
  let detail: string | undefined
  try {
    const body = await res.json()
    detail = body?.detail
  } catch { /* ignore */ }
  throw new ApiError(res.status, friendlyError(res.status, detail))
}

// ─── /health ─────────────────────────────────────────────────────────────────

export type HealthResponse = {
  status: string
  model: string
  docs: number | null
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`)
  return handleResponse<HealthResponse>(res)
}

// ─── /chat ───────────────────────────────────────────────────────────────────

export type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

export type Citation = {
  content: string
  product: string | null
  source: string | null
}

export type ChatResponse = {
  answer: string
  citations: Citation[]
  refused: boolean
}

export async function postChat(
  message: string,
  history: ChatMessage[],
): Promise<ChatResponse> {
  const headers = await authHeaders()
  const res = await fetch(`${BASE_URL}/chat`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body:    JSON.stringify({ message, history }),
  })
  return handleResponse<ChatResponse>(res)
}

// ─── /documents ──────────────────────────────────────────────────────────────

export type UploadResponse = {
  added:   number
  skipped: number
  message: string
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const headers = await authHeaders()
  const form    = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE_URL}/documents`, {
    method:  'POST',
    headers: { ...headers },
    body:    form,
  })
  return handleResponse<UploadResponse>(res)
}
