'use client'

import { useState, useCallback } from 'react'
import { useDropzone }            from 'react-dropzone'
import { Upload, FileText, CheckCircle, AlertCircle, XCircle, Info } from 'lucide-react'
import { uploadDocument, type UploadResponse, ApiError } from '@/lib/api'
import { Button } from '@/components/ui/Button'
import clsx from 'clsx'

const ACCEPTED_TYPES = {
  'application/json':    ['.json'],
  'text/plain':          ['.txt'],
  'text/csv':            ['.csv'],
  'application/pdf':     ['.pdf'],
}
const MAX_SIZE_MB = 5

type FileStatus = {
  file:     File
  state:    'pending' | 'uploading' | 'success' | 'error'
  result?:  UploadResponse
  error?:   string
}

export function UploadPanel() {
  const [queue, setQueue] = useState<FileStatus[]>([])

  const updateStatus = useCallback((name: string, patch: Partial<FileStatus>) => {
    setQueue(prev => prev.map(f => f.file.name === name ? { ...f, ...patch } : f))
  }, [])

  const onDrop = useCallback((accepted: File[]) => {
    const newEntries: FileStatus[] = accepted
      .filter(f => !queue.some(q => q.file.name === f.name))
      .map(f => ({ file: f, state: 'pending' }))
    setQueue(prev => [...prev, ...newEntries])
  }, [queue])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept:  ACCEPTED_TYPES,
    maxSize: MAX_SIZE_MB * 1024 * 1024,
    onDropRejected: rejections => {
      rejections.forEach(({ file, errors }) => {
        const msg = errors.map(e => e.message).join(', ')
        setQueue(prev => {
          const exists = prev.find(q => q.file.name === file.name)
          if (exists) return prev.map(q => q.file.name === file.name ? { ...q, state: 'error', error: msg } : q)
          return [...prev, { file, state: 'error', error: msg }]
        })
      })
    },
  })

  const uploadAll = async () => {
    const pending = queue.filter(f => f.state === 'pending')
    for (const entry of pending) {
      updateStatus(entry.file.name, { state: 'uploading' })
      try {
        const result = await uploadDocument(entry.file)
        updateStatus(entry.file.name, { state: 'success', result })
      } catch (err) {
        const msg = err instanceof ApiError ? err.message : 'Upload failed.'
        updateStatus(entry.file.name, { state: 'error', error: msg })
      }
    }
  }

  const remove = (name: string) => setQueue(prev => prev.filter(f => f.file.name !== name))
  const hasPending = queue.some(f => f.state === 'pending')

  return (
    <div className="max-w-2xl mx-auto py-8 px-4 flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-[var(--color-text)]">Knowledge Base</h2>
        <p className="text-sm text-[var(--color-text-muted)] mt-1">
          Upload documents to expand the bank&apos;s knowledge base. Supported: .json, .txt, .csv, .pdf (max {MAX_SIZE_MB} MB each).
        </p>
      </div>

      {/* Info banner */}
      <div className="flex items-start gap-3 bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-xl px-4 py-3 text-sm text-primary-800 dark:text-primary-200">
        <Info size={15} className="shrink-0 mt-0.5" />
        <span>Uploads are <strong>persistent</strong> and immediately searchable — no restart needed.</span>
      </div>

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={clsx(
          'border-2 border-dashed rounded-2xl px-6 py-10 text-center cursor-pointer transition-all duration-150',
          isDragActive
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
            : 'border-[var(--color-border)] hover:border-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-900/10',
        )}
        role="button"
        aria-label="File drop zone"
        tabIndex={0}
      >
        <input {...getInputProps()} aria-label="File input" />
        <Upload className="mx-auto mb-3 text-[var(--color-text-muted)]" size={32} />
        <p className="text-sm font-medium text-[var(--color-text)]">
          {isDragActive ? 'Drop files here…' : 'Drag & drop files, or click to browse'}
        </p>
        <p className="text-xs text-[var(--color-text-muted)] mt-1">.json · .txt · .csv · .pdf · max 5 MB</p>
      </div>

      {/* File list */}
      {queue.length > 0 && (
        <ul className="flex flex-col gap-2">
          {queue.map(entry => (
            <li
              key={entry.file.name}
              className="flex items-start gap-3 bg-[var(--color-panel)] border border-[var(--color-border)] rounded-xl px-4 py-3 text-sm shadow-card"
            >
              <FileText size={16} className="shrink-0 mt-0.5 text-[var(--color-text-muted)]" />
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate text-[var(--color-text)]">{entry.file.name}</p>
                <p className="text-xs text-[var(--color-text-muted)]">
                  {(entry.file.size / 1024).toFixed(1)} KB
                </p>
                {entry.state === 'success' && entry.result && (
                  <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                    Added {entry.result.added}, skipped {entry.result.skipped} — {entry.result.message}
                  </p>
                )}
                {entry.state === 'error' && (
                  <p className="text-xs text-red-600 dark:text-red-400 mt-1">{entry.error}</p>
                )}
              </div>
              <div className="shrink-0 flex items-center gap-2">
                {entry.state === 'uploading' && (
                  <svg className="animate-spin h-4 w-4 text-primary-600" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                  </svg>
                )}
                {entry.state === 'success'  && <CheckCircle size={16} className="text-green-500" />}
                {entry.state === 'error'    && <AlertCircle size={16} className="text-red-500" />}
                {entry.state !== 'uploading' && (
                  <button onClick={() => remove(entry.file.name)} aria-label={`Remove ${entry.file.name}`}>
                    <XCircle size={16} className="text-[var(--color-text-muted)] hover:text-red-500 transition-colors" />
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      {hasPending && (
        <Button onClick={uploadAll} className="self-end">
          <Upload size={15} />
          Upload {queue.filter(f => f.state === 'pending').length} file{queue.filter(f => f.state === 'pending').length !== 1 ? 's' : ''}
        </Button>
      )}
    </div>
  )
}
