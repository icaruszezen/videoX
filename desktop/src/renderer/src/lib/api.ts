import type { EngineManifest, JobRecord, MediaKind, SystemReport } from './types'

export const API_BASE = 'http://127.0.0.1:8765'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export interface CreateJobBody {
  engine_id: string
  input_path: string
  output_path: string
  media: MediaKind
  devices?: number[]
  params: Record<string, unknown>
}

function detailMessage(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) return String(item.msg)
        return JSON.stringify(item)
      })
      .join('；')
  }
  return '请求失败'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {})
      }
    })
  } catch {
    throw new ApiError('无法连接本地服务', 0)
  }
  if (!response.ok) {
    let message = response.statusText || '请求失败'
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (body.detail != null) message = detailMessage(body.detail)
    } catch {
      // The body was not JSON.
    }
    throw new ApiError(message, response.status)
  }
  return (await response.json()) as T
}

export function getSystem(): Promise<SystemReport> {
  return request('/system')
}

export function getEngines(): Promise<EngineManifest[]> {
  return request('/engines')
}

export function createJob(body: CreateJobBody): Promise<JobRecord> {
  return request('/jobs', { method: 'POST', body: JSON.stringify(body) })
}

export function getJob(jobId: string): Promise<JobRecord> {
  return request(`/jobs/${jobId}`)
}

export function cancelJob(jobId: string): Promise<JobRecord> {
  return request(`/jobs/${jobId}/cancel`, { method: 'POST' })
}
