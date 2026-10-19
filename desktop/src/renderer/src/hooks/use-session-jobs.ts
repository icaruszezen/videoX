import { useEffect, useState, type Dispatch, type SetStateAction } from 'react'

import { API_BASE } from '@/lib/api'
import { applyProgress, EVENT_TYPES } from '@/lib/format'
import type { ProgressEvent, SessionJob } from '@/lib/types'

interface SessionJobsState {
  jobs: SessionJob[]
  setJobs: Dispatch<SetStateAction<SessionJob[]>>
  selectedId: string | null
  selectJob: (jobId: string) => void
}

export function useSessionJobs(connected: boolean): SessionJobsState {
  const [jobs, setJobs] = useState<SessionJob[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [streamNonce, setStreamNonce] = useState(0)

  const selectJob = (jobId: string): void => {
    setSelectedId(jobId)
    setJobs((current) =>
      current.map((job) => (job.record.id === jobId ? { ...job, logs: [] } : job))
    )
  }

  useEffect(() => {
    if (!selectedId || !connected) return
    let active = true
    let finished = false
    let logs: string[] = []
    const source = new EventSource(`${API_BASE}/jobs/${selectedId}/events`)

    const handle = (event: Event): void => {
      if (!active) return
      const data = (event as MessageEvent<string>).data
      let payload: ProgressEvent
      try {
        payload = JSON.parse(data) as ProgressEvent
      } catch {
        return
      }
      if (payload.type === 'log' && payload.message) {
        logs = [...logs, payload.message].slice(-400)
      }
      setJobs((current) =>
        current.map((job) =>
          job.record.id === payload.job_id ? applyProgress(job, payload, logs) : job
        )
      )
      if (
        payload.type === 'completed' ||
        payload.type === 'failed' ||
        payload.type === 'cancelled'
      ) {
        finished = true
        source.close()
      }
    }

    for (const type of EVENT_TYPES) source.addEventListener(type, handle)
    source.onerror = () => {
      source.close()
      if (!active || finished) return
      setJobs((current) =>
        current.map((job) => (job.record.id === selectedId ? { ...job, logs: [] } : job))
      )
      window.setTimeout(() => {
        if (active) setStreamNonce((value) => value + 1)
      }, 1000)
    }

    return () => {
      active = false
      source.close()
    }
  }, [connected, selectedId, streamNonce])

  return { jobs, setJobs, selectedId, selectJob }
}
