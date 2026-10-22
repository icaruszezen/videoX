import { useEffect, useState, type Dispatch, type SetStateAction } from 'react'

import { API_BASE, cancelJob, getJob } from '@/lib/api'
import { applyProgress, EVENT_TYPES } from '@/lib/format'
import type { JobRecord, ProgressEvent, SessionJob } from '@/lib/types'

interface SessionJobsState {
  jobs: SessionJob[]
  setJobs: Dispatch<SetStateAction<SessionJob[]>>
  selectedId: string | null
  selectJob: (jobId: string) => void
}

function sessionJob(record: JobRecord): SessionJob {
  return { record, logs: [], message: record.error ?? '' }
}

export function useSessionJobs(connected: boolean): SessionJobsState {
  const [jobs, setJobs] = useState<SessionJob[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [streamNonce, setStreamNonce] = useState(0)
  const jobIds = jobs.map((job) => job.record.id).join(',')

  const selectJob = (jobId: string): void => {
    setSelectedId(jobId)
    setJobs((current) =>
      current.map((job) => (job.record.id === jobId ? { ...job, logs: [] } : job))
    )
  }

  useEffect(() => {
    if (!connected) return
    let active = true
    void (async () => {
      let ids: string[] = []
      try {
        ids = (await window.api?.readJobIds()) ?? []
      } catch {
        return
      }
      if (!active || ids.length === 0) return
      const records: JobRecord[] = []
      for (const id of ids) {
        try {
          records.push(await getJob(id))
        } catch {
          // The API process may have restarted and dropped in-memory jobs.
        }
      }
      if (!active || records.length === 0) return
      const preferred =
        records.find((record) => record.status === 'running' || record.status === 'queued') ??
        records[records.length - 1]
      setJobs((current) => (current.length > 0 ? current : records.map(sessionJob)))
      setSelectedId((current) => current ?? preferred.id)
    })()
    return () => {
      active = false
    }
  }, [connected])

  useEffect(() => {
    if (!jobIds || !window.api?.writeJobIds) return
    void window.api.writeJobIds(jobIds.split(','))
  }, [jobIds])

  useEffect(() => {
    if (!selectedId || !connected) return
    let active = true
    const pull = async (): Promise<void> => {
      try {
        const record = await getJob(selectedId)
        if (!active) return
        setJobs((current) =>
          current.map((job) =>
            job.record.id === record.id
              ? {
                  ...job,
                  record: {
                    ...job.record,
                    status: record.status,
                    stage: record.stage,
                    progress: record.progress,
                    error: record.error
                  },
                  message: job.record.stage === record.stage ? job.message : ''
                }
              : job
          )
        )
      } catch {
        // The event stream still carries the detailed log.
      }
    }
    void pull()
    const timer = window.setInterval(() => void pull(), 2000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [connected, selectedId])

  useEffect(() => {
    if (!selectedId || !connected) return
    let active = true
    let finished = false
    const logs: string[] = []
    let queue: ProgressEvent[] = []
    let timer = 0

    const flush = (): void => {
      timer = 0
      if (!active || queue.length === 0) return
      const batch = queue
      queue = []
      for (const payload of batch) {
        if (payload.type === 'log' && payload.message) logs.push(payload.message)
      }
      if (logs.length > 400) logs.splice(0, logs.length - 400)
      const snapshot = logs.slice()
      setJobs((current) =>
        current.map((job) => {
          let next = job
          for (const payload of batch) {
            if (next.record.id === payload.job_id) next = applyProgress(next, payload, snapshot)
          }
          return next
        })
      )
    }

    const schedule = (): void => {
      if (timer !== 0) return
      timer = window.setTimeout(flush, 150)
    }

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
      queue.push(payload)
      if (
        payload.type === 'completed' ||
        payload.type === 'failed' ||
        payload.type === 'cancelled'
      ) {
        finished = true
        if (timer !== 0) window.clearTimeout(timer)
        flush()
        source.close()
        return
      }
      schedule()
    }

    for (const type of EVENT_TYPES) source.addEventListener(type, handle)
    source.onerror = () => {
      source.close()
      if (timer !== 0) window.clearTimeout(timer)
      queue = []
      if (!active || finished) return
      logs.splice(0, logs.length)
      setJobs((current) =>
        current.map((job) => (job.record.id === selectedId ? { ...job, logs: [] } : job))
      )
      window.setTimeout(() => {
        if (active) setStreamNonce((value) => value + 1)
      }, 1000)
    }

    return () => {
      active = false
      if (timer !== 0) window.clearTimeout(timer)
      source.close()
    }
  }, [connected, selectedId, streamNonce])

  return { jobs, setJobs, selectedId, selectJob }
}

export async function cancelSessionJob(
  jobId: string,
  onJobsChange: (updater: (current: SessionJob[]) => SessionJob[]) => void
): Promise<void> {
  const record = await cancelJob(jobId)
  onJobsChange((current) =>
    current.map((job) =>
      job.record.id === record.id ? { ...job, record, message: '已取消' } : job
    )
  )
}
