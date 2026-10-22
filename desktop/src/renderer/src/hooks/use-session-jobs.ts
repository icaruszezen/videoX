import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react'

import { API_BASE, ApiError, cancelJob, getJob } from '@/lib/api'
import { applyProgress, EVENT_TYPES, isProtocolLine, isTerminal } from '@/lib/format'
import type { JobRecord, MediaKind, ProgressEvent, SessionJob } from '@/lib/types'

const LOG_LIMIT = 400
const LOST_MESSAGE = '本地服务里已没有这个任务，服务可能重启过'

type SetJobs = Dispatch<SetStateAction<SessionJob[]>>

interface SessionJobsState {
  jobs: SessionJob[]
  setJobs: SetJobs
  selectedIds: Record<MediaKind, string | null>
  selectJob: (jobId: string) => void
}

function sessionJob(record: JobRecord): SessionJob {
  return { record, logs: [], message: '' }
}

function jobMedia(job: SessionJob): MediaKind {
  return job.record.spec.media ?? 'video'
}

// The 任务 page and the 图片 page each show the job picked last for their media.
// Before anything is picked, a running job wins, then the newest one.
function selectedFor(jobs: SessionJob[], picks: string[], media: MediaKind): string | null {
  const pool = jobs.filter((job) => jobMedia(job) === media)
  for (let index = picks.length - 1; index >= 0; index--) {
    if (pool.some((job) => job.record.id === picks[index])) return picks[index]
  }
  const active = pool.find((job) => !isTerminal(job.record.status))
  return (active ?? pool[0])?.record.id ?? null
}

function mergeRecord(job: SessionJob, record: JobRecord): SessionJob {
  // A poll that started before the terminal event arrived must not undo it.
  if (isTerminal(job.record.status)) return job
  return {
    ...job,
    record: {
      ...job.record,
      status: record.status,
      stage: record.stage,
      progress: record.progress,
      error: record.error
    }
  }
}

function markLost(job: SessionJob): SessionJob {
  if (isTerminal(job.record.status)) return job
  return { ...job, record: { ...job.record, status: 'failed', error: LOST_MESSAGE } }
}

function isMissing(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}

export function useSessionJobs(connected: boolean): SessionJobsState {
  const [jobs, setJobs] = useState<SessionJob[]>([])
  const [picks, setPicks] = useState<string[]>([])
  const jobIds = jobs.map((job) => job.record.id).join(',')
  const activeIds = jobs
    .filter((job) => !isTerminal(job.record.status))
    .map((job) => job.record.id)
    .join(',')
  const selectedIds = {
    video: selectedFor(jobs, picks, 'video'),
    image: selectedFor(jobs, picks, 'image')
  }

  const selectJob = (jobId: string): void => {
    setPicks((current) =>
      current[current.length - 1] === jobId
        ? current
        : [...current.filter((id) => id !== jobId), jobId]
    )
  }

  useEffect(() => {
    if (!connected) return
    let active = true
    void (async () => {
      let ids: string[]
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
      const restored = records.map(sessionJob)
      setJobs((current) => (current.length > 0 ? current : restored))
      // Pin what the pages show first, so a restored job that finishes stays on
      // screen instead of the view moving on to another job.
      setPicks((current) =>
        current.length > 0
          ? current
          : (['video', 'image'] as const)
              .map((media) => selectedFor(restored, [], media))
              .filter((id): id is string => id !== null)
      )
    })()
    return () => {
      active = false
    }
  }, [connected])

  useEffect(() => {
    if (!jobIds || !window.api?.writeJobIds) return
    void window.api.writeJobIds(jobIds.split(','))
  }, [jobIds])

  // Keep every unfinished job fresh, not only the selected one, so the job lists
  // show the right status and progress. Finished jobs are not polled.
  useEffect(() => {
    if (!activeIds || !connected) return
    const ids = activeIds.split(',')
    let active = true
    const pull = async (): Promise<void> => {
      const results = await Promise.allSettled(ids.map((id) => getJob(id)))
      if (!active) return
      const records = new Map<string, JobRecord>()
      const lost = new Set<string>()
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') records.set(ids[index], result.value)
        else if (isMissing(result.reason)) lost.add(ids[index])
      })
      if (records.size === 0 && lost.size === 0) return
      setJobs((current) =>
        current.map((job) => {
          const record = records.get(job.record.id)
          if (record) return mergeRecord(job, record)
          if (lost.has(job.record.id)) return markLost(job)
          return job
        })
      )
    }
    void pull()
    const timer = window.setInterval(() => void pull(), 2000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [connected, activeIds])

  useJobStream(selectedIds.video, connected, setJobs)
  useJobStream(selectedIds.image, connected, setJobs)

  return { jobs, setJobs, selectedIds, selectJob }
}

// Follows one job's event stream. The server replays the job's history on every
// connect, so each connection rebuilds the log from scratch.
function useJobStream(jobId: string | null, connected: boolean, setJobs: SetJobs): void {
  const [streamNonce, setStreamNonce] = useState(0)
  const retries = useRef(0)

  useEffect(() => {
    retries.current = 0
  }, [jobId])

  useEffect(() => {
    if (!jobId || !connected) return
    let active = true
    let finished = false
    const logs: string[] = []
    let offset = 0
    let synced = false
    let queue: ProgressEvent[] = []
    let timer = 0
    let retryTimer = 0

    const flush = (): void => {
      timer = 0
      if (!active || queue.length === 0) return
      const batch = queue
      queue = []
      let added = 0
      for (const payload of batch) {
        if (payload.type === 'log' && payload.message && !isProtocolLine(payload.message)) {
          logs.push(payload.message)
          added += 1
        }
      }
      if (logs.length > LOG_LIMIT) {
        const dropped = logs.length - LOG_LIMIT
        logs.splice(0, dropped)
        offset += dropped
      }
      // Most batches only carry progress. Keep the same log array then, so the
      // log view does not re-render or re-scroll for nothing.
      const snapshot = added > 0 || !synced ? logs.slice() : null
      synced = true
      const logOffset = offset
      setJobs((current) =>
        current.map((job) => {
          if (job.record.id !== jobId) return job
          let next = job
          for (const payload of batch) {
            if (payload.job_id === jobId) next = applyProgress(next, payload)
          }
          return snapshot ? { ...next, logs: snapshot, logOffset } : next
        })
      )
    }

    const schedule = (): void => {
      if (timer !== 0) return
      timer = window.setTimeout(flush, 150)
    }

    const source = new EventSource(`${API_BASE}/jobs/${jobId}/events`)

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

    const reconnect = (): void => {
      const delay = Math.min(10_000, 1000 * 2 ** retries.current)
      retries.current += 1
      retryTimer = window.setTimeout(() => {
        if (active) setStreamNonce((value) => value + 1)
      }, delay)
    }

    for (const type of EVENT_TYPES) source.addEventListener(type, handle)
    source.onopen = () => {
      retries.current = 0
    }
    source.onerror = () => {
      source.close()
      if (timer !== 0) window.clearTimeout(timer)
      flush()
      if (!active || finished) return
      // Keep showing the current log while reconnecting. The next connection
      // replays the history and replaces it.
      void getJob(jobId).then(
        (record) => {
          if (!active) return
          setJobs((current) =>
            current.map((job) => (job.record.id === jobId ? mergeRecord(job, record) : job))
          )
          reconnect()
        },
        (error: unknown) => {
          if (!active) return
          if (isMissing(error)) {
            setJobs((current) =>
              current.map((job) => (job.record.id === jobId ? markLost(job) : job))
            )
            return
          }
          reconnect()
        }
      )
    }

    return () => {
      active = false
      if (timer !== 0) window.clearTimeout(timer)
      if (retryTimer !== 0) window.clearTimeout(retryTimer)
      source.close()
    }
  }, [connected, jobId, streamNonce, setJobs])
}

export async function cancelSessionJob(
  jobId: string,
  onJobsChange: (updater: (current: SessionJob[]) => SessionJob[]) => void
): Promise<void> {
  const record = await cancelJob(jobId)
  // The API answers before the worker has stopped, so a running job can come
  // back still marked running. The event stream or the poll delivers the end.
  onJobsChange((current) =>
    current.map((job) => {
      if (job.record.id !== record.id) return job
      if (!isTerminal(record.status)) return { ...job, message: '正在取消' }
      return { ...mergeRecord(job, record), message: '' }
    })
  )
}
