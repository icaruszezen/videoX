export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export type EventType =
  'queued' | 'stage' | 'log' | 'progress' | 'completed' | 'failed' | 'cancelled'

export interface JobSpec {
  engine_id: string
  input_path: string
  output_path: string
  devices: number[] | null
  params: Record<string, unknown>
}

export interface JobRecord {
  id: string
  spec: JobSpec
  status: JobStatus
  created_at: string
  error: string | null
  progress: number | null
  stage: string | null
}

export interface ProgressEvent {
  job_id: string
  type: EventType
  message: string
  progress: number | null
  stage: string | null
}

export interface GpuReport {
  index: number
  name: string
  memory_mib: number
}

export interface SystemReport {
  python: string
  project_root: string
  torch: string | null
  cuda_available: boolean
  ffmpeg: string | null
  ffprobe: string | null
  gpus: GpuReport[]
  seedvr2_present: boolean
  seedvr2_commit: string | null
  seedvr2_commit_expected: string | null
  model_dir: string
  models_present: string[]
  models_missing: string[]
  ready: boolean
  warnings: string[]
}

export interface SchemaProperty {
  type?: string
  default?: string | number | boolean
  minimum?: number
  description?: string
}

export interface EngineManifest {
  id: string
  name: string
  description: string
  capabilities: string[]
  params_schema: {
    type?: string
    properties?: Record<string, SchemaProperty>
  }
}

export interface SessionJob {
  record: JobRecord
  logs: string[]
  message: string
}
