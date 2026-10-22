export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export type EventType =
  'queued' | 'stage' | 'log' | 'progress' | 'preview' | 'completed' | 'failed' | 'cancelled'

export type MediaKind = 'video' | 'image'

export interface JobSpec {
  engine_id: string
  input_path: string
  output_path: string
  media?: MediaKind
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
  preview_frame?: number | null
}

export interface GpuReport {
  index: number
  name: string
  memory_mib: number
}

export interface EngineStatus {
  id: string
  name: string
  present: boolean
  commit: string | null
  commit_expected: string | null
  venv_ready: boolean
  model_dir: string
  models_present: string[]
  models_missing: string[]
  detail: string
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
  engines?: EngineStatus[]
  ready: boolean
  warnings: string[]
}

export interface SchemaProperty {
  type?: string
  title?: string
  default?: string | number | boolean
  minimum?: number
  maximum?: number
  description?: string
  enum?: string[]
  enum_labels?: Record<string, string>
  group?: string
  media?: MediaKind[]
}

export interface SchemaGroup {
  id: string
  title: string
}

export interface EngineManifest {
  id: string
  name: string
  description: string
  capabilities: string[]
  media?: MediaKind[]
  device_policy?: 'multi' | 'single'
  params_schema: {
    type?: string
    properties?: Record<string, SchemaProperty>
    groups?: SchemaGroup[]
  }
}

export interface SessionJob {
  record: JobRecord
  logs: string[]
  // Number of older log lines dropped from the front of `logs`.
  logOffset?: number
  message: string
  previewFrame?: number | null
}
