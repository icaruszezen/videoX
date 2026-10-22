import type { JobStatus, MediaKind, ProgressEvent, SchemaProperty, SessionJob } from './types'

export const EVENT_TYPES = [
  'queued',
  'stage',
  'log',
  'progress',
  'preview',
  'completed',
  'failed',
  'cancelled'
] as const

const STAGE_LABELS: Record<string, string> = {
  queued: '排队',
  prepare: '读取视频',
  restore: '修复',
  encoding: '编码',
  upscaling: '放大',
  decoding: '解码',
  post: '后处理',
  chunk: '分块',
  save: '保存',
  audio: '封装音轨',
  done: '完成'
}

export const PARAM_LABELS: Record<string, string> = {
  resolution: '短边分辨率',
  batch_size: '批大小',
  dit_model: '扩散模型',
  temporal_overlap: '时间重叠',
  prepend_frames: '前置帧',
  vae_encode_tiled: 'VAE 编码分块',
  vae_decode_tiled: 'VAE 解码分块',
  uniform_batch_size: '统一批大小',
  chunk_size: '分块大小',
  color_correction: '色彩校正',
  seed: '随机种子',
  attention_mode: '注意力模式',
  blocks_to_swap: '换出块数',
  dit_offload_device: '卸载设备'
}

export function paramLabel(key: string, title?: string): string {
  if (title) return title
  return PARAM_LABELS[key] ?? key
}

export function stageLabel(stage: string | null | undefined): string {
  if (!stage) return '等待中'
  return STAGE_LABELS[stage] ?? stage
}

export function statusLabel(status: JobStatus): string {
  switch (status) {
    case 'queued':
      return '排队中'
    case 'running':
      return '进行中'
    case 'completed':
      return '已完成'
    case 'failed':
      return '失败'
    case 'cancelled':
      return '已取消'
  }
}

export function statusVariant(
  status: JobStatus
): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case 'completed':
      return 'default'
    case 'failed':
      return 'destructive'
    case 'running':
      return 'secondary'
    default:
      return 'outline'
  }
}

export function fileName(filePath: string): string {
  const name = filePath.split(/[/\\]/).pop()
  return name && name.length > 0 ? name : filePath
}

export function fileStem(filePath: string): string {
  const name = fileName(filePath)
  const dot = name.lastIndexOf('.')
  return dot > 0 ? name.slice(0, dot) : name
}

export function suggestOutput(
  inputPath: string,
  projectRoot?: string,
  media: MediaKind = 'video',
  outputFormat: 'mp4' | 'png' = 'mp4',
  engineId = 'seedvr2'
): string {
  const stem = fileStem(inputPath)
  const tag = engineId || 'seedvr2'
  const name =
    media === 'image'
      ? `${stem}.${tag}.png`
      : outputFormat === 'png'
        ? stem
        : `${stem}.${tag}.mp4`
  const root = projectRoot?.replace(/[/\\]+$/, '')
  if (root) {
    const sep = root.includes('\\') ? '\\' : '/'
    return `${root}${sep}output${sep}${name}`
  }
  const sep = inputPath.includes('\\') ? '\\' : '/'
  const dir = inputPath.replace(/[/\\][^/\\]*$/, '')
  return `${dir}${sep}${name}`
}

export function formatTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString('zh-CN', { hour12: false })
}

export function formatMemory(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GB`
  return `${mib} MiB`
}

export function recommendedBatchSize(
  gpus: Array<{ index: number; memory_mib: number }>,
  selectedDevices: number[]
): number {
  const pool =
    selectedDevices.length > 0 ? gpus.filter((gpu) => selectedDevices.includes(gpu.index)) : gpus
  if (pool.length === 0) return 5
  const smallest = Math.min(...pool.map((gpu) => gpu.memory_mib))
  return smallest >= 15 * 1024 ? 9 : 5
}

export function shortCommit(commit: string | null): string {
  if (!commit) return '未知'
  return commit.slice(0, 8)
}

export function isTerminal(status: JobStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}

export function applyProgress(job: SessionJob, event: ProgressEvent, logs: string[]): SessionJob {
  const record = { ...job.record }
  if (event.stage) record.stage = event.stage
  if (event.progress != null) record.progress = event.progress
  if (event.type === 'queued') record.status = 'queued'
  if (event.type === 'completed') {
    record.status = 'completed'
    record.progress = event.progress ?? 1
    record.error = null
  }
  if (event.type === 'failed') {
    record.status = 'failed'
    record.error = event.message || '修复失败'
  }
  if (event.type === 'cancelled') {
    record.status = 'cancelled'
    record.error = null
  }
  if (
    (event.type === 'stage' ||
      event.type === 'log' ||
      event.type === 'progress' ||
      event.type === 'preview') &&
    record.status === 'queued'
  ) {
    record.status = 'running'
  }
  const message = event.type === 'log' ? job.message : event.message || job.message
  const previewFrame =
    event.type === 'preview' && event.preview_frame != null
      ? event.preview_frame
      : (job.previewFrame ?? null)
  return { record, logs, message, previewFrame }
}

export function isSimpleProperty(property: SchemaProperty): boolean {
  return (
    property.type === 'string' ||
    property.type === 'integer' ||
    property.type === 'number' ||
    property.type === 'boolean'
  )
}
