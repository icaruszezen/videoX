import { useState } from 'react'
import { toast } from 'sonner'

import { ApiError, createJob } from '@/lib/api'
import { recommendedBatchSize, suggestOutput } from '@/lib/format'
import {
  collectParams,
  schemaGroups,
  selectedOutputFormat,
  valuesAfterChange,
  visibleFields,
  type FieldGroup
} from '@/lib/task-schema'
import type {
  EngineManifest,
  MediaKind,
  SchemaProperty,
  SessionJob,
  SystemReport
} from '@/lib/types'
import type { PickKind } from '../../../shared/desktop'

interface UseMediaTaskFormOptions {
  media: MediaKind
  connected: boolean
  projectRoot?: string
  system: SystemReport | null
  engines: EngineManifest[]
  onJobsChange: (updater: (current: SessionJob[]) => SessionJob[]) => void
  onSelectJob: (jobId: string) => void
  onSubmitted?: (jobId: string) => void
}

export interface MediaTaskFormState {
  inputPath: string
  setInputPath: (value: string) => void
  shownOutput: string
  setOutputPath: (value: string) => void
  outputFormat: 'mp4' | 'png'
  inputLabel: string
  outputKind: PickKind
  engine: EngineManifest | undefined
  available: EngineManifest[]
  fields: Array<[string, SchemaProperty]>
  primaryFields: Array<[string, SchemaProperty]>
  advancedGroups: FieldGroup[]
  parameterGroups: FieldGroup[]
  values: Record<string, string | boolean>
  setField: (key: string, value: string | boolean) => void
  devices: number[]
  toggleDevice: (index: number, checked: boolean) => void
  formError: string | null
  submitting: boolean
  batchSizeSuggestion: number
  browseInput: () => Promise<void>
  browseOutput: () => Promise<void>
  changeEngine: (value: string) => void
  submit: () => Promise<void>
}

export function useMediaTaskForm({
  media,
  connected,
  projectRoot,
  system,
  engines,
  onJobsChange,
  onSelectJob,
  onSubmitted
}: UseMediaTaskFormOptions): MediaTaskFormState {
  const [inputPath, setInputPath] = useState('')
  const [outputPath, setOutputPathState] = useState('')
  const [outputTouched, setOutputTouched] = useState(false)
  const [engineId, setEngineId] = useState('seedvr2')
  const [values, setValues] = useState<Record<string, string | boolean>>({})
  const [devices, setDevices] = useState<number[]>([])
  const [formError, setFormError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const available = engines.filter((item) => (item.media ?? ['video']).includes(media))
  const engine = available.find((item) => item.id === engineId) ?? available[0]
  const fields = visibleFields(engine, media, values)
  const primaryFields = fields.filter(([, property]) => (property.group ?? 'primary') === 'primary')
  const advancedGroups = schemaGroups(engine, fields, ['primary'])
  const parameterGroups = schemaGroups(engine, fields)
  const outputFormat = selectedOutputFormat(media, values)
  const suggestedOutput = inputPath
    ? suggestOutput(
        inputPath,
        projectRoot ?? system?.project_root,
        media,
        outputFormat,
        engine?.id ?? 'seedvr2'
      )
    : ''
  const shownOutput = outputTouched ? outputPath : suggestedOutput
  const inputLabel = media === 'image' ? '输入图片' : '输入视频'
  const outputKind: PickKind =
    media === 'image' ? 'image' : outputFormat === 'png' ? 'directory' : 'video'
  const batchSizeSuggestion = recommendedBatchSize(system?.gpus ?? [], devices)

  function setOutputPath(value: string): void {
    setOutputTouched(true)
    setOutputPathState(value)
  }

  function setField(key: string, value: string | boolean): void {
    // A path picked for one output format is a file for MP4 and a directory for PNG,
    // so switching formats goes back to the suggested path.
    if (selectedOutputFormat(media, { ...values, [key]: value }) !== outputFormat) {
      setOutputTouched(false)
    }
    // Fields whose defaults follow this one (usually the model choice) move to
    // the new default unless the user picked a value that still fits.
    setValues((current) => valuesAfterChange(engine, media, current, key, value))
  }

  function changeEngine(value: string): void {
    const next = available.find((item) => item.id === value)
    setEngineId(value)
    setValues({})
    if (selectedOutputFormat(media, {}) !== outputFormat) setOutputTouched(false)
    if (next?.device_policy === 'single') {
      setDevices((current) => (current.length > 1 ? current.slice(0, 1) : current))
    }
  }

  function toggleDevice(index: number, checked: boolean): void {
    if (media === 'image' || engine?.device_policy === 'single') {
      setDevices(checked ? [index] : [])
      return
    }
    setDevices((current) => {
      const next = checked ? [...current, index] : current.filter((item) => item !== index)
      return [...new Set(next)].sort((left, right) => left - right)
    })
  }

  async function browseInput(): Promise<void> {
    if (!window.api) {
      toast.error('文件选择需要在 VideoX 桌面窗口中使用，也可以直接粘贴路径')
      return
    }
    const picked = await window.api.pickInput(media)
    if (picked) setInputPath(picked)
  }

  async function browseOutput(): Promise<void> {
    const suggested =
      shownOutput || suggestedOutput || (media === 'image' ? 'image.seedvr2.png' : 'restored.mp4')
    if (!window.api) {
      toast.error('文件选择需要在 VideoX 桌面窗口中使用，也可以直接粘贴路径')
      return
    }
    const picked = await window.api.pickOutput(suggested, outputKind)
    if (!picked) return
    setOutputTouched(true)
    setOutputPathState(picked)
  }

  async function submit(): Promise<void> {
    setFormError(null)
    if (!connected) {
      setFormError('本地服务未连接')
      return
    }
    if (!engine) {
      setFormError('没有可用的引擎')
      return
    }
    if (!inputPath.trim() || !shownOutput.trim()) {
      setFormError(media === 'image' ? '请填写输入图片和输出文件' : '请填写输入视频和输出文件')
      return
    }
    const built = collectParams(fields, values)
    if ('error' in built) {
      setFormError(built.error)
      return
    }
    setSubmitting(true)
    try {
      const record = await createJob({
        engine_id: engine.id,
        input_path: inputPath.trim(),
        output_path: shownOutput.trim(),
        media,
        ...(devices.length > 0 ? { devices } : {}),
        params: built.params
      })
      onJobsChange((current) => [{ record, logs: [], message: '已提交' }, ...current])
      onSelectJob(record.id)
      toast.success(media === 'image' ? '已开始处理图片' : '已开始修复')
      onSubmitted?.(record.id)
    } catch (error) {
      const message = error instanceof ApiError ? error.message : '提交失败'
      setFormError(message)
    } finally {
      setSubmitting(false)
    }
  }

  return {
    inputPath,
    setInputPath,
    shownOutput,
    setOutputPath,
    outputFormat,
    inputLabel,
    outputKind,
    engine,
    available,
    fields,
    primaryFields,
    advancedGroups,
    parameterGroups,
    values,
    setField,
    devices,
    toggleDevice,
    formError,
    submitting,
    batchSizeSuggestion,
    browseInput,
    browseOutput,
    changeEngine,
    submit
  }
}
