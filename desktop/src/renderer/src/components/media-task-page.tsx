import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ApiError, cancelJob, createJob } from '@/lib/api'
import {
  fileName,
  formatTime,
  isSimpleProperty,
  isTerminal,
  paramLabel,
  stageLabel,
  statusLabel,
  statusVariant,
  suggestOutput
} from '@/lib/format'
import type {
  EngineManifest,
  MediaKind,
  SchemaProperty,
  SessionJob,
  SystemReport
} from '@/lib/types'
import type { PickKind } from '../../../shared/desktop'

interface MediaTaskPageProps {
  media: MediaKind
  connected: boolean
  projectRoot?: string
  system: SystemReport | null
  engines: EngineManifest[]
  jobs: SessionJob[]
  selectedId: string | null
  onJobsChange: (updater: (current: SessionJob[]) => SessionJob[]) => void
  onSelectJob: (jobId: string) => void
}

interface FieldGroup {
  id: string
  title: string
  fields: Array<[string, SchemaProperty]>
}

function fieldTitle(name: string, property: SchemaProperty): string {
  return paramLabel(name, property.title)
}

function appliesTo(property: SchemaProperty, media: MediaKind): boolean {
  return !property.media || property.media.length === 0 || property.media.includes(media)
}

function visibleFields(
  engine: EngineManifest | undefined,
  media: MediaKind
): Array<[string, SchemaProperty]> {
  return Object.entries(engine?.params_schema.properties ?? {}).filter(
    ([, property]) => isSimpleProperty(property) && appliesTo(property, media)
  )
}

function groupedFields(
  engine: EngineManifest | undefined,
  fields: Array<[string, SchemaProperty]>
): FieldGroup[] {
  const declared = engine?.params_schema.groups ?? []
  const titles = new Map(declared.map((group) => [group.id, group.title]))
  const ids = declared.map((group) => group.id).filter((id) => id !== 'primary')
  for (const [, property] of fields) {
    const id = property.group ?? 'primary'
    if (id !== 'primary' && !ids.includes(id)) ids.push(id)
  }
  return ids
    .map((id) => ({
      id,
      title: titles.get(id) ?? id,
      fields: fields.filter(([, property]) => (property.group ?? 'primary') === id)
    }))
    .filter((group) => group.fields.length > 0)
}

export function MediaTaskPage({
  media,
  connected,
  projectRoot,
  system,
  engines,
  jobs,
  selectedId,
  onJobsChange,
  onSelectJob
}: MediaTaskPageProps): React.JSX.Element {
  const [inputPath, setInputPath] = useState('')
  const [outputPath, setOutputPath] = useState('')
  const [outputTouched, setOutputTouched] = useState(false)
  const [engineId, setEngineId] = useState('seedvr2')
  const [values, setValues] = useState<Record<string, string | boolean>>({})
  const [devices, setDevices] = useState<number[]>([])
  const [formError, setFormError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)

  const available = engines.filter((item) => (item.media ?? ['video']).includes(media))
  const engine = available.find((item) => item.id === engineId) ?? available[0]
  const fields = visibleFields(engine, media)
  const primaryFields = fields.filter(([, property]) => (property.group ?? 'primary') === 'primary')
  const groups = groupedFields(engine, fields)
  const pageJobs = jobs.filter((job) => (job.record.spec.media ?? 'video') === media)
  const selected = pageJobs.find((job) => job.record.id === selectedId) ?? null
  const progressValue = Math.round((selected?.record.progress ?? 0) * 100)
  const outputFormat = selectedOutputFormat(media, values)
  const suggestedOutput = inputPath
    ? suggestOutput(inputPath, projectRoot ?? system?.project_root, media, outputFormat)
    : ''
  const shownOutput = outputTouched ? outputPath : suggestedOutput
  const inputLabel = media === 'image' ? '输入图片' : '输入视频'
  const outputKind: PickKind =
    media === 'image' ? 'image' : outputFormat === 'png' ? 'directory' : 'video'

  useEffect(() => {
    const viewport = logRef.current?.parentElement
    if (viewport) viewport.scrollTop = viewport.scrollHeight
  }, [selected?.logs])

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
    setOutputPath(picked)
  }

  function changeEngine(value: string): void {
    setEngineId(value)
    setValues({})
  }

  function buildParams(): Record<string, unknown> | null {
    const params: Record<string, unknown> = {}
    for (const [key, property] of fields) {
      if (!(key in values)) continue
      const value = values[key]
      const title = fieldTitle(key, property)
      if (property.type === 'boolean') {
        const checked = Boolean(value)
        if (checked !== Boolean(property.default)) params[key] = checked
        continue
      }
      if (typeof value !== 'string' || value.trim() === '') continue
      if (property.enum) {
        if (value !== property.default) params[key] = value
        continue
      }
      if (property.type === 'integer' || property.type === 'number') {
        const numeric = Number(value)
        if (!Number.isFinite(numeric)) {
          setFormError(`${title} 不是有效数字`)
          return null
        }
        if (property.type === 'integer' && !Number.isInteger(numeric)) {
          setFormError(`${title} 需要是整数`)
          return null
        }
        if (property.minimum != null && numeric < property.minimum) {
          setFormError(`${title} 不能小于 ${property.minimum}`)
          return null
        }
        if (property.maximum != null && numeric > property.maximum) {
          setFormError(`${title} 不能大于 ${property.maximum}`)
          return null
        }
        if (numeric !== property.default) params[key] = numeric
        continue
      }
      if (value !== property.default) params[key] = value
    }
    return params
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
    const params = buildParams()
    if (!params) return
    setSubmitting(true)
    try {
      const record = await createJob({
        engine_id: engine.id,
        input_path: inputPath.trim(),
        output_path: shownOutput.trim(),
        media,
        ...(devices.length > 0 ? { devices } : {}),
        params
      })
      onJobsChange((current) => [{ record, logs: [], message: '已提交' }, ...current])
      onSelectJob(record.id)
      toast.success(media === 'image' ? '已开始处理图片' : '已开始修复')
    } catch (error) {
      const message = error instanceof ApiError ? error.message : '提交失败'
      setFormError(message)
    } finally {
      setSubmitting(false)
    }
  }

  async function cancelSelected(): Promise<void> {
    if (!selected) return
    setCancelling(true)
    try {
      const record = await cancelJob(selected.record.id)
      onJobsChange((current) =>
        current.map((job) =>
          job.record.id === record.id ? { ...job, record, message: '已取消' } : job
        )
      )
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '取消失败')
    } finally {
      setCancelling(false)
    }
  }

  function toggleDevice(index: number, checked: boolean): void {
    setDevices((current) => {
      const next = checked ? [...current, index] : current.filter((item) => item !== index)
      return [...new Set(next)].sort((left, right) => left - right)
    })
  }

  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
      <Card>
        <CardHeader>
          <CardTitle>{media === 'image' ? '新建图片任务' : '新建任务'}</CardTitle>
          <CardDescription>
            {engine?.description ??
              (media === 'image'
                ? '选择图片后提交到本地修复引擎。未改动的参数沿用服务端默认值。'
                : '选择视频后提交到本地修复引擎。未改动的参数沿用服务端默认值。')}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {formError ? (
            <Alert variant="destructive">
              <AlertTitle>无法提交</AlertTitle>
              <AlertDescription>{formError}</AlertDescription>
            </Alert>
          ) : null}
          <PathField
            id={`${media}-input-path`}
            label={inputLabel}
            value={inputPath}
            placeholder={media === 'image' ? '选择或粘贴图片路径' : '选择或粘贴视频路径'}
            onChange={setInputPath}
            onBrowse={() => void browseInput()}
          />
          <PathField
            id={`${media}-output-path`}
            label={outputFormat === 'png' && media === 'video' ? '输出目录' : '输出文件'}
            value={shownOutput}
            placeholder={
              media === 'image' ? 'output/图片名.seedvr2.png' : 'output/视频名.seedvr2.mp4'
            }
            hint={
              media === 'image'
                ? '默认保存为 PNG，透明通道会保留。'
                : outputFormat === 'png'
                  ? '每一帧会写成 PNG，目录里不再封装音轨。'
                  : undefined
            }
            onChange={(value) => {
              setOutputTouched(true)
              setOutputPath(value)
            }}
            onBrowse={() => void browseOutput()}
          />
          <div className="flex flex-col gap-2">
            <Label>引擎</Label>
            {engine ? (
              <Select value={engine.id} onValueChange={changeEngine} disabled={!connected}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="选择引擎" />
                </SelectTrigger>
                <SelectContent position="popper">
                  {available.map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input disabled placeholder={connected ? '没有可用引擎' : '等待服务'} />
            )}
          </div>
          {primaryFields.length > 0 ? (
            <div className="grid gap-4 sm:grid-cols-2">
              {primaryFields.map(([key, property]) => (
                <SchemaField
                  key={`${engine?.id ?? 'engine'}-${key}`}
                  name={key}
                  property={property}
                  value={values[key]}
                  onChange={(value) => setValues((current) => ({ ...current, [key]: value }))}
                />
              ))}
            </div>
          ) : null}
          <div className="flex flex-col gap-2">
            <Label>显卡</Label>
            {system && system.gpus.length > 0 ? (
              <div className="flex flex-col gap-2">
                {system.gpus.map((gpu) => (
                  <label key={gpu.index} className="flex items-center gap-2 text-sm">
                    <Checkbox
                      checked={devices.includes(gpu.index)}
                      onCheckedChange={(checked) => toggleDevice(gpu.index, checked === true)}
                    />
                    <span>
                      [{gpu.index}] {gpu.name}
                    </span>
                  </label>
                ))}
                <p className="text-xs text-muted-foreground">不选则由服务端自动选择设备。</p>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                连接服务后可以选择显卡，不选则自动决定。
              </p>
            )}
          </div>
          {groups.map((group) => (
            <Collapsible key={group.id}>
              <CollapsibleTrigger asChild>
                <Button type="button" variant="outline" size="sm" className="w-fit">
                  {group.title}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="grid gap-4 pt-4 sm:grid-cols-2">
                {group.fields.map(([key, property]) => (
                  <SchemaField
                    key={`${engine?.id ?? 'engine'}-${key}`}
                    name={key}
                    property={property}
                    value={values[key]}
                    onChange={(value) => setValues((current) => ({ ...current, [key]: value }))}
                  />
                ))}
              </CollapsibleContent>
            </Collapsible>
          ))}
          <Button disabled={!connected || submitting || !engine} onClick={() => void submit()}>
            {submitting ? '正在提交' : media === 'image' ? '开始处理' : '开始修复'}
          </Button>
        </CardContent>
      </Card>

      <div className="flex flex-col gap-4">
        <Card>
          <CardHeader>
            <CardTitle>当前任务</CardTitle>
            <CardDescription>
              {selected ? fileName(selected.record.spec.input_path) : '提交后在这里查看进度和日志'}
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {!selected ? (
              <p className="py-8 text-center text-sm text-muted-foreground">还没有任务</p>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={statusVariant(selected.record.status)}>
                    {statusLabel(selected.record.status)}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {stageLabel(selected.record.stage)}
                  </span>
                  <span className="ml-auto font-mono text-xs text-muted-foreground">
                    {selected.record.id}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <Progress value={progressValue} />
                  <span className="w-10 text-right text-xs text-muted-foreground">
                    {progressValue}%
                  </span>
                </div>
                {selected.message ? <p className="text-sm">{selected.message}</p> : null}
                {selected.record.error ? (
                  <Alert variant="destructive">
                    <AlertTitle>失败</AlertTitle>
                    <AlertDescription>{selected.record.error}</AlertDescription>
                  </Alert>
                ) : null}
                <ScrollArea className="h-64 rounded-lg border">
                  <div ref={logRef} className="space-y-1 p-3 font-mono text-xs leading-5">
                    {selected.logs.length === 0 ? (
                      <p className="text-muted-foreground">等待输出</p>
                    ) : (
                      selected.logs.map((line, index) => <p key={`${index}-${line}`}>{line}</p>)
                    )}
                  </div>
                </ScrollArea>
                <div className="flex justify-end">
                  <Button
                    variant="outline"
                    disabled={cancelling || isTerminal(selected.record.status)}
                    onClick={() => void cancelSelected()}
                  >
                    {cancelling ? '正在取消' : '取消'}
                  </Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>
        {pageJobs.length > 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>本次会话</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2">
              {pageJobs.map((job) => (
                <button
                  key={job.record.id}
                  type="button"
                  className="flex items-center gap-3 rounded-lg border px-3 py-2 text-left hover:bg-muted/50"
                  onClick={() => onSelectJob(job.record.id)}
                >
                  <Badge variant={statusVariant(job.record.status)}>
                    {statusLabel(job.record.status)}
                  </Badge>
                  <span className="min-w-0 flex-1 truncate text-sm">
                    {fileName(job.record.spec.input_path)}
                  </span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {formatTime(job.record.created_at)}
                  </span>
                </button>
              ))}
            </CardContent>
          </Card>
        ) : null}
      </div>
    </div>
  )
}

function selectedOutputFormat(
  media: MediaKind,
  values: Record<string, string | boolean>
): 'mp4' | 'png' {
  if (media === 'image') return 'png'
  return values.output_format === 'png' ? 'png' : 'mp4'
}

function PathField({
  id,
  label,
  value,
  placeholder,
  hint,
  onChange,
  onBrowse
}: {
  id: string
  label: string
  value: string
  placeholder: string
  hint?: string
  onChange: (value: string) => void
  onBrowse: () => void
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={id}>{label}</Label>
      <div className="flex gap-2">
        <Input
          id={id}
          value={value}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
        <Button type="button" variant="outline" onClick={onBrowse}>
          浏览
        </Button>
      </div>
      {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  )
}

function SchemaField({
  name,
  property,
  value,
  onChange
}: {
  name: string
  property: SchemaProperty
  value: string | boolean | undefined
  onChange: (value: string | boolean) => void
}): React.JSX.Element {
  const title = fieldTitle(name, property)
  if (property.type === 'boolean') {
    const checked = typeof value === 'boolean' ? value : Boolean(property.default)
    return (
      <div className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2 sm:col-span-2">
        <div>
          <Label htmlFor={`param-${name}`}>{title}</Label>
          {property.description ? (
            <p className="text-xs text-muted-foreground">{property.description}</p>
          ) : null}
        </div>
        <Switch
          id={`param-${name}`}
          checked={checked}
          onCheckedChange={(checkedValue) => onChange(checkedValue === true)}
        />
      </div>
    )
  }
  if (property.enum && property.enum.length > 0) {
    const current =
      typeof value === 'string' && property.enum.includes(value)
        ? value
        : typeof property.default === 'string' && property.enum.includes(property.default)
          ? property.default
          : property.enum[0]
    return (
      <div className="flex flex-col gap-2">
        <Label>{title}</Label>
        <Select value={current} onValueChange={onChange}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper">
            {property.enum.map((option) => (
              <SelectItem key={option} value={option}>
                {property.enum_labels?.[option] ?? option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {property.description ? (
          <p className="text-xs text-muted-foreground">{property.description}</p>
        ) : null}
      </div>
    )
  }
  const text =
    typeof value === 'string'
      ? value
      : property.default != null && property.default !== ''
        ? String(property.default)
        : ''
  const placeholder = text === '' ? '留空使用默认' : undefined
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={`param-${name}`}>{title}</Label>
      <Input
        id={`param-${name}`}
        type={property.type === 'string' ? 'text' : 'number'}
        min={property.minimum}
        max={property.maximum}
        step={property.type === 'number' ? 'any' : 1}
        value={text}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
      {property.description ? (
        <p className="text-xs text-muted-foreground">{property.description}</p>
      ) : null}
    </div>
  )
}
