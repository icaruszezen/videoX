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
import type { EngineManifest, SchemaProperty, SessionJob, SystemReport } from '@/lib/types'

const PRIMARY_PARAMS = new Set(['resolution', 'batch_size'])

interface RestorePageProps {
  connected: boolean
  projectRoot?: string
  system: SystemReport | null
  engines: EngineManifest[]
  jobs: SessionJob[]
  selectedId: string | null
  onJobsChange: (updater: (current: SessionJob[]) => SessionJob[]) => void
  onSelectJob: (jobId: string) => void
}

function propertyDefault(property: SchemaProperty | undefined): string {
  if (!property || property.default == null || typeof property.default === 'boolean') return ''
  return String(property.default)
}

export function RestorePage({
  connected,
  projectRoot,
  system,
  engines,
  jobs,
  selectedId,
  onJobsChange,
  onSelectJob
}: RestorePageProps): React.JSX.Element {
  const [inputPath, setInputPath] = useState('')
  const [outputPath, setOutputPath] = useState('')
  const [outputTouched, setOutputTouched] = useState(false)
  const [engineId, setEngineId] = useState('seedvr2')
  const [resolution, setResolution] = useState('')
  const [batchSize, setBatchSize] = useState('')
  const [devices, setDevices] = useState<number[]>([])
  const [extra, setExtra] = useState<Record<string, string | boolean>>({})
  const [formError, setFormError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)

  const engine = engines.find((item) => item.id === engineId) ?? engines[0]
  const properties = engine?.params_schema.properties ?? {}
  const selected = jobs.find((job) => job.record.id === selectedId) ?? null
  const progressValue = Math.round((selected?.record.progress ?? 0) * 100)
  const resolutionDefault = properties.resolution?.default
  const shownResolution =
    resolution !== ''
      ? resolution
      : typeof resolutionDefault === 'number'
        ? String(resolutionDefault)
        : ''
  const suggestedOutput = inputPath
    ? suggestOutput(inputPath, projectRoot ?? system?.project_root)
    : ''
  const shownOutput = outputTouched ? outputPath : suggestedOutput

  useEffect(() => {
    const viewport = logRef.current?.parentElement
    if (viewport) viewport.scrollTop = viewport.scrollHeight
  }, [selected?.logs])

  async function browseInput(): Promise<void> {
    if (!window.api) {
      toast.error('文件选择需要在 VideoX 桌面窗口中使用，也可以直接粘贴路径')
      return
    }
    const picked = await window.api.pickInput()
    if (picked) setInputPath(picked)
  }

  async function browseOutput(): Promise<void> {
    const suggested = shownOutput || suggestedOutput || 'restored.mp4'
    if (!window.api) {
      toast.error('文件选择需要在 VideoX 桌面窗口中使用，也可以直接粘贴路径')
      return
    }
    const picked = await window.api.pickOutput(suggested)
    if (!picked) return
    setOutputTouched(true)
    setOutputPath(picked)
  }

  function changeEngine(value: string): void {
    setEngineId(value)
    setExtra({})
    setBatchSize('')
    const next = engines.find((item) => item.id === value)
    const defaultResolution = next?.params_schema.properties?.resolution?.default
    setResolution(typeof defaultResolution === 'number' ? String(defaultResolution) : '')
  }

  function toggleDevice(index: number, checked: boolean): void {
    setDevices((current) => {
      const next = checked ? [...current, index] : current.filter((item) => item !== index)
      return [...new Set(next)].sort((left, right) => left - right)
    })
  }

  function buildParams(): Record<string, unknown> | null {
    const params: Record<string, unknown> = {}
    const resolutionDefault = properties.resolution?.default
    if (resolution.trim() !== '') {
      const value = Number(resolution)
      if (!Number.isInteger(value) || value < 1) {
        setFormError('短边分辨率需要是正整数')
        return null
      }
      if (value !== resolutionDefault) params.resolution = value
    }
    if (batchSize.trim() !== '') {
      const value = Number(batchSize)
      if (!Number.isInteger(value) || value < 1) {
        setFormError('批大小需要是正整数')
        return null
      }
      params.batch_size = value
    }
    for (const [key, property] of Object.entries(properties)) {
      if (PRIMARY_PARAMS.has(key) || !isSimpleProperty(property) || !(key in extra)) continue
      const value = extra[key]
      if (property.type === 'boolean') {
        const checked = Boolean(value)
        if (checked !== Boolean(property.default)) params[key] = checked
        continue
      }
      if (typeof value !== 'string' || value.trim() === '') continue
      if (property.type === 'integer' || property.type === 'number') {
        const numeric = Number(value)
        if (!Number.isFinite(numeric)) {
          setFormError(`${paramLabel(key)} 不是有效数字`)
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
    if (!inputPath.trim() || !shownOutput.trim()) {
      setFormError('请填写输入视频和输出文件')
      return
    }
    const params = buildParams()
    if (!params) return
    setSubmitting(true)
    try {
      const record = await createJob({
        engine_id: engine?.id ?? engineId,
        input_path: inputPath.trim(),
        output_path: shownOutput.trim(),
        ...(devices.length > 0 ? { devices } : {}),
        params
      })
      onJobsChange((current) => [{ record, logs: [], message: '已提交' }, ...current])
      onSelectJob(record.id)
      toast.success('已开始修复')
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

  const extraEntries = Object.entries(properties).filter(
    ([key, property]) => !PRIMARY_PARAMS.has(key) && isSimpleProperty(property)
  )

  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
      <Card>
        <CardHeader>
          <CardTitle>新建任务</CardTitle>
          <CardDescription>
            {engine?.description ?? '选择视频后提交到本地修复引擎。未改动的参数沿用服务端默认值。'}
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
            id="input-path"
            label="输入视频"
            value={inputPath}
            placeholder="选择或粘贴视频路径"
            onChange={setInputPath}
            onBrowse={() => void browseInput()}
          />
          <PathField
            id="output-path"
            label="输出文件"
            value={shownOutput}
            placeholder="output/视频名.seedvr2.mp4"
            onChange={(value) => {
              setOutputTouched(true)
              setOutputPath(value)
            }}
            onBrowse={() => void browseOutput()}
          />
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-2">
              <Label>引擎</Label>
              {engine ? (
                <Select value={engine.id} onValueChange={changeEngine} disabled={!connected}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="选择引擎" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    {engines.map((item) => (
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
            <div className="flex flex-col gap-2">
              <Label htmlFor="resolution">短边分辨率</Label>
              <Input
                id="resolution"
                type="number"
                min={1}
                value={shownResolution}
                placeholder={propertyDefault(properties.resolution) || '1080'}
                onChange={(event) => setResolution(event.target.value)}
              />
            </div>
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="batch-size">批大小</Label>
            <Input
              id="batch-size"
              type="number"
              min={1}
              value={batchSize}
              placeholder="留空则按显存自动选择"
              onChange={(event) => setBatchSize(event.target.value)}
            />
            {properties.batch_size?.description ? (
              <p className="text-xs text-muted-foreground">{properties.batch_size.description}</p>
            ) : null}
          </div>
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
          {extraEntries.length > 0 ? (
            <Collapsible>
              <CollapsibleTrigger asChild>
                <Button variant="outline" size="sm" className="w-fit">
                  更多参数
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="grid gap-4 pt-4 sm:grid-cols-2">
                {extraEntries.map(([key, property]) => (
                  <ExtraField
                    key={key}
                    name={key}
                    property={property}
                    value={extra[key]}
                    onChange={(value) => setExtra((current) => ({ ...current, [key]: value }))}
                  />
                ))}
              </CollapsibleContent>
            </Collapsible>
          ) : null}
          <Button disabled={!connected || submitting} onClick={() => void submit()}>
            {submitting ? '正在提交' : '开始修复'}
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
        {jobs.length > 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>本次会话</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2">
              {jobs.map((job) => (
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

function PathField({
  id,
  label,
  value,
  placeholder,
  onChange,
  onBrowse
}: {
  id: string
  label: string
  value: string
  placeholder: string
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
    </div>
  )
}

function ExtraField({
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
  if (property.type === 'boolean') {
    const checked = typeof value === 'boolean' ? value : Boolean(property.default)
    return (
      <div className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2 sm:col-span-2">
        <div>
          <Label htmlFor={`param-${name}`}>{paramLabel(name)}</Label>
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
  const text = typeof value === 'string' ? value : ''
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={`param-${name}`}>{paramLabel(name)}</Label>
      <Input
        id={`param-${name}`}
        type={property.type === 'string' ? 'text' : 'number'}
        value={text}
        placeholder={propertyDefault(property) || '默认'}
        onChange={(event) => onChange(event.target.value)}
      />
      {property.description ? (
        <p className="text-xs text-muted-foreground">{property.description}</p>
      ) : null}
    </div>
  )
}
