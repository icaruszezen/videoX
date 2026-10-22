import { useState } from 'react'
import { toast } from 'sonner'

import { FrameCompare } from '@/components/frame-compare'
import { JobLog } from '@/components/job-log'
import { JobSummary } from '@/components/job-summary'
import { EngineSelect, GpuPicker, PathField, SchemaField } from '@/components/param-fields'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { cancelSessionJob } from '@/hooks/use-session-jobs'
import { useMediaTaskForm } from '@/hooks/use-media-task-form'
import { fileName, formatTime, isTerminal, statusLabel, statusVariant } from '@/lib/format'
import { gpuHelp, outputHelp } from '@/lib/task-schema'
import type { EngineManifest, MediaKind, SessionJob, SystemReport } from '@/lib/types'
import { cn } from 'cn'

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
  const [cancelling, setCancelling] = useState(false)
  const form = useMediaTaskForm({
    media,
    connected,
    projectRoot,
    system,
    engines,
    onJobsChange,
    onSelectJob
  })
  const pageJobs = jobs.filter((job) => (job.record.spec.media ?? 'video') === media)
  const selected = pageJobs.find((job) => job.record.id === selectedId) ?? null
  const pngSequence = media === 'video' && form.outputFormat === 'png'

  async function cancelSelected(): Promise<void> {
    if (!selected) return
    setCancelling(true)
    try {
      await cancelSessionJob(selected.record.id, onJobsChange)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '取消失败')
    } finally {
      setCancelling(false)
    }
  }

  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
      <Card>
        <CardHeader>
          <CardTitle>{media === 'image' ? '新建图片任务' : '新建任务'}</CardTitle>
          <CardDescription>
            {form.engine?.description ??
              (media === 'image'
                ? '选择图片后提交到本地修复引擎。未改动的参数沿用服务端默认值。'
                : '选择视频后提交到本地修复引擎。未改动的参数沿用服务端默认值。')}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {form.formError ? (
            <Alert variant="destructive">
              <AlertTitle>无法提交</AlertTitle>
              <AlertDescription>{form.formError}</AlertDescription>
            </Alert>
          ) : null}
          <PathField
            id={`${media}-input-path`}
            label={form.inputLabel}
            help={
              media === 'image'
                ? '要处理的原图。可以浏览选择，也可以直接粘贴路径。'
                : '要修复的原视频。可以浏览选择，也可以直接粘贴路径。'
            }
            value={form.inputPath}
            placeholder={media === 'image' ? '选择或粘贴图片路径' : '选择或粘贴视频路径'}
            onChange={form.setInputPath}
            onBrowse={() => void form.browseInput()}
          />
          <PathField
            id={`${media}-output-path`}
            label={pngSequence ? '输出目录' : '输出文件'}
            help={outputHelp(media, form.outputFormat, form.engine?.id)}
            value={form.shownOutput}
            placeholder={
              media === 'image'
                ? `output/图片名.${form.engine?.id ?? 'seedvr2'}.png`
                : pngSequence
                  ? 'output/视频名'
                  : `output/视频名.${form.engine?.id ?? 'seedvr2'}.mp4`
            }
            hint={pngSequence ? '每一帧会写成 PNG，目录里不再封装音轨。' : undefined}
            onChange={form.setOutputPath}
            onBrowse={() => void form.browseOutput()}
          />
          <EngineSelect
            engine={form.engine}
            available={form.available}
            connected={connected}
            onChange={form.changeEngine}
          />
          {form.primaryFields.length > 0 ? (
            <div className="grid gap-4 sm:grid-cols-2">
              {form.primaryFields.map(([key, property]) => (
                <SchemaField
                  key={`${form.engine?.id ?? 'engine'}-${key}`}
                  name={key}
                  property={property}
                  value={form.values[key]}
                  suggested={key === 'batch_size' ? form.batchSizeSuggestion : undefined}
                  onChange={(value) => form.setField(key, value)}
                />
              ))}
            </div>
          ) : null}
          <GpuPicker
            gpus={system?.gpus ?? []}
            devices={form.devices}
            help={gpuHelp(form.engine?.device_policy, media)}
            onToggle={form.toggleDevice}
          />
          {form.advancedGroups.map((group) => (
            <Collapsible key={group.id}>
              <CollapsibleTrigger asChild>
                <Button type="button" variant="outline" size="sm" className="w-fit">
                  {group.title}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="grid gap-4 pt-4 sm:grid-cols-2">
                {group.fields.map(([key, property]) => (
                  <SchemaField
                    key={`${form.engine?.id ?? 'engine'}-${key}`}
                    name={key}
                    property={property}
                    value={form.values[key]}
                    suggested={key === 'batch_size' ? form.batchSizeSuggestion : undefined}
                    onChange={(value) => form.setField(key, value)}
                  />
                ))}
              </CollapsibleContent>
            </Collapsible>
          ))}
          <Button
            disabled={!connected || form.submitting || !form.engine}
            onClick={() => void form.submit()}
          >
            {form.submitting ? '正在提交' : media === 'image' ? '开始处理' : '开始修复'}
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
                <JobSummary job={selected} />
                {selected.previewFrame != null || !isTerminal(selected.record.status) ? (
                  <FrameCompare jobId={selected.record.id} frame={selected.previewFrame ?? null} />
                ) : null}
                <JobLog
                  key={selected.record.id}
                  lines={selected.logs}
                  offset={selected.logOffset}
                  finished={isTerminal(selected.record.status)}
                  className="h-64"
                />
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
                  aria-current={job.record.id === selected?.record.id ? 'true' : undefined}
                  className={cn(
                    'flex items-center gap-3 rounded-lg border px-3 py-2 text-left outline-none hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50',
                    job.record.id === selected?.record.id &&
                      'border-primary/70 bg-muted/60 hover:bg-muted/60'
                  )}
                  onClick={() => onSelectJob(job.record.id)}
                >
                  <Badge variant={statusVariant(job.record.status)}>
                    {statusLabel(job.record.status)}
                  </Badge>
                  <span
                    className="min-w-0 flex-1 truncate text-sm"
                    title={job.record.spec.input_path}
                  >
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
