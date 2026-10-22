import { EngineSelect, GpuPicker, PathField, SchemaField } from '@/components/param-fields'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useMediaTaskForm } from '@/hooks/use-media-task-form'
import { gpuHelp, outputHelp } from '@/lib/task-schema'
import type { EngineManifest, SessionJob, SystemReport } from '@/lib/types'

interface RestorePageProps {
  connected: boolean
  projectRoot?: string
  system: SystemReport | null
  engines: EngineManifest[]
  onJobsChange: (updater: (current: SessionJob[]) => SessionJob[]) => void
  onSelectJob: (jobId: string) => void
  onOpenTasks: () => void
}

export function RestorePage({
  connected,
  projectRoot,
  system,
  engines,
  onJobsChange,
  onSelectJob,
  onOpenTasks
}: RestorePageProps): React.JSX.Element {
  const form = useMediaTaskForm({
    media: 'video',
    connected,
    projectRoot,
    system,
    engines,
    onJobsChange,
    onSelectJob,
    onSubmitted: onOpenTasks
  })
  const pngSequence = form.outputFormat === 'png'

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="min-h-0 flex-1 overflow-auto pr-1">
        <div className="flex flex-col gap-4 pb-1">
          <Card>
            <CardHeader>
              <CardTitle>素材</CardTitle>
              <CardDescription>
                {form.engine?.description ??
                  '选择视频后提交到本地修复引擎。未改动的参数沿用服务端默认值。'}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="grid gap-4 lg:grid-cols-2">
                <PathField
                  id="video-input-path"
                  label={form.inputLabel}
                  help="要修复的原视频。可以浏览选择，也可以直接粘贴路径。"
                  value={form.inputPath}
                  placeholder="选择或粘贴视频路径"
                  onChange={form.setInputPath}
                  onBrowse={() => void form.browseInput()}
                />
                <PathField
                  id="video-output-path"
                  label={pngSequence ? '输出目录' : '输出文件'}
                  help={outputHelp('video', form.outputFormat, form.engine?.id)}
                  value={form.shownOutput}
                  placeholder={
                    pngSequence
                      ? 'output/视频名'
                      : `output/视频名.${form.engine?.id ?? 'seedvr2'}.mp4`
                  }
                  hint={pngSequence ? '每一帧会写成 PNG，目录里不再封装音轨。' : undefined}
                  onChange={form.setOutputPath}
                  onBrowse={() => void form.browseOutput()}
                />
              </div>
              <div className="grid items-start gap-4 lg:grid-cols-[minmax(14rem,20rem)_minmax(0,1fr)]">
                <EngineSelect
                  engine={form.engine}
                  available={form.available}
                  connected={connected}
                  onChange={form.changeEngine}
                />
                <GpuPicker
                  gpus={system?.gpus ?? []}
                  devices={form.devices}
                  layout="cards"
                  help={gpuHelp(form.engine?.device_policy, 'video')}
                  onToggle={form.toggleDevice}
                />
              </div>
            </CardContent>
          </Card>

          <div className="grid items-start gap-4 xl:grid-cols-2">
            {form.parameterGroups.map((group) => (
              <Card key={group.id} className="xl:last:odd:col-span-2">
                <CardHeader>
                  <CardTitle>{group.title}</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-4 sm:grid-cols-2">
                  {group.fields.map(([key, property]) => (
                    <SchemaField
                      key={`${form.engine?.id ?? 'engine'}-${key}`}
                      name={key}
                      property={property}
                      value={form.values[key]}
                      suggested={key === 'batch_size' ? form.batchSizeSuggestion : undefined}
                      idPrefix="video-"
                      onChange={(value) => form.setField(key, value)}
                    />
                  ))}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>

      <div className="flex shrink-0 flex-col gap-3 border-t pt-3">
        {form.formError ? (
          <Alert variant="destructive">
            <AlertTitle>无法提交</AlertTitle>
            <AlertDescription>{form.formError}</AlertDescription>
          </Alert>
        ) : null}
        <div className="flex justify-end">
          <Button
            disabled={!connected || form.submitting || !form.engine}
            onClick={() => void form.submit()}
          >
            {form.submitting ? '正在提交' : '开始修复'}
          </Button>
        </div>
      </div>
    </div>
  )
}
