import { useState } from 'react'
import { toast } from 'sonner'

import { FrameCompare } from '@/components/frame-compare'
import { JobLog } from '@/components/job-log'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cancelSessionJob } from '@/hooks/use-session-jobs'
import {
  fileName,
  formatTime,
  isTerminal,
  stageLabel,
  statusLabel,
  statusVariant
} from '@/lib/format'
import type { SessionJob } from '@/lib/types'
import { cn } from 'cn'

interface TasksPageProps {
  jobs: SessionJob[]
  selectedId: string | null
  onJobsChange: (updater: (current: SessionJob[]) => SessionJob[]) => void
  onSelectJob: (jobId: string) => void
}

export function TasksPage({
  jobs,
  selectedId,
  onJobsChange,
  onSelectJob
}: TasksPageProps): React.JSX.Element {
  const [cancellingId, setCancellingId] = useState<string | null>(null)
  const videoJobs = jobs.filter((job) => (job.record.spec.media ?? 'video') === 'video')
  const selected = videoJobs.find((job) => job.record.id === selectedId) ?? null

  if (videoJobs.length === 0) {
    return (
      <div className="flex min-h-64 flex-1 items-center justify-center rounded-xl border border-dashed">
        <p className="max-w-sm text-center text-sm text-muted-foreground">
          还没有视频修复任务。在修复页提交后，进度、日志和画面对比会显示在这里。
        </p>
      </div>
    )
  }

  return (
    <div className="grid min-h-0 flex-1 grid-rows-[minmax(0,0.38fr)_minmax(0,1fr)] gap-4 overflow-hidden lg:grid-cols-[18rem_minmax(0,1fr)] lg:grid-rows-1">
      <Card className="min-h-0 overflow-hidden">
        <CardHeader className="shrink-0">
          <CardTitle>本次会话</CardTitle>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 px-0">
          <ScrollArea className="h-full">
            <div className="flex flex-col gap-2 px-4 pb-1">
              {videoJobs.map((job) => {
                const active = job.record.id === selected?.record.id
                const progressValue = Math.round((job.record.progress ?? 0) * 100)
                return (
                  <button
                    key={job.record.id}
                    type="button"
                    className={cn(
                      'flex flex-col gap-2 rounded-lg border px-3 py-2 text-left hover:bg-muted/50',
                      active && 'border-primary bg-muted/60'
                    )}
                    onClick={() => onSelectJob(job.record.id)}
                  >
                    <div className="flex items-center gap-2">
                      <Badge variant={statusVariant(job.record.status)}>
                        {statusLabel(job.record.status)}
                      </Badge>
                      <span className="ml-auto shrink-0 text-xs text-muted-foreground">
                        {formatTime(job.record.created_at)}
                      </span>
                    </div>
                    <span className="truncate text-sm">{fileName(job.record.spec.input_path)}</span>
                    <div className="flex items-center gap-2">
                      <Progress value={progressValue} />
                      <span className="w-8 shrink-0 text-right text-xs text-muted-foreground">
                        {progressValue}%
                      </span>
                    </div>
                  </button>
                )
              })}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {selected ? (
        <TaskDetail
          job={selected}
          cancelling={cancellingId === selected.record.id}
          onCancel={() => {
            setCancellingId(selected.record.id)
            void cancelSessionJob(selected.record.id, onJobsChange)
              .catch((error: unknown) => {
                toast.error(error instanceof Error ? error.message : '取消失败')
              })
              .finally(() => setCancellingId(null))
          }}
        />
      ) : (
        <div className="flex h-full min-h-40 items-center justify-center rounded-xl border border-dashed">
          <p className="text-sm text-muted-foreground">选择左侧任务查看日志和画面对比</p>
        </div>
      )}
    </div>
  )
}

function TaskDetail({
  job,
  cancelling,
  onCancel
}: {
  job: SessionJob
  cancelling: boolean
  onCancel: () => void
}): React.JSX.Element {
  const progressValue = Math.round((job.record.progress ?? 0) * 100)
  return (
    <Card className="min-h-0 overflow-hidden">
      <CardHeader className="shrink-0">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle className="truncate">{fileName(job.record.spec.input_path)}</CardTitle>
            <p className="truncate font-mono text-xs text-muted-foreground">{job.record.id}</p>
          </div>
          <Button
            variant="outline"
            disabled={cancelling || isTerminal(job.record.status)}
            onClick={onCancel}
          >
            {cancelling ? '正在取消' : '取消'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={statusVariant(job.record.status)}>{statusLabel(job.record.status)}</Badge>
          <span className="text-sm text-muted-foreground">{stageLabel(job.record.stage)}</span>
        </div>
        <div className="flex items-center gap-3">
          <Progress value={progressValue} />
          <span className="w-10 text-right text-xs text-muted-foreground">{progressValue}%</span>
        </div>
        {job.message ? <p className="text-sm">{job.message}</p> : null}
        {job.previewFrame != null || !isTerminal(job.record.status) ? (
          <FrameCompare jobId={job.record.id} frame={job.previewFrame ?? null} />
        ) : null}
        {job.record.error ? (
          <Alert variant="destructive">
            <AlertTitle>失败</AlertTitle>
            <AlertDescription>{job.record.error}</AlertDescription>
          </Alert>
        ) : null}
        <JobLog lines={job.logs} className="min-h-32 flex-1" />
      </CardContent>
    </Card>
  )
}
