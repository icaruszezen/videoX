import { useState } from 'react'
import { toast } from 'sonner'

import { FrameCompare } from '@/components/frame-compare'
import { JobLog } from '@/components/job-log'
import { JobSummary } from '@/components/job-summary'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cancelSessionJob } from '@/hooks/use-session-jobs'
import {
  fileName,
  formatClock,
  formatTime,
  isTerminal,
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
    <div className="grid min-h-0 flex-1 grid-rows-[minmax(0,0.38fr)_minmax(0,1fr)] gap-4 overflow-hidden md:grid-cols-[15rem_minmax(0,1fr)] md:grid-rows-1 xl:grid-cols-[18rem_minmax(0,1fr)]">
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
                const name = fileName(job.record.spec.input_path)
                return (
                  <button
                    key={job.record.id}
                    type="button"
                    aria-current={active ? 'true' : undefined}
                    className={cn(
                      'flex min-w-0 flex-col gap-2 rounded-lg border px-3 py-2 text-left outline-none hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50',
                      active && 'border-primary/70 bg-muted hover:bg-muted'
                    )}
                    onClick={() => onSelectJob(job.record.id)}
                  >
                    <div className="flex items-center gap-2">
                      <Badge variant={statusVariant(job.record.status)}>
                        {statusLabel(job.record.status)}
                      </Badge>
                      <span
                        className="ml-auto shrink-0 text-xs text-muted-foreground tabular-nums"
                        title={formatTime(job.record.created_at)}
                      >
                        {formatClock(job.record.created_at)}
                      </span>
                    </div>
                    <span className="min-w-0 truncate text-sm" title={job.record.spec.input_path}>
                      {name}
                    </span>
                    <div className="flex items-center gap-2">
                      <Progress value={progressValue} aria-label={`${name} 进度`} />
                      <span className="w-8 shrink-0 text-right text-xs text-muted-foreground tabular-nums">
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
          cancelling={cancellingId === selected.record.id && !isTerminal(selected.record.status)}
          onCancel={() => {
            const jobId = selected.record.id
            setCancellingId(jobId)
            // Stay in the cancelling state until the job reports that it stopped.
            void cancelSessionJob(jobId, onJobsChange).catch((error: unknown) => {
              setCancellingId((current) => (current === jobId ? null : current))
              toast.error(error instanceof Error ? error.message : '取消失败')
            })
          }}
        />
      ) : (
        <div className="flex h-full min-h-40 items-center justify-center rounded-xl border border-dashed">
          <p className="text-sm text-muted-foreground">选择一个任务查看日志和画面对比</p>
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
  const finished = isTerminal(job.record.status)
  return (
    <Card className="min-h-0 overflow-hidden">
      <CardHeader className="shrink-0">
        <div className="flex min-w-0 items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle className="truncate" title={job.record.spec.input_path}>
              {fileName(job.record.spec.input_path)}
            </CardTitle>
            <p className="truncate text-xs text-muted-foreground">
              <span className="font-mono">{job.record.id}</span>
              <span className="mx-1.5">·</span>
              {formatTime(job.record.created_at)}
            </p>
          </div>
          <Button
            variant="outline"
            className="shrink-0"
            disabled={cancelling || finished}
            onClick={onCancel}
          >
            {cancelling ? '正在取消' : '取消'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
        <JobSummary job={job} />
        {job.previewFrame != null || !finished ? (
          <FrameCompare
            jobId={job.record.id}
            frame={job.previewFrame ?? null}
            paneClassName="h-28 [@media(min-height:720px)]:h-40"
          />
        ) : null}
        <JobLog
          key={job.record.id}
          lines={job.logs}
          offset={job.logOffset}
          finished={finished}
          className="min-h-40 flex-1"
        />
      </CardContent>
    </Card>
  )
}
