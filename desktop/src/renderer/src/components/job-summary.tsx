import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { stageLabel, statusLabel, statusVariant } from '@/lib/format'
import type { SessionJob } from '@/lib/types'

// Status, progress, output path and failure reason of one job. The status line
// only carries the stage and engine message while the job runs; finished jobs
// say everything with the badge, the output path and the failure alert.
export function JobSummary({ job }: { job: SessionJob }): React.JSX.Element {
  const { record } = job
  const progressValue = Math.round((record.progress ?? 0) * 100)
  const running = record.status === 'running'
  const message = running ? job.message : ''

  return (
    <>
      <div className="flex min-w-0 items-center gap-2">
        <Badge variant={statusVariant(record.status)}>{statusLabel(record.status)}</Badge>
        {running ? (
          <span className="shrink-0 text-sm text-muted-foreground">
            {stageLabel(record.stage, record.spec.media)}
          </span>
        ) : null}
        {message ? (
          <span className="min-w-0 truncate text-xs text-muted-foreground" title={message}>
            {message}
          </span>
        ) : null}
      </div>
      <div className="flex items-center gap-3">
        <Progress value={progressValue} aria-label="任务进度" />
        <span className="w-10 shrink-0 text-right text-xs text-muted-foreground tabular-nums">
          {progressValue}%
        </span>
      </div>
      <p className="flex min-w-0 gap-2 text-xs text-muted-foreground">
        <span className="shrink-0">输出</span>
        <span className="min-w-0 font-mono wrap-anywhere">{record.spec.output_path}</span>
      </p>
      {record.error ? (
        <Alert variant="destructive">
          <AlertTitle>失败原因</AlertTitle>
          <AlertDescription className="wrap-anywhere">{record.error}</AlertDescription>
        </Alert>
      ) : null}
    </>
  )
}
