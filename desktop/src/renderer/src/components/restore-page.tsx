import { MediaTaskPage } from '@/components/media-task-page'
import type { EngineManifest, SessionJob, SystemReport } from '@/lib/types'

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

export function RestorePage(props: RestorePageProps): React.JSX.Element {
  return <MediaTaskPage media="video" {...props} />
}
