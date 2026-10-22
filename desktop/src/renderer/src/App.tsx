import { useCallback, useEffect, useState } from 'react'

import { AppShell, type AppPage } from '@/components/app-shell'
import { ImagePage } from '@/components/image-page'
import { RestorePage } from '@/components/restore-page'
import { SystemPage } from '@/components/system-page'
import { TasksPage } from '@/components/tasks-page'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'
import { getEngines, getSystem } from '@/lib/api'
import { useSessionJobs } from '@/hooks/use-session-jobs'
import type { EngineManifest, SystemReport } from '@/lib/types'
import type { ServerState } from '../../shared/desktop'

function App(): React.JSX.Element {
  const [page, setPage] = useState<AppPage>('restore')
  const [connected, setConnected] = useState(false)
  const [system, setSystem] = useState<SystemReport | null>(null)
  const [engines, setEngines] = useState<EngineManifest[]>([])
  const [server, setServer] = useState<ServerState | null>(null)
  const [starting, setStarting] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  const { jobs, setJobs, selectedIds, selectJob } = useSessionJobs(connected)

  const refresh = useCallback(async (): Promise<void> => {
    if (window.api) {
      try {
        setServer(await window.api.serverStatus())
      } catch {
        // The desktop bridge can be unavailable while the window is reloading.
      }
    }
    try {
      setSystem(await getSystem())
      setConnected(true)
    } catch {
      setConnected(false)
      return
    }
    try {
      setEngines(await getEngines())
    } catch {
      // Keep the last engine list when a refresh fails.
    }
  }, [])

  useEffect(() => {
    const kickoff = window.setTimeout(() => void refresh(), 0)
    const timer = window.setInterval(() => void refresh(), 5000)
    return () => {
      window.clearTimeout(kickoff)
      window.clearInterval(timer)
    }
  }, [refresh, reloadKey])

  return (
    <TooltipProvider>
      <AppShell
        page={page}
        onPageChange={setPage}
        connected={connected}
        server={server}
        starting={starting}
        onServerChange={(state) => {
          setServer(state)
          setReloadKey((value) => value + 1)
        }}
        onStartingChange={setStarting}
      >
        <div className={page === 'restore' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
          <RestorePage
            connected={connected}
            projectRoot={server?.projectRoot ?? system?.project_root}
            system={system}
            engines={engines}
            onJobsChange={setJobs}
            onSelectJob={selectJob}
            onOpenTasks={() => setPage('tasks')}
          />
        </div>
        <div className={page === 'tasks' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
          <TasksPage
            jobs={jobs}
            selectedId={selectedIds.video}
            onJobsChange={setJobs}
            onSelectJob={selectJob}
          />
        </div>
        <div className={page === 'image' ? 'block' : 'hidden'}>
          <ImagePage
            connected={connected}
            projectRoot={server?.projectRoot ?? system?.project_root}
            system={system}
            engines={engines}
            jobs={jobs}
            selectedId={selectedIds.image}
            onJobsChange={setJobs}
            onSelectJob={selectJob}
          />
        </div>
        <div className={page === 'system' ? 'block' : 'hidden'}>
          <SystemPage
            connected={connected}
            system={system}
            onRefresh={() => setReloadKey((value) => value + 1)}
          />
        </div>
      </AppShell>
      <Toaster />
    </TooltipProvider>
  )
}

export default App
