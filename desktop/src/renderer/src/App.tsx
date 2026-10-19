import { useCallback, useEffect, useState } from 'react'

import { AppShell, type AppPage } from '@/components/app-shell'
import { RestorePage } from '@/components/restore-page'
import { SystemPage } from '@/components/system-page'
import { Toaster } from '@/components/ui/sonner'
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
  const { jobs, setJobs, selectedId, selectJob } = useSessionJobs(connected)

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
    <>
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
        <div className={page === 'restore' ? 'block' : 'hidden'}>
          <RestorePage
            connected={connected}
            projectRoot={server?.projectRoot ?? system?.project_root}
            system={system}
            engines={engines}
            jobs={jobs}
            selectedId={selectedId}
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
    </>
  )
}

export default App
