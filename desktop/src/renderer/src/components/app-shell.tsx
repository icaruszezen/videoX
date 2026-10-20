import type { ReactNode } from 'react'
import { ClapperboardIcon, CpuIcon, ImageIcon } from 'lucide-react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from 'cn'
import type { ServerState } from '../../../shared/desktop'

export type AppPage = 'restore' | 'image' | 'system'

interface AppShellProps {
  page: AppPage
  onPageChange: (page: AppPage) => void
  connected: boolean
  server: ServerState | null
  starting: boolean
  onServerChange: (state: ServerState) => void
  onStartingChange: (starting: boolean) => void
  children: ReactNode
}

export function AppShell({
  page,
  onPageChange,
  connected,
  server,
  starting,
  onServerChange,
  onStartingChange,
  children
}: AppShellProps): React.JSX.Element {
  const owned = Boolean(server?.owned && server.running)

  async function startServer(): Promise<void> {
    if (!window.api) {
      toast.error('请在 VideoX 桌面窗口中启动服务，或先在终端运行 videox serve')
      return
    }
    onStartingChange(true)
    try {
      const state = await window.api.serverStart()
      onServerChange(state)
      if (state.error) toast.error(state.error)
      else if (state.running) toast.success('本地服务已就绪')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '启动服务失败')
    } finally {
      onStartingChange(false)
    }
  }

  async function stopServer(): Promise<void> {
    if (!window.api) return
    onStartingChange(true)
    try {
      const state = await window.api.serverStop()
      onServerChange(state)
      if (state.error) toast.error(state.error)
      else toast.success('已停止本地服务')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '停止服务失败')
    } finally {
      onStartingChange(false)
    }
  }

  return (
    <div className="flex h-full bg-background text-foreground">
      <aside className="flex w-60 shrink-0 flex-col border-r bg-card">
        <div className="px-4 py-5">
          <div className="text-lg font-semibold tracking-tight">VideoX</div>
          <div className="text-xs text-muted-foreground">本地视频修复</div>
        </div>
        <nav className="flex flex-col gap-1 px-3">
          <Button
            variant={page === 'restore' ? 'secondary' : 'ghost'}
            className="justify-start"
            onClick={() => onPageChange('restore')}
          >
            <ClapperboardIcon data-icon="inline-start" />
            修复
          </Button>
          <Button
            variant={page === 'image' ? 'secondary' : 'ghost'}
            className="justify-start"
            onClick={() => onPageChange('image')}
          >
            <ImageIcon data-icon="inline-start" />
            图片
          </Button>
          <Button
            variant={page === 'system' ? 'secondary' : 'ghost'}
            className="justify-start"
            onClick={() => onPageChange('system')}
          >
            <CpuIcon data-icon="inline-start" />
            系统
          </Button>
        </nav>
        <div className="mt-auto space-y-3 p-4">
          <div className="flex items-center justify-between gap-2">
            <Badge variant={connected ? 'default' : 'outline'}>
              {connected ? '已连接' : '未连接'}
            </Badge>
            {owned ? <span className="text-xs text-muted-foreground">本应用启动</span> : null}
            {server?.running && !server.owned ? (
              <span className="text-xs text-muted-foreground">外部服务</span>
            ) : null}
          </div>
          <p className="text-xs text-muted-foreground">127.0.0.1:8765</p>
          {owned ? (
            <Button
              className="w-full"
              variant="outline"
              disabled={starting}
              onClick={() => void stopServer()}
            >
              {starting ? '正在停止' : '停止服务'}
            </Button>
          ) : (
            <Button
              className="w-full"
              disabled={starting || connected}
              onClick={() => void startServer()}
            >
              {starting ? '正在启动' : connected ? '服务已在运行' : '启动服务'}
            </Button>
          )}
        </div>
      </aside>
      <main className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center justify-between border-b px-6">
          <h1 className="text-base font-medium">
            {page === 'restore' ? '修复' : page === 'image' ? '图片' : '系统'}
          </h1>
          <span className={cn('text-xs text-muted-foreground')}>
            {connected ? 'API 可用' : '等待本地 API'}
          </span>
        </header>
        <div className="min-h-0 flex-1 overflow-auto p-6">{children}</div>
      </main>
    </div>
  )
}
