import { spawn, spawnSync, type ChildProcess } from 'child_process'
import { existsSync } from 'fs'
import { join, resolve } from 'path'
import { app } from 'electron'

import type { ServerState } from '../shared/desktop'

export const API_HOST = '127.0.0.1'
export const API_PORT = 8765
export const API_BASE = `http://${API_HOST}:${API_PORT}`

let serverProcess: ChildProcess | null = null
let serverOwned = false

export function findProjectRoot(): string {
  const candidates = [
    process.cwd(),
    resolve(process.cwd(), '..'),
    app.getAppPath(),
    resolve(app.getAppPath(), '..')
  ]
  for (const candidate of candidates) {
    if (existsSync(join(candidate, 'pyproject.toml'))) return candidate
  }
  return resolve(app.getAppPath(), '..')
}

export async function probeApi(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/system`, { signal: AbortSignal.timeout(2500) })
    return response.ok
  } catch {
    return false
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolveSleep) => {
    setTimeout(resolveSleep, ms)
  })
}

function serveLaunch(root: string): { command: string; args: string[] } | null {
  const scripts = join(root, '.venv', 'Scripts')
  const serveArgs = ['serve', '--host', API_HOST, '--port', String(API_PORT)]
  const videoxName = process.platform === 'win32' ? 'videox.exe' : 'videox'
  const videox = join(scripts, videoxName)
  if (existsSync(videox)) return { command: videox, args: serveArgs }
  const pythonName = process.platform === 'win32' ? 'python.exe' : 'python'
  const python = join(scripts, pythonName)
  if (existsSync(python)) return { command: python, args: ['-m', 'videox', ...serveArgs] }
  return null
}

export function killOwnedServer(): void {
  const child = serverProcess
  serverProcess = null
  serverOwned = false
  if (child?.pid == null) return
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/F', '/T', '/PID', String(child.pid)], { windowsHide: true })
    return
  }
  try {
    child.kill('SIGTERM')
  } catch {
    // The process has already exited.
  }
}

async function currentState(error?: string): Promise<ServerState> {
  const running = await probeApi()
  if (!running && serverProcess?.exitCode != null) {
    serverProcess = null
    serverOwned = false
  }
  return {
    running,
    owned: running && serverOwned,
    baseUrl: API_BASE,
    projectRoot: findProjectRoot(),
    error
  }
}

export function getServerState(): Promise<ServerState> {
  return currentState()
}

export async function startServer(): Promise<ServerState> {
  if (await probeApi()) return currentState()

  const root = findProjectRoot()
  const launch = serveLaunch(root)
  if (!launch) {
    return currentState('找不到 .venv 里的 videox，请先运行 scripts\\setup.ps1')
  }

  const child = spawn(launch.command, launch.args, {
    cwd: root,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  serverProcess = child
  serverOwned = true
  let stderr = ''
  let spawnError = ''
  child.stdout?.on('data', () => {
    // Drain the pipe so the process cannot block on a full buffer.
  })
  child.stderr?.on('data', (chunk: Buffer) => {
    stderr = (stderr + chunk.toString()).slice(-2000)
  })
  child.on('error', (error) => {
    spawnError = error.message
  })
  child.on('exit', () => {
    if (serverProcess === child) {
      serverProcess = null
      serverOwned = false
    }
  })

  const deadline = Date.now() + 60_000
  while (Date.now() < deadline) {
    if (spawnError || child.exitCode != null) {
      serverProcess = null
      serverOwned = false
      const detail = spawnError || stderr.trim() || `服务进程已退出（${child.exitCode ?? '未知'}）`
      return currentState(detail)
    }
    if (await probeApi()) return currentState()
    await sleep(400)
  }

  killOwnedServer()
  return currentState('启动服务超时')
}

export async function stopServer(): Promise<ServerState> {
  if (!serverOwned) {
    const running = await probeApi()
    return {
      running,
      owned: false,
      baseUrl: API_BASE,
      projectRoot: findProjectRoot(),
      error: running ? '当前服务不是由本应用启动的' : undefined
    }
  }
  killOwnedServer()
  await sleep(300)
  return currentState()
}
