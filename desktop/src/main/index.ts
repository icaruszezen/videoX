import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron'
import { mkdirSync, readFileSync, writeFileSync } from 'fs'
import { dirname, join } from 'path'
import { electronApp, is, optimizer } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

import { findProjectRoot, getServerState, killOwnedServer, startServer, stopServer } from './server'

// The restore job fills the NVIDIA device. Sharing that device for window
// compositing makes Windows drop the D3D context and the GPU process exits
// with code 34, which freezes the window while CUDA keeps running.
app.disableHardwareAcceleration()

const JOB_ID = /^[a-f0-9]{12}$/

function sessionJobFile(): string {
  return join(findProjectRoot(), 'output', '.session-jobs.json')
}

function readSessionJobIds(): string[] {
  try {
    const parsed = JSON.parse(readFileSync(sessionJobFile(), 'utf8')) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed.filter((item): item is string => typeof item === 'string' && JOB_ID.test(item))
  } catch {
    return []
  }
}

function writeSessionJobIds(ids: unknown): void {
  if (!Array.isArray(ids)) return
  const clean = ids.filter((item): item is string => typeof item === 'string' && JOB_ID.test(item))
  if (clean.length === 0) return
  const file = sessionJobFile()
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, JSON.stringify(clean))
}

let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 800,
    minWidth: 960,
    minHeight: 640,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#18181b',
    title: 'VideoX',
    icon,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      backgroundThrottling: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function registerIpc(): void {
  ipcMain.handle('dialog:pick-input', async (_event, kind: unknown) => {
    const image = kind === 'image'
    const options = {
      title: image ? '选择要处理的图片' : '选择要修复的视频',
      properties: ['openFile'] as Array<'openFile'>,
      filters: image
        ? [{ name: '图片', extensions: ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'] }]
        : [{ name: '视频', extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v'] }]
    }
    const result = mainWindow
      ? await dialog.showOpenDialog(mainWindow, options)
      : await dialog.showOpenDialog(options)
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0]
  })

  ipcMain.handle('dialog:pick-output', async (_event, defaultPath: unknown, kind: unknown) => {
    const fallback = typeof defaultPath === 'string' ? defaultPath : undefined
    if (kind === 'directory') {
      const options = {
        title: '选择 PNG 序列目录',
        defaultPath: fallback,
        properties: ['openDirectory', 'createDirectory'] as Array<
          'openDirectory' | 'createDirectory'
        >
      }
      const result = mainWindow
        ? await dialog.showOpenDialog(mainWindow, options)
        : await dialog.showOpenDialog(options)
      if (result.canceled || result.filePaths.length === 0) return null
      return result.filePaths[0]
    }
    const image = kind === 'image'
    const options = {
      title: image ? '保存处理后的图片' : '保存修复结果',
      defaultPath: fallback,
      filters: image
        ? [
            { name: 'PNG', extensions: ['png'] },
            { name: 'JPEG', extensions: ['jpg', 'jpeg'] },
            { name: 'WebP', extensions: ['webp'] },
            { name: 'BMP', extensions: ['bmp'] },
            { name: 'TIFF', extensions: ['tif', 'tiff'] }
          ]
        : [{ name: 'MP4', extensions: ['mp4'] }]
    }
    const result = mainWindow
      ? await dialog.showSaveDialog(mainWindow, options)
      : await dialog.showSaveDialog(options)
    if (result.canceled || !result.filePath) return null
    return result.filePath
  })

  ipcMain.handle('server:status', () => getServerState())
  ipcMain.handle('server:start', () => startServer())
  ipcMain.handle('server:stop', () => stopServer())
  ipcMain.handle('session:read-jobs', () => readSessionJobIds())
  ipcMain.handle('session:write-jobs', (_event, ids: unknown) => {
    writeSessionJobIds(ids)
  })
}

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.videox.desktop')
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })
  registerIpc()
  createWindow()
  let lastGpuReload = 0
  app.on('child-process-gone', (_event, details) => {
    if (details.type !== 'GPU' || details.reason === 'clean-exit') return
    const now = Date.now()
    if (now - lastGpuReload < 15_000) return
    const crashedWindow = mainWindow
    if (!crashedWindow || crashedWindow.isDestroyed()) return
    lastGpuReload = now
    crashedWindow.webContents.reload()
  })
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('before-quit', () => {
  killOwnedServer()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
