import { contextBridge, ipcRenderer } from 'electron'

import type { DesktopApi } from '../shared/desktop'

const api: DesktopApi = {
  pickInput: (kind = 'video') => ipcRenderer.invoke('dialog:pick-input', kind),
  pickOutput: (defaultPath, kind = 'video') =>
    ipcRenderer.invoke('dialog:pick-output', defaultPath, kind),
  serverStatus: () => ipcRenderer.invoke('server:status'),
  serverStart: () => ipcRenderer.invoke('server:start'),
  serverStop: () => ipcRenderer.invoke('server:stop')
}

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
}
