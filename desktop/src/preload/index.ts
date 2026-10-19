import { contextBridge, ipcRenderer } from 'electron'

import type { DesktopApi } from '../shared/desktop'

const api: DesktopApi = {
  pickInput: () => ipcRenderer.invoke('dialog:pick-input'),
  pickOutput: (defaultPath) => ipcRenderer.invoke('dialog:pick-output', defaultPath),
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
