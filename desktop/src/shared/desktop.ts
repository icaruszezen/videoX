export interface ServerState {
  running: boolean
  owned: boolean
  baseUrl: string
  projectRoot: string
  error?: string
}

export type PickKind = 'video' | 'image' | 'directory'

export interface DesktopApi {
  pickInput: (kind?: 'video' | 'image') => Promise<string | null>
  pickOutput: (defaultPath: string, kind?: PickKind) => Promise<string | null>
  serverStatus: () => Promise<ServerState>
  serverStart: () => Promise<ServerState>
  serverStop: () => Promise<ServerState>
  readJobIds: () => Promise<string[]>
  writeJobIds: (ids: string[]) => Promise<void>
}
