export interface ServerState {
  running: boolean
  owned: boolean
  baseUrl: string
  projectRoot: string
  error?: string
}

export interface DesktopApi {
  pickInput: () => Promise<string | null>
  pickOutput: (defaultPath: string) => Promise<string | null>
  serverStatus: () => Promise<ServerState>
  serverStart: () => Promise<ServerState>
  serverStop: () => Promise<ServerState>
}
