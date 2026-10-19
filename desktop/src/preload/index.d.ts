import type { DesktopApi } from '../shared/desktop'

declare global {
  interface Window {
    api?: DesktopApi
  }
}

export {}
