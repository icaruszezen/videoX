import { useEffect, useRef, useState } from 'react'
import { ZoomInIcon } from 'lucide-react'
import { Dialog as DialogPrimitive } from 'radix-ui'

import { Button } from '@/components/ui/button'
import { API_BASE } from '@/lib/api'

const MIN_SCALE = 1
const MAX_SCALE = 12

interface View {
  scale: number
  pan: { x: number; y: number }
}

interface FrameCompareProps {
  jobId: string
  frame: number | null
  // Height of the inline preview panes.
  paneClassName?: string
}

export function FrameCompare({
  jobId,
  frame,
  paneClassName = 'h-40'
}: FrameCompareProps): React.JSX.Element {
  const [failedFrame, setFailedFrame] = useState<number | null>(null)
  const [openJobId, setOpenJobId] = useState<string | null>(null)
  const view = useRef<View>({ scale: MIN_SCALE, pan: { x: 0, y: 0 } })
  const [shown, setShown] = useState<View>({ scale: MIN_SCALE, pan: { x: 0, y: 0 } })
  const areaRef = useRef<HTMLDivElement>(null)
  const sourceReady = frame == null || failedFrame !== frame
  const open = openJobId === jobId

  function applyView(next: View): void {
    view.current = next
    setShown(next)
  }

  useEffect(() => {
    const node = areaRef.current
    if (!node || !open) return
    const onWheel = (event: WheelEvent): void => {
      const pane = (event.target as Element | null)?.closest('[data-zoom-pane]')
      if (!(pane instanceof HTMLElement)) return
      event.preventDefault()
      const rect = pane.getBoundingClientRect()
      const next = zoomAt(view.current, {
        x: event.clientX - rect.left - rect.width / 2,
        y: event.clientY - rect.top - rect.height / 2,
        deltaY: event.deltaY
      })
      applyView(next)
    }
    node.addEventListener('wheel', onWheel, { passive: false })
    return () => node.removeEventListener('wheel', onWheel)
  }, [open])

  if (frame == null) {
    return (
      <p className="rounded-lg border border-dashed px-3 py-6 text-center text-sm text-muted-foreground">
        处理过程中会在这里显示对比画面（部分模型在结束时才生成）
      </p>
    )
  }

  const sourceUrl = previewUrl(jobId, 'source', frame)
  const outputUrl = previewUrl(jobId, 'output', frame)

  function openViewer(): void {
    applyView({ scale: MIN_SCALE, pan: { x: 0, y: 0 } })
    setOpenJobId(jobId)
  }

  function resetView(): void {
    applyView({ scale: MIN_SCALE, pan: { x: 0, y: 0 } })
  }

  return (
    <div className="flex flex-col gap-2">
      <ComparePair
        frame={frame}
        sourceUrl={sourceUrl}
        outputUrl={outputUrl}
        sourceReady={sourceReady}
        onSourceError={setFailedFrame}
        className={paneClassName}
      />
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-muted-foreground">第 {frame} 帧</span>
        <Button type="button" variant="outline" size="sm" onClick={openViewer}>
          <ZoomInIcon />
          放大查看
        </Button>
      </div>
      <DialogPrimitive.Root open={open} onOpenChange={(next) => setOpenJobId(next ? jobId : null)}>
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/75" />
          <DialogPrimitive.Content className="fixed top-1/2 left-1/2 z-[60] flex h-[min(92vh,880px)] w-[min(96vw,1200px)] -translate-x-1/2 -translate-y-1/2 flex-col gap-3 rounded-xl border bg-background p-4 shadow-lg outline-none">
            <div className="flex items-center justify-between gap-3">
              <DialogPrimitive.Title className="text-sm font-medium">
                第 {frame} 帧
              </DialogPrimitive.Title>
              <div className="flex gap-2">
                <Button type="button" variant="outline" size="sm" onClick={resetView}>
                  适合窗口
                </Button>
                <DialogPrimitive.Close asChild>
                  <Button type="button" variant="outline" size="sm">
                    关闭
                  </Button>
                </DialogPrimitive.Close>
              </div>
            </div>
            <DialogPrimitive.Description className="text-xs text-muted-foreground">
              {sourceReady
                ? '滚轮放大，拖拽移动。两边看的是画面上的同一区域。'
                : '滚轮放大，拖拽移动。'}
            </DialogPrimitive.Description>
            <div ref={areaRef} className="flex min-h-0 flex-1 flex-col">
              <ComparePair
                frame={frame}
                sourceUrl={sourceUrl}
                outputUrl={outputUrl}
                sourceReady={sourceReady}
                onSourceError={setFailedFrame}
                view={shown}
                className="h-full"
                onPan={(pan) => applyView({ scale: view.current.scale, pan })}
              />
            </div>
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>
    </div>
  )
}

function ComparePair({
  frame,
  sourceUrl,
  outputUrl,
  sourceReady,
  onSourceError,
  view,
  className,
  onPan
}: {
  frame: number
  sourceUrl: string
  outputUrl: string
  sourceReady: boolean
  onSourceError: (frame: number) => void
  view?: View
  className: string
  onPan?: (pan: { x: number; y: number }) => void
}): React.JSX.Element {
  return (
    <div
      className={`grid min-h-0 gap-2 ${sourceReady ? 'grid-cols-2' : 'grid-cols-1'} ${view ? 'h-full min-h-0 flex-1' : ''}`}
    >
      {sourceReady ? (
        <ZoomPane label="原片" className={className} view={view} onPan={onPan}>
          <img
            key={`source-${frame}`}
            src={sourceUrl}
            alt="原片"
            draggable={false}
            className="h-full w-full object-contain"
            onError={() => onSourceError(frame)}
          />
        </ZoomPane>
      ) : null}
      <ZoomPane label="输出" className={className} view={view} onPan={onPan}>
        <img
          key={`output-${frame}`}
          src={outputUrl}
          alt="输出"
          draggable={false}
          className="h-full w-full object-contain"
        />
      </ZoomPane>
    </div>
  )
}

function ZoomPane({
  label,
  className,
  view,
  onPan,
  children
}: {
  label: string
  className: string
  view?: View
  onPan?: (pan: { x: number; y: number }) => void
  children: React.ReactNode
}): React.JSX.Element {
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null)
  const scale = view?.scale ?? 1
  const pan = view?.pan ?? { x: 0, y: 0 }

  return (
    <figure className={`flex min-h-0 min-w-0 flex-col gap-1 ${className}`}>
      <figcaption className="text-xs text-muted-foreground">{label}</figcaption>
      <div
        data-zoom-pane={onPan ? '' : undefined}
        className="flex min-h-0 flex-1 touch-none items-center justify-center overflow-hidden rounded-lg border bg-black select-none"
        style={{ cursor: onPan ? (scale > 1 ? 'grab' : 'zoom-in') : undefined }}
        onPointerDown={
          onPan
            ? (event) => {
                event.currentTarget.setPointerCapture(event.pointerId)
                drag.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y }
              }
            : undefined
        }
        onPointerMove={
          onPan
            ? (event) => {
                const start = drag.current
                if (!start) return
                onPan({
                  x: start.panX + event.clientX - start.x,
                  y: start.panY + event.clientY - start.y
                })
              }
            : undefined
        }
        onPointerUp={() => {
          drag.current = null
        }}
      >
        <div
          className="pointer-events-none flex h-full w-full items-center justify-center"
          style={
            view ? { transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})` } : undefined
          }
        >
          {children}
        </div>
      </div>
    </figure>
  )
}

function previewUrl(jobId: string, kind: 'source' | 'output', frame: number): string {
  return `${API_BASE}/jobs/${jobId}/preview/${kind}?frame=${frame}`
}

function zoomAt(current: View, cursor: { x: number; y: number; deltaY: number }): View {
  const scale = Math.min(
    MAX_SCALE,
    Math.max(MIN_SCALE, current.scale * Math.exp(-cursor.deltaY * 0.0015))
  )
  if (scale === MIN_SCALE) return { scale, pan: { x: 0, y: 0 } }
  const ratio = scale / current.scale
  return {
    scale,
    pan: {
      x: cursor.x - (cursor.x - current.pan.x) * ratio,
      y: cursor.y - (cursor.y - current.pan.y) * ratio
    }
  }
}
