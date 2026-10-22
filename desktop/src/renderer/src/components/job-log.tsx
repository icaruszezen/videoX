import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ArrowDownIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from 'cn'

// How close to the bottom (px) still counts as following the output.
const FOLLOW_SLACK = 24

interface LogView {
  lines: string[]
  offset: number
}

export function JobLog({
  lines,
  offset = 0,
  finished = false,
  className
}: {
  lines: string[]
  offset?: number
  finished?: boolean
  className?: string
}): React.JSX.Element {
  const contentRef = useRef<HTMLDivElement>(null)
  const latest = useRef<LogView>({ lines, offset })
  const following = useRef(true)
  // While the reader is scrolled up, keep showing the lines they were reading so
  // new output and the line limit do not move the text under them.
  const [frozen, setFrozen] = useState<LogView | null>(null)
  const shown = frozen ?? { lines, offset }

  useLayoutEffect(() => {
    latest.current = { lines, offset }
  }, [lines, offset])

  useLayoutEffect(() => {
    if (frozen) return
    const viewport = viewportOf(contentRef.current)
    if (viewport) viewport.scrollTop = viewport.scrollHeight
  }, [lines, frozen])

  useEffect(() => {
    const viewport = viewportOf(contentRef.current)
    if (!viewport) return
    let lastTop = viewport.scrollTop
    const onScroll = (): void => {
      const top = viewport.scrollTop
      const gap = viewport.scrollHeight - top - viewport.clientHeight
      const movedUp = top < lastTop
      lastTop = top
      // Snapping to the end never lowers scrollTop, and a shrinking log clamps it
      // to the end, so an upward move with room below is the reader scrolling.
      // Freeze right away instead of waiting for a bigger gap, or the next
      // update would snap back and undo small trackpad or smooth scrolls.
      if (movedUp && gap > 1) setFrozen((current) => current ?? latest.current)
      else if (gap <= FOLLOW_SLACK) setFrozen(null)
    }
    // The page can be hidden (display:none) while lines arrive, and the window
    // can be resized. Either way, a log that follows the output stays at the end.
    const observer = new ResizeObserver(() => {
      const gap = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
      if (gap > 0 && following.current) viewport.scrollTop = viewport.scrollHeight
    })
    viewport.addEventListener('scroll', onScroll, { passive: true })
    observer.observe(viewport)
    if (contentRef.current) observer.observe(contentRef.current)
    return () => {
      viewport.removeEventListener('scroll', onScroll)
      observer.disconnect()
    }
  }, [])

  useLayoutEffect(() => {
    following.current = frozen === null
  }, [frozen])

  return (
    <div className={cn('relative min-h-0', className)}>
      <ScrollArea className="h-full rounded-lg border bg-muted/30">
        <div ref={contentRef} className="p-3 font-mono text-xs leading-5">
          {shown.offset > 0 ? (
            <p className="mb-1 text-muted-foreground">
              较早的 {shown.offset} 行已省略，只保留最近 {shown.lines.length} 行
            </p>
          ) : null}
          {shown.lines.length === 0 ? (
            <p className="text-muted-foreground">{finished ? '没有日志输出' : '等待输出'}</p>
          ) : (
            <div className="whitespace-pre-wrap wrap-anywhere">
              {shown.lines.map((line, index) => (
                <div key={shown.offset + index}>{line}</div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
      {frozen ? (
        <Button
          type="button"
          size="sm"
          variant="secondary"
          className="absolute right-4 bottom-3 shadow-md"
          onClick={() => setFrozen(null)}
        >
          <ArrowDownIcon data-icon="inline-start" />
          {finished ? '回到底部' : '跟随最新输出'}
        </Button>
      ) : null}
    </div>
  )
}

function viewportOf(node: HTMLElement | null): HTMLElement | null {
  return node?.closest<HTMLElement>('[data-slot=scroll-area-viewport]') ?? null
}
