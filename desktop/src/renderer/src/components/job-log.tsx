import { useEffect, useRef } from 'react'

import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from 'cn'

export function JobLog({
  lines,
  className
}: {
  lines: string[]
  className?: string
}): React.JSX.Element {
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const viewport = logRef.current?.parentElement
    if (viewport) viewport.scrollTop = viewport.scrollHeight
  }, [lines])

  return (
    <ScrollArea className={cn('rounded-lg border', className)}>
      <div ref={logRef} className="p-3 font-mono text-xs leading-5">
        {lines.length === 0 ? (
          <p className="text-muted-foreground">等待输出</p>
        ) : (
          <pre className="break-all whitespace-pre-wrap">{lines.join('\n')}</pre>
        )}
      </div>
    </ScrollArea>
  )
}
