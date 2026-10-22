import { CircleHelpIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { ENGINE_HELP, GPU_HELP, fieldTitle } from '@/lib/task-schema'
import type { EngineManifest, GpuReport, SchemaProperty } from '@/lib/types'
import { cn } from 'cn'

export function FieldLabel({
  label,
  htmlFor,
  help
}: {
  label: string
  htmlFor?: string
  help?: string
}): React.JSX.Element {
  return (
    <div className="flex items-center gap-1">
      <Label htmlFor={htmlFor}>{label}</Label>
      {help ? <HelpTip label={label} text={help} /> : null}
    </div>
  )
}

function HelpTip({ label, text }: { label: string; text: string }): React.JSX.Element {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex size-4 shrink-0 items-center justify-center rounded-full text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none"
          aria-label={`${label}说明`}
        >
          <CircleHelpIcon className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top">{text}</TooltipContent>
    </Tooltip>
  )
}

export function PathField({
  id,
  label,
  help,
  value,
  placeholder,
  hint,
  onChange,
  onBrowse
}: {
  id: string
  label: string
  help: string
  value: string
  placeholder: string
  hint?: string
  onChange: (value: string) => void
  onBrowse: () => void
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <FieldLabel htmlFor={id} label={label} help={help} />
      <div className="flex gap-2">
        <Input
          id={id}
          value={value}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
        <Button type="button" variant="outline" onClick={onBrowse}>
          浏览
        </Button>
      </div>
      {hint ? <p className="text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  )
}

export function EngineSelect({
  engine,
  available,
  connected,
  onChange
}: {
  engine: EngineManifest | undefined
  available: EngineManifest[]
  connected: boolean
  onChange: (value: string) => void
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <FieldLabel label="引擎" help={ENGINE_HELP} />
      {engine ? (
        <Select value={engine.id} onValueChange={onChange} disabled={!connected}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="选择引擎" />
          </SelectTrigger>
          <SelectContent position="popper">
            {available.map((item) => (
              <SelectItem key={item.id} value={item.id}>
                {item.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <Input disabled placeholder={connected ? '没有可用引擎' : '等待服务'} />
      )}
    </div>
  )
}

export function GpuPicker({
  gpus,
  devices,
  layout = 'list',
  help = GPU_HELP,
  onToggle
}: {
  gpus: GpuReport[]
  devices: number[]
  layout?: 'list' | 'cards'
  help?: string
  onToggle: (index: number, checked: boolean) => void
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <FieldLabel label="显卡" help={help} />
      {gpus.length > 0 ? (
        <div className={layout === 'cards' ? 'flex flex-wrap gap-2' : 'flex flex-col gap-2'}>
          {gpus.map((gpu) => {
            const checked = devices.includes(gpu.index)
            return (
              <label
                key={gpu.index}
                className={cn(
                  'flex items-center gap-2 text-sm',
                  layout === 'cards' &&
                    'cursor-pointer rounded-lg border px-3 py-2 transition-colors hover:bg-muted/50',
                  layout === 'cards' && checked && 'border-primary bg-primary/5'
                )}
              >
                <Checkbox
                  checked={checked}
                  onCheckedChange={(value) => onToggle(gpu.index, value === true)}
                />
                <span>
                  [{gpu.index}] {gpu.name}
                </span>
              </label>
            )
          })}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">连接服务后可以选择显卡。</p>
      )}
    </div>
  )
}

export function SchemaField({
  name,
  property,
  value,
  suggested,
  idPrefix = '',
  onChange
}: {
  name: string
  property: SchemaProperty
  value: string | boolean | undefined
  suggested?: number
  idPrefix?: string
  onChange: (value: string | boolean) => void
}): React.JSX.Element {
  const title = fieldTitle(name, property)
  const fieldId = `${idPrefix}param-${name}`
  if (property.type === 'boolean') {
    const checked = typeof value === 'boolean' ? value : Boolean(property.default)
    return (
      <div className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2 col-span-full">
        <FieldLabel htmlFor={fieldId} label={title} help={property.description} />
        <Switch
          id={fieldId}
          checked={checked}
          onCheckedChange={(checkedValue) => onChange(checkedValue === true)}
        />
      </div>
    )
  }
  if (property.enum && property.enum.length > 0) {
    const current =
      typeof value === 'string' && property.enum.includes(value)
        ? value
        : typeof property.default === 'string' && property.enum.includes(property.default)
          ? property.default
          : property.enum[0]
    return (
      <div className="flex flex-col gap-2">
        <FieldLabel label={title} help={property.description} />
        <Select value={current} onValueChange={onChange}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper">
            {property.enum.map((option) => (
              <SelectItem key={option} value={option}>
                {property.enum_labels?.[option] ?? option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    )
  }
  const text =
    typeof value === 'string'
      ? value
      : suggested != null
        ? String(suggested)
        : property.default != null && property.default !== ''
          ? String(property.default)
          : ''
  const placeholder =
    text === '' ? (name === 'batch_size' ? '留空则按显存自动选择' : '留空使用默认') : undefined
  return (
    <div className="flex flex-col gap-2">
      <FieldLabel htmlFor={fieldId} label={title} help={property.description} />
      <Input
        id={fieldId}
        type={property.type === 'string' ? 'text' : 'number'}
        min={property.minimum}
        max={property.maximum}
        step={property.type === 'number' ? 'any' : 1}
        value={text}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  )
}
