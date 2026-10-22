import { isSimpleProperty, paramLabel } from '@/lib/format'
import type {
  EngineManifest,
  MediaKind,
  SchemaCondition,
  SchemaProperty,
  SchemaScalar
} from '@/lib/types'

export interface FieldGroup {
  id: string
  title: string
  fields: Array<[string, SchemaProperty]>
}

export const ENGINE_HELP =
  '实际执行修复的模型。视频可以在 SeedVR2、FlashVSR、SwiftVR 和 RealBasicVSR 之间切换，切换后配置项和默认值会换成该模型自己的。'
export const IMAGE_ENGINE_HELP =
  '实际处理图片的模型。SeedVR2 是扩散修复，细节最多也最慢；Real-ESRGAN 快且省显存，适合照片和动漫；SwinIR 能超分、去噪和去 JPEG 压缩块；SCUNet 专门去噪；GFPGAN 修复人脸。切换引擎或引擎里的模型后，下面的配置项和默认值会跟着换。'
export const GPU_HELP =
  '勾选参与计算的显卡。都不选时，有两张及以上显卡会用前两张，只有一张就用那一张。多卡时把视频按时间切成几段，每张卡处理一段并各加载一份模型。'

export function gpuHelp(policy: 'multi' | 'single' | undefined, media: MediaKind): string {
  if (media === 'image') {
    return '图片只用一张显卡。选中多张时只用第一张，都不选时用第一张可用的卡。'
  }
  if (policy === 'single') {
    return '这个模型只用一张显卡。选中多张时只保留第一张。都不选时用第一张可用的卡。'
  }
  return GPU_HELP
}

export function fieldTitle(name: string, property: SchemaProperty): string {
  return paramLabel(name, property.title)
}

export function engineHelp(media: MediaKind): string {
  return media === 'image' ? IMAGE_ENGINE_HELP : ENGINE_HELP
}

function appliesTo(property: SchemaProperty, media: MediaKind): boolean {
  return !property.media || property.media.length === 0 || property.media.includes(media)
}

function conditionHolds(
  condition: SchemaCondition,
  current: Record<string, SchemaScalar | undefined>
): boolean {
  return Object.entries(condition).every(([key, allowed]) => {
    const value = current[key]
    return value !== undefined && allowed.includes(value)
  })
}

/** Applies visible_when and the first matching override. Null means the field is hidden. */
function resolveProperty(
  property: SchemaProperty,
  current: Record<string, SchemaScalar | undefined>
): SchemaProperty | null {
  if (property.visible_when && !conditionHolds(property.visible_when, current)) return null
  const override = property.overrides?.find((item) => conditionHolds(item.when, current))
  if (!override) return property
  const merged: SchemaProperty & { when?: SchemaCondition } = { ...property, ...override }
  delete merged.when
  return merged
}

function typedValue(
  property: SchemaProperty,
  value: string | boolean | undefined
): SchemaScalar | undefined {
  if (value === undefined || value === '') return property.default
  if (property.type === 'boolean') return Boolean(value)
  if (property.type === 'integer' || property.type === 'number') {
    const numeric = Number(value)
    return Number.isFinite(numeric) ? numeric : property.default
  }
  return value
}

/**
 * Fields for this engine and media, resolved against the current form values
 * in schema order, the same way the server resolves them.
 */
export function visibleFields(
  engine: EngineManifest | undefined,
  media: MediaKind,
  values: Record<string, string | boolean> = {}
): Array<[string, SchemaProperty]> {
  const current: Record<string, SchemaScalar | undefined> = {}
  const fields: Array<[string, SchemaProperty]> = []
  for (const [key, property] of Object.entries(engine?.params_schema.properties ?? {})) {
    if (!isSimpleProperty(property) || !appliesTo(property, media)) continue
    const resolved = resolveProperty(property, current)
    if (!resolved) continue
    current[key] = typedValue(resolved, values[key])
    fields.push([key, resolved])
  }
  return fields
}

/**
 * Fields whose defaults or limits follow `key`, directly or through another
 * such field, in schema order. Fields whose visibility alone follows `key`
 * are not included: hidden fields are never sent, so their values can stay.
 */
export function dependentFields(engine: EngineManifest | undefined, key: string): string[] {
  const properties = Object.entries(engine?.params_schema.properties ?? {})
  const found = new Set<string>()
  const pending = [key]
  while (pending.length > 0) {
    const source = pending.pop()!
    for (const [name, property] of properties) {
      if (found.has(name)) continue
      if ((property.overrides ?? []).some((item) => source in item.when)) {
        found.add(name)
        pending.push(name)
      }
    }
  }
  return properties.map(([name]) => name).filter((name) => found.has(name))
}

function sameAsDefault(property: SchemaProperty, value: string | boolean): boolean {
  if (value === '') return true
  if (property.type === 'integer' || property.type === 'number') {
    return Number(value) === property.default
  }
  return value === property.default
}

function fitsProperty(property: SchemaProperty, value: string | boolean): boolean {
  if (property.type === 'boolean') return typeof value === 'boolean'
  if (property.enum) return property.enum.includes(String(value))
  if (property.type === 'integer' || property.type === 'number') {
    const numeric = Number(value)
    // Leave text that is not a number to collectParams, which names the field.
    if (!Number.isFinite(numeric)) return true
    if (property.minimum !== undefined && numeric < property.minimum) return false
    if (property.maximum !== undefined && numeric > property.maximum) return false
  }
  return true
}

/**
 * Form values after `key` is set. A dependent field's stored value is dropped,
 * so the field shows the new default, when the user had left it at the old
 * default or when it no longer fits the new choices or limits. Any other
 * value the user picked stays.
 */
export function valuesAfterChange(
  engine: EngineManifest | undefined,
  media: MediaKind,
  current: Record<string, string | boolean>,
  key: string,
  value: string | boolean
): Record<string, string | boolean> {
  const before = new Map(visibleFields(engine, media, current))
  const next = { ...current, [key]: value }
  for (const name of dependentFields(engine, key)) {
    if (!(name in next)) continue
    const old = before.get(name)
    const now = new Map(visibleFields(engine, media, next)).get(name)
    if ((old && sameAsDefault(old, next[name])) || (now && !fitsProperty(now, next[name]))) {
      delete next[name]
    }
  }
  return next
}

export function schemaGroups(
  engine: EngineManifest | undefined,
  fields: Array<[string, SchemaProperty]>,
  omit: readonly string[] = []
): FieldGroup[] {
  const declared = engine?.params_schema.groups ?? []
  const titles = new Map(declared.map((group) => [group.id, group.title]))
  const hidden = new Set(omit)
  const ids = declared.map((group) => group.id).filter((id) => !hidden.has(id))
  for (const [, property] of fields) {
    const id = property.group ?? 'primary'
    if (!hidden.has(id) && !ids.includes(id)) ids.push(id)
  }
  return ids
    .map((id) => ({
      id,
      title: titles.get(id) ?? id,
      fields: fields.filter(([, property]) => (property.group ?? 'primary') === id)
    }))
    .filter((group) => group.fields.length > 0)
}

export function selectedOutputFormat(
  media: MediaKind,
  values: Record<string, string | boolean>
): 'mp4' | 'png' {
  if (media === 'image') return 'png'
  return values.output_format === 'png' ? 'png' : 'mp4'
}

export function outputHelp(
  media: MediaKind,
  outputFormat: 'mp4' | 'png',
  engineId = 'seedvr2'
): string {
  const tag = engineId || 'seedvr2'
  if (media === 'image') {
    return `结果保存位置。默认放到项目 output 目录，文件名加上 .${tag}，保存为 PNG，透明通道会保留。`
  }
  if (outputFormat === 'png') {
    return 'PNG 序列的输出目录。每一帧写成一张 PNG，不再封装音轨，体积更大，方便后期。'
  }
  return `结果保存位置。默认放到项目 output 目录，文件名加上 .${tag}。MP4 会尽量保留原片音轨。`
}

function stepError(title: string, step: number, base: number): string {
  const offset = ((base % step) + step) % step
  if (offset === 0) return `${title} 需要是 ${step} 的倍数`
  const examples = [1, 2, 3].map((k) => base + k * step).join('、')
  return `${title} 需要是 ${step} 的倍数加 ${offset}，例如 ${examples}`
}

export function collectParams(
  fields: Array<[string, SchemaProperty]>,
  values: Record<string, string | boolean>
): { params: Record<string, unknown> } | { error: string } {
  const params: Record<string, unknown> = {}
  for (const [key, property] of fields) {
    if (!(key in values)) continue
    const value = values[key]
    const title = fieldTitle(key, property)
    if (property.type === 'boolean') {
      const checked = Boolean(value)
      if (checked !== Boolean(property.default)) params[key] = checked
      continue
    }
    if (typeof value !== 'string' || value.trim() === '') continue
    if (property.enum) {
      if (value !== property.default) params[key] = value
      continue
    }
    if (property.type === 'integer' || property.type === 'number') {
      const numeric = Number(value)
      if (!Number.isFinite(numeric)) return { error: `${title} 不是有效数字` }
      if (property.type === 'integer' && !Number.isInteger(numeric)) {
        return { error: `${title} 需要是整数` }
      }
      if (property.minimum != null && numeric < property.minimum) {
        return { error: `${title} 不能小于 ${property.minimum}` }
      }
      if (property.maximum != null && numeric > property.maximum) {
        return { error: `${title} 不能大于 ${property.maximum}` }
      }
      if (property.step != null && property.step > 1) {
        const base = property.minimum ?? 0
        if ((numeric - base) % property.step !== 0) {
          return { error: stepError(title, property.step, base) }
        }
      }
      if (numeric !== property.default) params[key] = numeric
      continue
    }
    if (value !== property.default) params[key] = value
  }
  return { params }
}
