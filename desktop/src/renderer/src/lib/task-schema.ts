import { isSimpleProperty, paramLabel } from '@/lib/format'
import type { EngineManifest, MediaKind, SchemaProperty } from '@/lib/types'

export interface FieldGroup {
  id: string
  title: string
  fields: Array<[string, SchemaProperty]>
}

export const ENGINE_HELP =
  '实际执行修复的模型。视频可以在 SeedVR2、FlashVSR、SwiftVR 和 RealBasicVSR 之间切换，切换后配置项和默认值会换成该模型自己的。图片仍使用 SeedVR2。'
export const GPU_HELP =
  '勾选参与计算的显卡。都不选时，有两张及以上显卡会用前两张，只有一张就用那一张。双卡按帧分段，每张卡各加载一份模型。'

export function gpuHelp(policy: 'multi' | 'single' | undefined): string {
  if (policy === 'single') {
    return '这个模型只用一张显卡。选中多张时只保留第一张。都不选时用第一张可用的卡。'
  }
  return GPU_HELP
}

export function fieldTitle(name: string, property: SchemaProperty): string {
  return paramLabel(name, property.title)
}

function appliesTo(property: SchemaProperty, media: MediaKind): boolean {
  return !property.media || property.media.length === 0 || property.media.includes(media)
}

export function visibleFields(
  engine: EngineManifest | undefined,
  media: MediaKind
): Array<[string, SchemaProperty]> {
  return Object.entries(engine?.params_schema.properties ?? {}).filter(
    ([, property]) => isSimpleProperty(property) && appliesTo(property, media)
  )
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
      if (numeric !== property.default) params[key] = numeric
      continue
    }
    if (value !== property.default) params[key] = value
  }
  return { params }
}
