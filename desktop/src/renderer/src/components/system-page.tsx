import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { formatMemory, shortCommit } from '@/lib/format'
import type { EngineStatus, SystemReport } from '@/lib/types'

interface SystemPageProps {
  connected: boolean
  system: SystemReport | null
  onRefresh: () => void
}

function WeightBadges({
  present,
  missing
}: {
  present: string[]
  missing: string[]
}): React.JSX.Element {
  return (
    <div className="flex flex-wrap gap-2 pt-1">
      {present.map((name) => (
        <Badge key={name} variant="secondary">
          {name}
        </Badge>
      ))}
      {missing.map((name) => (
        <Badge key={name} variant="destructive">
          缺少 {name}
        </Badge>
      ))}
    </div>
  )
}

function EngineCard({ engine }: { engine: EngineStatus }): React.JSX.Element {
  const checkedOut =
    engine.builtin || engine.id === 'realbasicvsr'
      ? '内置'
      : engine.present
        ? `已检出 ${shortCommit(engine.commit)}`
        : '未检出'
  return (
    <Card>
      <CardHeader>
        <CardTitle>{engine.name}</CardTitle>
        <CardDescription>{engine.model_dir}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <Row label="检出" value={checkedOut} />
        <Row label="虚拟环境" value={engine.venv_ready ? '已就绪' : '未安装'} />
        {engine.commit_expected ? (
          <Row label="期望提交" value={shortCommit(engine.commit_expected)} />
        ) : null}
        <Row label="说明" value={engine.detail || '—'} />
        <WeightBadges present={engine.models_present} missing={engine.models_missing} />
      </CardContent>
    </Card>
  )
}

function Row({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <div className="flex items-start justify-between gap-6 border-b border-border/70 py-2.5 last:border-0">
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className="text-right break-all">{value}</span>
    </div>
  )
}

export function SystemPage({ connected, system, onRefresh }: SystemPageProps): React.JSX.Element {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4">
      {!connected ? (
        <Alert>
          <AlertTitle>服务未连接</AlertTitle>
          <AlertDescription>启动侧栏中的本地服务，或在终端运行 videox serve。</AlertDescription>
        </Alert>
      ) : null}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium">运行环境</h2>
          {system ? (
            <Badge variant={system.ready ? 'default' : 'destructive'}>
              {system.ready ? '可以运行' : '尚未就绪'}
            </Badge>
          ) : null}
        </div>
        <Button variant="outline" size="sm" onClick={onRefresh}>
          刷新
        </Button>
      </div>
      {!system ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            还没有读到系统信息。
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>运行时</CardTitle>
              <CardDescription>{system.project_root}</CardDescription>
            </CardHeader>
            <CardContent>
              <Row label="Python" value={system.python} />
              <Row label="PyTorch" value={system.torch ?? '未安装'} />
              <Row label="CUDA" value={system.cuda_available ? '可用' : '不可用'} />
              <Row label="ffmpeg" value={system.ffmpeg ?? '未找到'} />
              <Row label="ffprobe" value={system.ffprobe ?? '未找到'} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>显卡</CardTitle>
              <CardDescription>
                {system.gpus.length > 0 ? `${system.gpus.length} 张` : '未检测到 NVIDIA 显卡'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {system.gpus.length === 0 ? (
                <p className="text-muted-foreground">没有可用的 CUDA 设备。</p>
              ) : (
                system.gpus.map((gpu) => (
                  <Row
                    key={gpu.index}
                    label={`GPU ${gpu.index}`}
                    value={`${gpu.name} · ${formatMemory(gpu.memory_mib)}`}
                  />
                ))
              )}
            </CardContent>
          </Card>
          {system.engines && system.engines.length > 0 ? (
            system.engines.map((engine) => <EngineCard key={engine.id} engine={engine} />)
          ) : (
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>SeedVR2</CardTitle>
                <CardDescription>{system.model_dir}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Row
                  label="检出"
                  value={
                    system.seedvr2_present
                      ? `已检出 ${shortCommit(system.seedvr2_commit)}`
                      : '未检出'
                  }
                />
                <Row label="期望提交" value={shortCommit(system.seedvr2_commit_expected)} />
                <WeightBadges present={system.models_present} missing={system.models_missing} />
              </CardContent>
            </Card>
          )}
          {system.warnings.length > 0 ? (
            <Alert variant="destructive" className="lg:col-span-2">
              <AlertTitle>注意</AlertTitle>
              <AlertDescription>
                <ul className="list-disc space-y-1 pl-4">
                  {system.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          ) : null}
        </div>
      )}
    </div>
  )
}
