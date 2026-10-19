# VideoX

本地视频修复核心。当前引擎是 SeedVR2，命令行和本机 API 共用同一套任务。以后的 Electron 界面只需要调用这个 API，新的修复模型只需要再注册一个引擎。

默认按每张卡 12–16GB 配置：3B FP8、VAE 分块、时间重叠 3 帧。约 12GB 的卡每批 5 帧，约 16GB 及以上每批 9 帧。检测到两张卡时使用 `0,1`。

## 准备

需要 Windows、Python 3.12 或 3.13、NVIDIA 驱动。ffmpeg 不在 PATH 里时，安装脚本会下载一份便携版。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\videox doctor
```

## 修复

```powershell
.\.venv\Scripts\videox restore input.mp4 -o output\restored.mp4
```

短边分辨率可以用 `--resolution 720` 覆盖。模型在首次修复时下载到 `models\SEEDVR2`。成片会把原音轨拷回去。

## 本地 API

```powershell
.\.venv\Scripts\videox serve
```

默认监听 `127.0.0.1:8765`。

- `GET /engines`：引擎和参数 schema
- `GET /system`：Python、显卡、ffmpeg、SeedVR2 和模型是否就绪
- `POST /jobs`：创建任务
- `GET /jobs/{id}`：任务状态
- `GET /jobs/{id}/events`：SSE 进度
- `POST /jobs/{id}/cancel`：取消任务

OpenAPI 在 `/openapi.json`。
