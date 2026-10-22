# VideoX

本地运行的视频和图片超分修复工具。项目由三部分组成：

- **修复核心**（Python 包 `videox`）：把不同的修复模型包装成统一的“引擎”，负责环境准备、任务执行、进度解析和音轨回封。
- **命令行和本机 API**：`videox` 命令和 FastAPI 服务共用同一套任务执行逻辑。
- **桌面界面**（`desktop/`，Electron）：通过本机 API 提交任务、查看进度，并把处理中的一帧和原片并排对比。

目前内置八个引擎：SeedVR2、FlashVSR、SwiftVR、RealBasicVSR 处理视频（SeedVR2 也能处理图片），Real-ESRGAN、SwinIR、SCUNet、GFPGAN 只处理图片。所有处理都在本机 NVIDIA 显卡上完成。

## 引擎

| 引擎 | 方法 | 支持 | 显卡 | 放大方式 | 运行环境 |
| --- | --- | --- | --- | --- | --- |
| SeedVR2（默认） | 一步扩散修复 | 视频、图片 | 可多卡 | 按短边分辨率输出，默认 1080 | 主虚拟环境 |
| FlashVSR v1.1 | 一步扩散超分 | 视频 | 单卡 | 4 倍（可选 2 倍） | 独立虚拟环境 |
| SwiftVR | 一步生成式修复，面向实时超分 | 视频 | 单卡 | 1–4 倍，默认 4 倍 | 独立虚拟环境 |
| RealBasicVSR | 非扩散的循环超分 | 视频 | 单卡 | 固定 4 倍 | 主虚拟环境 |
| Real-ESRGAN | GAN 超分 | 图片 | 单卡 | 2 倍或 4 倍模型，成品 1–8 倍，默认跟随模型 | 主虚拟环境 |
| SwinIR | Transformer 修复 | 图片 | 单卡 | 超分模型 2 倍或 4 倍（成品 1–8 倍）；去噪和去 JPEG 压缩模型不放大 | 主虚拟环境 |
| SCUNet | 盲去噪 | 图片 | 单卡 | 不放大 | 主虚拟环境 |
| GFPGAN | 人脸修复，背景可交给 Real-ESRGAN | 图片 | 单卡 | 1、2、4 倍，默认 2 倍 | 主虚拟环境 |

- **SeedVR2**：默认使用 3B FP8 模型、VAE 分块、时间重叠 3 帧，按每张卡 12–16GB 显存配置。视频批大小按所选显卡中显存最小的一张自动选择：约 12GB 每批 5 帧，约 16GB 及以上每批 9 帧（批大小必须是 4n+1）。检测到两张及以上显卡时默认使用前两张，按帧分段，每张卡各加载一份模型。还可以切换 3B/7B、FP16/GGUF、Sharp 等模型，以及色彩校正、显存卸载、`torch.compile`、PNG 序列输出、10-bit MP4 等选项。
- **FlashVSR**：默认 Tiny Long 模式，比 SeedVR2 更快、更省显存。依赖 Block-Sparse-Attention 稀疏注意力内核，安装时要在本机编译，没装好时 FlashVSR 不能使用（其他引擎不受影响）。模型要求宽高是 128 的倍数、帧数是 8n+1，封装会先补边、补帧，成片再裁回原片放大后的尺寸和原来的帧数，帧率与原片一致。
- **SwiftVR**：权重是研究许可，个人本地使用可以，商用需要作者书面授权。
- **RealBasicVSR**：网络结构直接实现在本仓库里，不需要额外检出上游代码，速度通常更快，细节和人脸弱于扩散模型。
- **图片引擎**（Real-ESRGAN、SwinIR、SCUNet、GFPGAN）：网络结构也直接实现在本仓库里，用主环境运行。每个引擎都有几个模型可选，例如 Real-ESRGAN 的照片、动漫和轻量通用模型，SwinIR 的超分、去噪和去 JPEG 压缩模型，GFPGAN 的 v1.4/v1.3。表单按所选模型切换参数和默认值：只有用得上的参数才显示（例如去噪强度只对轻量通用模型出现，GFPGAN 只修人脸时不显示背景分块设置），倍数等默认值跟着模型变。大图分块处理，显存不够时把分块大小调小。GFPGAN 的背景放大复用 Real-ESRGAN 的模型和权重。

单卡引擎在选了多张卡时只用第一张。各引擎的完整参数、默认值和说明由 `GET /engines` 返回，桌面界面也按这份 schema 生成表单。使用前请看一下文末的[注意事项](#注意事项)。

## 环境要求

- Windows（安装脚本、ffmpeg 下载和桌面端启动服务都按 Windows 路径编写）
- Python 3.12 或 3.13（安装脚本只检查不低于 3.12）。`setup.ps1` 用 PATH 里的 `python` 命令建环境，运行前先确认 `python --version` 指向的是正确版本，而不是 Microsoft Store 的占位程序
- NVIDIA 显卡和驱动（`nvidia-smi` 可用）
- Git（用来检出各引擎的上游代码）
- 足够的磁盘空间：主环境、FlashVSR 和 SwiftVR 各装一份 CUDA 版 PyTorch，另外还有模型权重
- Node.js 22.12 或更新版本和 npm（只有桌面界面需要）

ffmpeg 不在 PATH 里时，安装脚本会下载一份便携版到 `tools\ffmpeg`。

## 安装

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\videox doctor
```

`setup.ps1` 会创建 `.venv`，以可编辑方式安装本项目，然后运行 `videox setup` 完成剩下的准备：

1. 检查 Python 版本和 `nvidia-smi`。
2. 找不到 ffmpeg/ffprobe 时下载便携版。
3. 把 SeedVR2 上游代码检出到 `third_party\seedvr2`，提交号固定在 `third_party\seedvr2.rev`。
4. 在主环境安装 CUDA 版 PyTorch（先试 cu128，再试 cu130）和 SeedVR2 依赖（跳过 flash-attn、SageAttention 等需要单独编译的包）。
5. 检出 FlashVSR 和 SwiftVR（提交号见 `third_party\*.rev`），各自创建 `third_party\<引擎>\.venv` 并安装依赖；FlashVSR 还必须编译 Block-Sparse-Attention，这一步需要本机能编译 CUDA 扩展（与 PyTorch 匹配的 CUDA Toolkit 和 Visual Studio C++ 生成工具）。

FlashVSR 或 SwiftVR 没装好时，`videox setup` 会报出是哪个引擎失败并以非 0 退出，此时 SeedVR2 和 RealBasicVSR 已经可以使用。`videox setup` 可以重复运行，每次都会把三个环境里的 PyTorch 升级到对应索引的最新版本。

`videox doctor` 会列出 Python、PyTorch/CUDA、ffmpeg/ffprobe、显卡、SeedVR2 的检出提交，以及每个引擎是否就绪和缺少的权重；其他引擎没检出、虚拟环境没装好或提交号不符时会以“注意”列出。检测到显卡、装有 CUDA 版 PyTorch、找得到 ffmpeg/ffprobe，并且 SeedVR2 已检出到 `seedvr2.rev` 指定的提交时，显示“可以运行: 是”并以 0 退出，否则以 1 退出。权重缺失不影响“可以运行”，但对应引擎那一行会显示“未就绪”。每个引擎完整的提交号和虚拟环境状态见 `GET /system`。

### 模型权重

权重在首次使用对应引擎时自动下载：

| 引擎 | 目录 | 来源和文件 |
| --- | --- | --- |
| SeedVR2 | `models\SEEDVR2` | 由 SeedVR2 上游代码下载，默认需要 `seedvr2_ema_3b_fp8_e4m3fn.safetensors` 和 `ema_vae_fp16.safetensors`；换用其他扩散模型时再下载对应文件 |
| FlashVSR | `models\FlashVSR` | Hugging Face `JunhaoZhuang/FlashVSR-v1.1`：`diffusion_pytorch_model_streaming_dmd.safetensors`、`LQ_proj_in.ckpt`，Tiny 和 Tiny Long 模式再加 `TCDecoder.ckpt`；`Wan2.1_VAE.pth` 只在第一次用 Full 模式时下载 |
| SwiftVR | `models\SwiftVR` | Hugging Face `H-oliday/SwiftVR`：`reae.safetensors`、`prompt_embedding.safetensors`、`transformer/config.json`、`transformer/diffusion_pytorch_model.safetensors` |
| RealBasicVSR | `models\RealBasicVSR` | OpenMMLab（MMEditing）发布的 `realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth`，本地保存为 `RealBasicVSR_x4.pth` |
| Real-ESRGAN | `models\RealESRGAN` | `xinntao/Real-ESRGAN` 的 GitHub Releases，按所选模型下载，默认是 `RealESRGAN_x4plus.pth` |
| SwinIR | `models\SwinIR` | `JingyunLiang/SwinIR` 的 GitHub Releases（v0.0），按所选模型下载，默认是 `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` |
| SCUNet | `models\SCUNet` | `cszn/KAIR` 的 GitHub Releases（v1.0）：`scunet_color_real_psnr.pth`、`scunet_color_real_gan.pth` |
| GFPGAN | `models\GFPGAN` | `TencentARC/GFPGAN` 的 GitHub Releases：`GFPGANv1.4.pth` 或 `GFPGANv1.3.pth`；`xinntao/facexlib` 的人脸检测权重 `detection_Resnet50_Final.pth`。背景放大用 `models\RealESRGAN` 里的权重 |

FlashVSR、SwiftVR 和 RealBasicVSR 的下载地址写在代码里，直接访问 huggingface.co 和 download.openmmlab.com；图片引擎访问 github.com。图片引擎只下载所选模型需要的文件，换模型时再下载对应文件。网络不通时可以手动下载，按上表的文件名和相对路径放进对应目录，已存在且非空的文件不会重新下载。下载中断或收到的字节数和服务器声明的不一致时，任务失败且不留下半截文件，下次运行会重新下载。

## 命令行

```powershell
# 修复视频，默认使用 SeedVR2
.\.venv\Scripts\videox restore input.mp4 -o output\restored.mp4

# 换引擎
.\.venv\Scripts\videox restore input.mp4 --engine flashvsr

# 图片按扩展名自动识别
.\.venv\Scripts\videox restore photo.jpg

# 图片引擎和它的参数
.\.venv\Scripts\videox restore photo.jpg --engine gfpgan --param upscale=4 --param strength=0.7
```

`restore` 的参数：

| 参数 | 说明 |
| --- | --- |
| `-o, --output` | 输出路径，相对路径按当前目录解析。省略时写到项目根目录下的 `output\<原文件名去掉扩展名>.<引擎>.mp4`，例如 `input.mp4` 用 SeedVR2 处理后是 `output\input.seedvr2.mp4`；图片是 `.png` |
| `--engine` | `seedvr2`（默认）、`flashvsr`、`swiftvr`、`realbasicvsr`；图片还可以用 `realesrgan`、`swinir`、`scunet`、`gfpgan` |
| `--devices` | CUDA 设备号，例如 `0,1`。省略时 SeedVR2 在有两张及以上显卡时用前两张，否则用第一张；其他引擎都是单卡引擎，只用列表里的第一张 |
| `--resolution` | 输出短边分辨率，只对 SeedVR2 生效 |
| `--batch-size` | 批大小（4n+1），只对 SeedVR2 生效 |
| `--param 名称=值` | 任意引擎参数，可以重复。名称和取值范围见 `GET /engines` 返回的 `params_schema`，值按字段类型转换，布尔值写 `true`/`false` |

例如 `--param model=real_x2_m --param tile=256`（SwinIR），或 `--param chunk_size=300`（SeedVR2）。

视频的处理流程是：引擎先在输出位置旁写出一份无声视频（`<输出名>.video-only.mp4`）。原片有音轨时，把原片的第一条音轨拷回（拷贝失败会转成 192k AAC），音轨截到和画面一样长；SeedVR2 设置了 `skip_first_frames` 时，音轨也从对应的时间点开始，音画保持同步。没有音轨时直接封装成片。输出容器由扩展名决定，视频输出的扩展名不是 mp4、m4v、mov、mkv、avi 时，在文件名后补 `.mp4`（例如 `a.ts` 写成 `a.ts.mp4`）。图片识别的扩展名是 png、jpg、jpeg、webp、bmp、tif、tiff，图片输出的扩展名不在其中时改成 `.png`。图片引擎输出 PNG、WebP、TIFF 时保留透明通道，并带上原图的 ICC 色彩配置；WebP 每边最多 16383 像素，JPEG 最多 65500 像素，成品超出时在开始处理前就报错。

其他子命令：

- `videox setup`：准备或修复运行环境（见上文）。
- `videox doctor`：检查本机能否运行。
- `videox serve [--host 127.0.0.1] [--port 8765]`：启动本机 API。

## 本机 API

```powershell
.\.venv\Scripts\videox serve
```

默认监听 `127.0.0.1:8765`，OpenAPI 文档在 `/openapi.json`。服务没有鉴权，并且允许任意来源跨域访问（CORS 为 `*`），本机浏览器里打开的网页也能调用它。不要监听到公网地址，不用时关掉服务。

| 接口 | 说明 |
| --- | --- |
| `GET /engines` | 引擎列表：名称、说明、支持的媒体（`media`）、显卡策略（`device_policy`）和参数 schema |
| `GET /system` | 运行环境报告，和 `videox doctor` 用的是同一份数据 |
| `POST /jobs` | 创建任务，返回 202 和任务记录 |
| `GET /jobs/{id}` | 任务状态、阶段、进度和错误信息 |
| `GET /jobs/{id}/events` | SSE 事件流 |
| `GET /jobs/{id}/preview/{source\|output}` | 原片/修复结果的预览帧（JPEG），还没有预览时返回 404 |
| `POST /jobs/{id}/cancel` | 取消任务，并结束引擎进程 |

创建任务的请求体：

```json
{
  "engine_id": "seedvr2",
  "input_path": "D:\\videos\\input.mp4",
  "output_path": "D:\\videos\\input.seedvr2.mp4",
  "media": "video",
  "devices": [0, 1],
  "params": { "resolution": 1080 }
}
```

- `engine_id` 默认 `seedvr2`。`media` 是 `video` 或 `image`，默认 `video`，需要和输入文件的类型一致。`devices` 省略时自动选择显卡。
- `params` 里只放需要改的参数，键名和取值范围见 `GET /engines` 返回的 `params_schema`，没给的参数用引擎默认值。
- 输入文件不存在、引擎未知、`media` 与文件类型不符或引擎不支持该媒体时，`POST /jobs` 直接返回 400。`params` 在提交时不校验：引擎不认识的键会被忽略，取值不合法时接口照样返回 202，任务随后以 `failed` 结束，原因在 `error` 里。例外是 `output_format`：任务执行器对所有引擎都会读它，设为 `png` 时一律按 PNG 序列目录处理，所以只对 SeedVR2 传这个参数。
- SeedVR2 的 `output_format` 设为 `png` 时，`output_path` 是一个目录，每帧写成一张 PNG，不封装音轨。

任务状态有 `queued`、`running`、`completed`、`failed`、`cancelled`。SSE 事件类型有 `queued`、`stage`、`log`、`progress`、`preview`、`completed`、`failed`、`cancelled`，每条事件的 `data` 是一个 JSON 对象。连接时会先重放这个任务已有的事件（最多最近 2000 条），空闲时每 15 秒发一条保活注释，任务结束后服务端关闭连接。

`progress` 取值 0 到 1，是整个任务的进度，不会在换阶段时回到 0。SeedVR2 把编码、放大、解码、后处理折算成一条进度，分段处理（`chunk_size`）和多张显卡的进度也合在一起；其他引擎按阶段（`stage`，依次是 download、load、restore）分配：下载权重占 0–5%，加载模型占 5–10%，处理占 10–100%，GFPGAN 先修人脸（stage 为 face，10–55%）再处理背景（55–100%）。

任务记录只保存在服务进程的内存里，服务重启后就查不到了。任务按提交顺序一个一个执行，前一个结束后才开始下一个，排队中的任务也可以取消。服务退出时会结束正在运行的引擎进程。预览图写在 `output\.previews\<任务 id>`。

## 桌面界面

```powershell
cd desktop
npm install
npm run dev
```

窗口固定连接 `127.0.0.1:8765`，用 `--host`/`--port` 改过地址的服务连不上。服务没开时，可以点侧栏的“启动服务”，界面会用 `.venv` 里的 `videox serve` 拉起服务。由界面拉起的服务在点“停止服务”或退出窗口时会被强制结束，正在运行的引擎进程也会一起结束，未完成的任务直接中断；长任务建议在终端里单独运行 `videox serve`，这时界面只连接，不会关掉它。

界面有四页：

- **修复**：选择视频、输出位置、引擎和显卡，按所选引擎的参数 schema 生成分组表单。未改动的参数沿用服务端默认值。
- **任务**：本次会话的视频任务列表，显示阶段、进度、日志，可以取消任务，还可以把原片和修复结果的同一帧并排放大对比。
- **图片**：选择图片引擎和其中的模型处理单张图片，表单按所选模型切换参数和默认值，任务进度和对比显示在同一页。
- **系统**：显示 `GET /system` 的内容，包括显卡、PyTorch、ffmpeg、各引擎的检出、虚拟环境和权重状态。

对比用的预览帧由引擎写出：SeedVR2 在后处理阶段更新为最近完成的一帧，RealBasicVSR 每处理完一段更新一次，FlashVSR 和 SwiftVR 只在整段处理完后给出首帧，图片引擎在整张处理完后给出结果。

窗口刷新后，界面会从 `output\.session-jobs.json` 找回本次会话的任务（前提是服务进程还在）。桌面端关闭了 Electron 的硬件加速，避免窗口和修复任务争用同一张显卡导致界面卡死。

打包命令是 `npm run build:win`（electron-builder）。打出来的程序只有界面，不带 Python 端；它依次在启动时的工作目录、工作目录的上一级、Electron 应用路径（打包后是 `resources\app.asar`）及其上一级里找 `pyproject.toml` 来确定项目位置。找不到时“启动服务”会报错，这时可以先在终端运行 `videox serve`，或者把工作目录设为项目根目录（或它的直接子目录）再启动程序。

## 注意事项

- **长视频的内存**：SeedVR2 的 `chunk_size` 默认是 120，每次读入 120 帧分段处理；设为 0 时整段视频一次读入，只适合很短的片段。FlashVSR 的封装会先把整段视频放大到输出分辨率后全部放进内存再交给模型，内存占用随时长和分辨率增长，长片或高分辨率素材建议先剪短或换用其他引擎。
- **音轨和其他流**：回封只保留原片的第一条音轨，其他音轨和字幕不会保留。设置了 SeedVR2 的 `skip_first_frames` 但读不到原片帧率时，为免音画错开，成片不带音轨。
- **覆盖和中间文件**：已存在的输出文件会被直接覆盖，不会提示。回封失败或任务取消时，`<输出名>.video-only.mp4` 会留在输出目录里。

## 目录结构

```text
src/videox/
  cli.py            命令行入口
  server.py         FastAPI 本机 API
  setup_env.py      videox setup：ffmpeg、PyTorch、上游代码和引擎虚拟环境
  doctor.py         环境检查，供 videox doctor 和 GET /system 共用
  paths.py          项目根目录和各类数据目录
  jobs/             任务模型、内存任务存储、任务执行（子进程、进度、取消、音轨回封）
  engines/          引擎协议、注册表、参数 schema 工具、进度协议
    seedvr2/  flashvsr/  swiftvr/  realbasicvsr/
    realesrgan/  swinir/  scunet/  gfpgan/   图片引擎，共用 image_engine.py 和 image_io.py
  media/            ffprobe/ffmpeg 调用、图片和视频的区分
  hardware/         通过 nvidia-smi 查询显卡，默认显卡和批大小
desktop/            Electron + React + Tailwind 桌面界面
scripts/setup.ps1   一键安装脚本
third_party/*.rev   上游仓库的固定提交号
tests/              pytest 测试
```

运行时生成、不进版本库的目录：`.venv`、`models`、`output`、`tools`、`.work`，以及 `third_party` 下的各引擎检出。

项目根目录默认按源码位置推断。命令行和 API 可以用环境变量 `VIDEOX_ROOT` 指定根目录，它要指向完整的项目目录（含 `src`、`third_party`、`.venv`），因为 FlashVSR 和 SwiftVR 的子进程从 `<根目录>\src` 导入 `videox`。桌面端不读这个变量，总是按上面的方式查找 `pyproject.toml`。

## 开发

运行测试：

```powershell
.\.venv\Scripts\python -m pytest
```

桌面端检查：

```powershell
cd desktop
npm run lint
npm run typecheck
```

### 新增引擎

在 `src/videox/engines/` 下新建一个包，实现 `engines/base.py` 里的 `RestorationEngine` 协议，再在 `engines/registry.py` 的 `default_registry()` 里注册。注册后，引擎会出现在 `GET /engines` 和桌面表单里。

- `id` 属性：引擎的唯一标识，注册表按它索引。
- `manifest()`：返回 id、名称、说明、能力（`capabilities`，必填，取值 `upscale`、`interpolate`、`face`、`denoise`）、支持的媒体（默认只有 `video`）、显卡策略（`multi` 或 `single`，默认 `multi`）和参数 schema。schema 顶层的 `groups` 决定分组的标题和顺序，每个字段可以带 `group`、`media`、`enum_labels`、`enum_descriptions`（每个选项一句说明，显示在下拉框下方）、`description`；桌面界面只为 string、integer、number、boolean 类型的字段生成表单。可以用 `engines/fields.py` 里的辅助函数构造。
- 字段之间的联动：`visible_when` 让字段只在前面某个字段取指定值时显示和生效，`overrides` 在前面字段取指定值时替换默认值、范围或选项（用 `fields.conditional()` 构造）。条件只能引用排在前面的字段，服务端的 `resolve_params()` 和桌面界面按字段顺序用同一套规则解析，隐藏的字段不会发给引擎。
- `build_command()`：把任务转换成一条子进程命令行，引擎在独立进程里运行。视频任务收到的 `output_path` 是中间文件 `<输出名>.video-only.mp4`，引擎只需要写出画面，音轨回封由任务执行器统一处理。
- `parse_log_line()`：从引擎输出中解析阶段和进度，任务执行器只调用这个方法。要用通用协议，就在这里调用 `engines/progress.py` 的 `parse_engine_line()`。这样子进程每行打印 `{"videox_progress": {"phase": ..., "current": ..., "total": ...}}` 就能报告进度，打印 `{"videox_preview": {"frame": N}}`（N ≥ 1）就能通知新的预览帧。子进程里可以直接用 `emit_progress()`；预览可以用 `engines/preview_io.py` 的 `publish_pair()`：第一个参数传 `os.environ.get("VIDEOX_PREVIEW_DIR")`（任务执行器为每个任务设置的预览目录），它会把 `output.jpg`（必需）和 `source.jpg`（可选）写进这个目录，再发出预览事件。
- `working_directory()`、`subprocess_env()`：子进程的工作目录和环境变量。
- 可选 `model_directory(root)`：返回这个引擎的权重目录。不提供时，`ctx.model_dir` 指向 SeedVR2 的 `models\SEEDVR2`。
- 可选 `resolve_python(ctx)`：返回子进程使用的 Python。不提供时用主虚拟环境的 Python。

只处理图片、在主环境运行的引擎可以继承 `engines/image_engine.py` 的 `MainImageEngine`，再在 `doctor.py` 的 `_IMAGE_ENGINES` 里登记。其他引擎在 `setup_env.py` 的安装步骤和 `doctor.py` 的 `_engine_statuses()` 里都是逐个写的。新引擎如果要检出上游代码、用独立虚拟环境或检查权重，需要在这两处补上，否则 `videox setup` 不会准备它，`videox doctor` 和系统页也不会显示它。
