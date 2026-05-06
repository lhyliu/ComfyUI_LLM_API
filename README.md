# ComfyUI_LLM_API

用于 ComfyUI 的 OpenAI-compatible Chat Completions 节点，支持文本与图片输入。

OpenAI-compatible Chat Completions node for ComfyUI, with text and image input support.

## 安装 / Install

在本插件目录运行：

Run in this custom node directory:

```bash
pip install -r requirements.txt
```

安装后重启 ComfyUI。

Restart ComfyUI after installation.

## 安全 / Security

推荐使用 `api_key_env` 从环境变量读取 API key。`api_key` 明文字段会保存进 workflow，请不要分享包含明文 key 的 workflow、截图、日志或 JSON。

Prefer `api_key_env` for API keys. The plain `api_key` widget is saved into workflows, so do not share workflows, screenshots, logs, or JSON files containing a key.

PowerShell 示例 / Example:

```powershell
$env:OPENAI_API_KEY='...'
```

## 快速开始 / Quick Start

1. 添加节点 `LLM API Chat`。
2. 选择 `provider_profile`。
3. 预设 provider 可留空 `api_baseurl`；自定义 provider 需要填写 base URL。
4. 推荐填写 `api_key_env`，例如 `OPENAI_API_KEY`。
5. 填写 `model`、`system_prompt`、`prompt`。
6. 需要图片理解时连接 `images`，并确认 provider/model 支持图片。

1. Add the `LLM API Chat` node.
2. Select `provider_profile`.
3. Leave `api_baseurl` empty for preset providers; custom providers need a base URL.
4. Prefer `api_key_env`, for example `OPENAI_API_KEY`.
5. Set `model`, `system_prompt`, and `prompt`.
6. Connect `images` only when the provider/model supports image input.

## 节点 / Node

- Node ID: `LLM_API_CHAT_NODE`
- Display name: `LLM API Chat`
- Outputs: `response`, `error`

成功时 `response` 有内容且 `error` 为空；失败时 `response` 为空且 `error` 以 `LLM API Error` 开头。

On success, `response` contains model text and `error` is empty. On failure, `response` is empty and `error` starts with `LLM API Error`.

## 主要输入 / Inputs

| Input | 中文说明 | English |
| --- | --- | --- |
| `provider_profile` | provider 预设 | provider preset |
| `api_baseurl` | 可选 base URL 覆盖 | optional base URL override |
| `api_key` | 明文 API key，不推荐分享 | plain API key; do not share |
| `api_key_env` | 环境变量名，优先使用 | environment variable name, preferred |
| `model` | 模型名称 | model name |
| `system_prompt` | system 消息 | system message |
| `prompt` | user 消息 | user message |
| `temperature` | 采样温度 | sampling temperature |
| `execution_seed` | 仅用于 ComfyUI 重新执行/缓存扰动，不传 API | ComfyUI execution/cache control only; not sent to API |
| `filter_thinking` | 移除 `<think>...</think>` 内容 | remove `<think>...</think>` blocks |
| `max_images` | 最多发送图片数 | maximum images to send |
| `image_preset` | 图片编码预设 | image encoding preset |
| `custom_image_max_mb` | custom 预设下的单图 MB 上限 | per-image MB limit for `custom` |
| `custom_image_max_side` | custom 预设下的长边像素上限 | long-side pixel limit for `custom` |
| `images` | 可选 ComfyUI 图片 batch | optional ComfyUI image batch |

## Provider 预设 / Provider Presets

| Profile | Default base URL | Images |
| --- | --- | --- |
| `custom_text_image` | custom/empty | yes |
| `custom_text_only` | custom/empty | no |
| `openai` | `https://api.openai.com/v1` | yes |
| `deepseek` | `https://api.deepseek.com` | no |
| `qwen_dashscope_cn` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | yes |
| `doubao_ark_cn` | `https://ark.cn-beijing.volces.com/api/v3` | yes |
| `zhipu_glm` | `https://open.bigmodel.cn/api/paas/v4` | yes |
| `minimax_global` | `https://api.minimax.io/v1` | no |

文本-only profile 连接图片时会在 API 调用前返回错误。

Text-only profiles reject connected images before making an API call.

## 图片预设 / Image Presets

| Preset | Max size | Max side | 用途 / Use |
| --- | ---: | ---: | --- |
| `fast` | 0.5 MB | 768 px | 快速预览 / quick preview |
| `balanced` | 1.5 MB | 1280 px | 默认推荐 / recommended default |
| `detail` | 3.0 MB | 1600 px | 更多细节 / more detail |
| `ocr_high` | 5.0 MB | 2048 px | OCR、截图、UI、图表 / OCR, screenshots, UI, diagrams |
| `custom` | custom | custom | 手动调优 / manual tuning |

MB 为十进制。大多数工作流建议从 `balanced` 开始。

MB values are decimal. Start with `balanced` for most workflows.

## 示例 / Examples

- `examples/text_only_workflow.json`
- `examples/image_chat_workflow.json`

示例使用 `api_key_env` 占位，不包含 API key。图片示例包含 `examples/example.png` 作为可直接加载的占位图。

Examples use `api_key_env` placeholders and contain no API keys. The image example includes `examples/example.png` as a loadable placeholder image.

## 发布 / Release

Registry metadata lives in `pyproject.toml`, including `[project]` and `[tool.comfy]`. Before publishing, confirm `PublisherId`, `Repository`, `version`, and `LICENSE` match your target release account and repository.

Registry 发布元数据位于 `pyproject.toml`，包含 `[project]` 与 `[tool.comfy]`。发布前请确认 `PublisherId`、`Repository`、`version` 与 `LICENSE` 符合目标发布账号和仓库。

## 手动测试 / Manual Smoke Test

该脚本只在手动运行时发送真实请求，默认单元测试不会联网。

This script sends real API requests only when run manually. Unit tests stay offline.

```powershell
$env:LLM_API_PROVIDER_PROFILE='openai'
$env:LLM_API_KEY='...'
$env:LLM_API_MODEL='gpt-4.1-mini'
E:\ComfyUI-aki-v2\python\python.exe .\scripts\smoke_provider.py
```

图片测试可额外设置：

For image smoke testing, also set:

```powershell
$env:LLM_API_IMAGE_PATH='C:\path\to\image.png'
```

## 不支持 / Not Supported

本节点不支持视频、文件上传、streaming 或 Responses API。

This node does not support video, file upload, streaming, or the Responses API.
