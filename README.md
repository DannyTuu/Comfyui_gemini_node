# ComfyUI Gemini Node | ComfyUI Gemini 节点

[中文](#中文说明) | [English](#english)

## 中文说明

一个 ComfyUI 自定义节点插件：通过 REST API 调用 Google Gemini 兼容接口，实现图像分析、文生图、图像编辑。

### 功能

- 图像分析（最多 4 张输入图，可按需修改）
- 文生图（返回图像 + 文本描述 + JSON 响应信息）
- 图像编辑（最多 4 张输入图）
- 配置管理：保存/读取/清空 `api_key` 与 `base_url`

### 安装

1. 将本仓库放到 `ComfyUI/custom_nodes/` 目录下：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/DannyTuu/Comfyui_gemini_node.git
```

2. 安装依赖（在你的 ComfyUI Python 环境中执行）：

```bash
python -m pip install -U requests pillow numpy
```

备注：大多数 ComfyUI 环境已经自带 `torch`；如果缺失再安装。

3. 重启 ComfyUI。

### 配置

有两种方式：

- 推荐：在 ComfyUI 里使用节点 `Google Gemini - Config Manager`，执行 `set/get/clear`。
- 手动：在插件目录旁创建 `gemini_config.json`（已在 `.gitignore` 中忽略，避免误提交）。示例结构：

```json
{
  "api_key": "YOUR_API_KEY",
  "base_url": "https://your-gemini-compatible-endpoint/v1beta",
  "key_base_url_map": {}
}
```

所有图像相关节点都支持运行时传入 `api_key` / `base_url` 作为覆盖；当提供 `api_key` 时，会保存该 key 对应的 `base_url`，方便下次自动复用。

推荐API中转：https://yunwu.ai/register?aff=ns1E

### 节点一览

- `Google Gemini - Image Analysis`
  - 输入：`prompt`、可选 `image1..image4`、`api_key/base_url`、`temperature/top_p/max_tokens/system_instruction`
  - 输出：`analysis`（文本）
- `Google Gemini - Image Generation`
  - 输入：`prompt`、`model`、可选 `aspect_ratio/image_size/seed`、`api_key/base_url` 等
  - 输出：`image`、`description`、`response_info`（JSON 字符串）
- `Google Gemini - Image Editing`
  - 输入：`edit_prompt`、可选 `image1..image4`、`aspect_ratio/image_size/seed`、`api_key/base_url` 等
  - 输出：`edited_image`、`result`、`response_info`（JSON 字符串）
- `Google Gemini - Config Manager`
  - 输入：`api_key`、可选 `action(set/get/clear)`、`base_url`
  - 输出：`status`

### 安全提示

- 默认 `base_url` 来自代码里的 `DEFAULT_BASE_URL`。如果你需要走自定义兼容网关，可以自行覆盖 `base_url`。
- `api_key` 会被发送到你配置的 `base_url`；请自行评估风险。
- 不要把真实 key 写进仓库提交；本项目已忽略 `gemini_config.json`、`.env*` 等本地敏感文件。

### 开发与测试（可选）

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install requests pillow torch numpy pytest
python -m compileall .
python -m pytest -q test_google_api.py -k config_manager
```

---

## English

A ComfyUI custom node plugin that calls Google Gemini-compatible REST endpoints for image analysis, text-to-image generation, and image editing.

### Features

- Image analysis (up to 4 input images, adjustable if needed)
- Text-to-image (returns image + description + JSON response info)
- Image editing (up to 4 input images)
- Config manager: set/get/clear `api_key` and `base_url`

### Installation

1. Clone into `ComfyUI/custom_nodes/`:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/DannyTuu/Comfyui_gemini_node.git
```

2. Install dependencies in your ComfyUI Python environment:

```bash
python -m pip install -U requests pillow numpy
```

Note: `torch` is usually already bundled with ComfyUI; install it only if missing.

3. Restart ComfyUI.

### Configuration

Two options:

- Recommended: use the `Google Gemini - Config Manager` node inside ComfyUI (`set/get/clear`).
- Manual: create `gemini_config.json` next to the plugin (already ignored by `.gitignore` to avoid accidental commits):

```json
{
  "api_key": "YOUR_API_KEY",
  "base_url": "https://your-gemini-compatible-endpoint/v1beta",
  "key_base_url_map": {}
}
```

All image nodes accept optional runtime `api_key` / `base_url` overrides. When an `api_key` is provided, the plugin also persists the key-to-base_url mapping for reuse.

Recommended API relay: https://yunwu.ai/register?aff=ns1E

### Nodes

- `Google Gemini - Image Analysis`: returns `analysis` text
- `Google Gemini - Image Generation`: returns `image`, `description`, `response_info` (JSON string)
- `Google Gemini - Image Editing`: returns `edited_image`, `result`, `response_info` (JSON string)
- `Google Gemini - Config Manager`: returns `status`

### Security Notes

- The default `base_url` is defined in code as `DEFAULT_BASE_URL`. If you need a custom compatible gateway, override `base_url` explicitly.
- Your `api_key` is sent to whatever `base_url` you configure. Assess the risk accordingly.
- Do not commit real keys. This repo ignores `gemini_config.json`, `.env*`, etc.
