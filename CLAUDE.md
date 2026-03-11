# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A ComfyUI custom node plugin that wraps the Google Gemini API for image analysis, generation, and editing. Communicates with Gemini via raw REST requests (no official SDK), defaulting to a third-party proxy (`yunwu.ai`).

## Development Commands

```bash
# Environment setup
python -m venv .venv && .venv\Scripts\activate
pip install requests pillow torch numpy pytest

# Syntax check
python -m compileall .

# Run integration tests (requires valid API key in gemini_config.json)
python test_google_api.py
```

## Architecture

### Core module: `google_api.py`

All logic lives in one file, organized into four layers:

1. **Tensor/PIL utilities** — `pil2tensor()`, `tensor2pil()` for ComfyUI ↔ PIL conversion.
2. **Config persistence** — `get_config()` / `save_config()` read/write `gemini_config.json` (stored next to `__init__.py`). Config holds `api_key`, `base_url`, and a `key_base_url_map` dict that remembers which base_url was used with each API key.
3. **`GoogleGeminiClient`** — HTTP client that builds URLs, resolves base_url (runtime param → key_base_url_map → config default → `DEFAULT_BASE_URL`), and calls `generateContent`. Uses `Authorization: Bearer` header plus `?key=` query param.
4. **ComfyUI node classes** — Each node creates its own `GoogleGeminiClient` instance in `__init__` and calls `apply_runtime_config()` when the user supplies `api_key`/`base_url` at runtime.

### Node classes

| Class | Purpose | Key method |
|---|---|---|
| `GoogleGeminiImageAnalysis` | Vision analysis (up to 4 images) | `analyze()` |
| `GoogleGeminiImageGeneration` | Text-to-image generation | `generate()` |
| `GoogleGeminiImageEditing` | Image editing with prompt (up to 4 input images) | `edit()` |
| `GoogleGeminiConfigManager` | Set/get/clear API key and base_url | `manage()` |

### Node registration

`__init__.py` imports node classes and exposes `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. Duplicate registration maps also exist at the bottom of `google_api.py`.

## Key Patterns

- Every node accepts optional `api_key` and `base_url` string inputs that override the saved config.
- `apply_runtime_config()` persists the key↔url pair on first use, so subsequent runs remember the mapping.
- `normalize_base_url()` strips whitespace and trailing slashes for consistent storage/comparison.
- Image nodes use `responseModalities: ["TEXT", "IMAGE"]` in generation config; analysis node only returns text.
- All node methods return tuples matching their `RETURN_TYPES`; errors are returned as string values (not raised).

## Conventions

- Chinese comments and print statements throughout; maintain this style.
- 4-space indentation, type hints on function signatures.
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- When adding a new node, update both `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py` and at the bottom of `google_api.py`.
