# Repository Guidelines

## Project Structure & Module Organization
- `google_api.py` is the core module: Gemini client logic, image/tensor helpers, config read/write, and ComfyUI node classes.
- `__init__.py` exposes `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` for ComfyUI node registration.
- `test_google_api.py` contains manual/integration-style test functions and usage examples.
- `.vscode/settings.json` stores local editor defaults (Conda-based Python environment).

## Build, Test, and Development Commands
- Create and activate an environment:
  - `python -m venv .venv`
  - `.venv\Scripts\activate`
- Install dependencies used by this plugin/tests:
  - `python -m pip install requests pillow torch numpy pytest`
- Run focused checks:
  - `python -m pytest -q test_google_api.py -k config_manager` (local config flow)
  - `python -m compileall .` (quick syntax smoke check)
- Run end-to-end examples (requires valid API key and network):
  - `python test_google_api.py`

## Coding Style & Naming Conventions
- Follow Python conventions with 4-space indentation and clear type hints.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Keep API transport logic in `GoogleGeminiClient`; keep node classes focused on ComfyUI inputs/outputs.
- When adding nodes, update both registration maps in `__init__.py` (and mirrored maps in `google_api.py` if retained).

## Testing Guidelines
- Prefer isolated tests for helpers (`pil2tensor`, `tensor2pil`, config functions) before network-dependent flows.
- Separate or clearly label tests that call live Gemini endpoints.
- Do not commit real keys; keep local secrets in `gemini_config.json` only.

## Commit & Pull Request Guidelines
- Git history is not available in this snapshot; use concise, imperative commit subjects (Conventional style is recommended), e.g. `fix: handle empty API response`.
- Keep commits focused and scoped to one change.
- PRs should include: summary, touched files, validation steps/outputs, and screenshots or workflow examples for node behavior changes.

## Security & Configuration Tips
- Treat `gemini_config.json` as sensitive local config; exclude it from version control.
- Avoid hardcoding API keys or provider-specific endpoints in test/demo code.
