# Docker Setup (CPU/UI First)

This folder provides a production-style Docker workflow for running the COMPASS UI with public API backends.

## What this is for

- Fast, reproducible onboarding for UI/API usage.
- Reproducible runtime without local Python setup.
- CPU-first portability across macOS, Linux, and Windows (Docker Desktop/WSL).

## What this is not for

- GPU-accelerated local inference in containers.
- HPC/Slurm execution.

Use `hpc/` for GPU/HPC workflows.

## Files in this folder

- `Dockerfile`: default UI/API image (curated lightweight dependencies)
- `Dockerfile.full`: optional full-dependency image (`requirements.txt`)
- `requirements.ui.txt`: curated UI/API dependencies
- `entrypoint.sh`: starts `main.py --ui` and binds to `0.0.0.0:5005`
- `.dockerignore`: build-context exclusions applied via `tar --exclude-from`

## Prerequisites

- Docker Desktop (or Docker Engine) with Buildx available.
- A valid `OPENROUTER_API_KEY` for public API usage.

## Local install vs Docker

Both are valid:

- Local install (`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`) works and is best for active development, debugging, and editing code in-place.
- Docker is recommended for reproducible execution because it:
  - Pins the runtime environment at build time (base OS + Python + wheels actually installed).
  - Avoids host-specific Python/toolchain issues (Apple Silicon vs Intel, Homebrew Python quirks, global site-packages, etc.).
  - Provides a clean dependency surface: the default image installs only the UI/API dependencies (`docker/requirements.ui.txt`) instead of the full `requirements.txt` (which includes optional heavy local-inference deps).
  - Makes “works on my machine” less likely: users run the same container entrypoint and the UI binds predictably to `0.0.0.0:5005`.

## Quickstart (recommended default image)

Run these commands from the repo root (`multi_agent_system/`).
On Windows, run them in WSL or Git Bash (recommended) so the `tar | docker buildx` pipeline works as written.

1. Build image (choose platform):

Apple Silicon Mac (M1/M2/M3):
```bash
tar --exclude-from=docker/.dockerignore -cf - . | docker buildx build \
  --platform linux/arm64 \
  -f docker/Dockerfile \
  -t compass-ui:local \
  --load \
  -
```

Intel Mac / Linux / Windows Docker Desktop:
```bash
tar --exclude-from=docker/.dockerignore -cf - . | docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t compass-ui:local \
  --load \
  -
```

2. Run container:
```bash
export OPENROUTER_API_KEY="<your_openrouter_api_key>"
docker run --rm \
  -p 5005:5005 \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  --name compass-ui \
  compass-ui:local
```

3. Open UI:
- `http://localhost:5005`

## Verification

- Container logs should show:
  - `Launching COMPASS Dashboard...`
  - `Dashboard live at http://0.0.0.0:5005`
- Optional check:
```bash
curl -I http://localhost:5005
```

## Stop

- If running in foreground: `Ctrl+C`
- If detached: `docker stop compass-ui`

## Optional variant: full dependency image

This image includes local-inference dependencies (`torch`, `transformers`, `bitsandbytes`) for users who explicitly want them.

```bash
tar --exclude-from=docker/.dockerignore -cf - . | docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.full \
  -t compass-full:amd64 \
  --load \
  -
```
