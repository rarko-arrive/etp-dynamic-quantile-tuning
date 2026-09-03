#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root:
#
#   ./scripts/install.sh
#   make install   # equivalent
#
# Single-shot cross-platform setup: creates .env if missing, builds the uv
# virtual environment, ensures .venv exists for the IDE (real dir or symlink),
# registers the Jupyter kernel, and trusts .envrc.

cd "$(dirname "$0")/.."

PROJECT_NAME="$(basename "$PWD")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

# --- .env bootstrap ---
# Create .env from .env.example on first run so the user never has to
# cp a file manually. Only needed for SNOWFLAKE_PASSWORD (make data) and
# UV_PROJECT_ENVIRONMENT; everything else works without it.
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo "  → Edit SNOWFLAKE_PASSWORD if you need 'make data' (Snowflake access)"
fi

# Load .env so UV_PROJECT_ENVIRONMENT (and any future overrides) take effect.
set -a; source .env; set +a
# Remove any trailing slash so mkdir / path-join don't produce double slashes.
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT%/}"
# dotenv / quoted .env values leave ~ literal; expand for mkdir and uv.
if [[ "$UV_PROJECT_ENVIRONMENT" == \~/* ]]; then
    UV_PROJECT_ENVIRONMENT="$HOME/${UV_PROJECT_ENVIRONMENT:2}"
fi

# --- virtual environment ---
# Default: .venv in the repo root (Mac / local disk clones).
# Azure ML: set UV_PROJECT_ENVIRONMENT to an absolute path on local disk
# (e.g. /home/azureuser/uv-venvs/etp-dqt) — cloudfiles is slow and
# /mnt is wiped on compute restarts. Bootstrap then symlinks .venv -> that
# path so Cursor/VS Code's ${workspaceFolder}/.venv/bin/python works.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv}"
# Clear any stale VIRTUAL_ENV from a parent shell or direnv so uv doesn't warn
# about a mismatch with the project environment path.
unset VIRTUAL_ENV

if [[ "$UV_PROJECT_ENVIRONMENT" != /* ]]; then
    UV_PROJECT_ENVIRONMENT="$(pwd)/$UV_PROJECT_ENVIRONMENT"
fi

mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

uv sync --all-groups

# IDE bridge: committed settings use ${workspaceFolder}/.venv/bin/python.
REPO_VENV="$(pwd)/.venv"
if [[ "$UV_PROJECT_ENVIRONMENT" != "$REPO_VENV" ]]; then
    if [[ -e .venv && ! -L .venv ]]; then
        echo "ERROR: .venv exists and is not a symlink, but" >&2
        echo "       UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT" >&2
        echo "       Move/remove .venv, or set UV_PROJECT_ENVIRONMENT=.venv" >&2
        exit 1
    fi
    ln -sfn "$UV_PROJECT_ENVIRONMENT" .venv
    echo "Linked .venv -> $UV_PROJECT_ENVIRONMENT (workspace interpreter)"
fi

# --- Jupyter kernel ---
# Display name tracks the repo directory (e.g. Python (etp-dqt)).
uv run python -m ipykernel install \
    --user \
    --name "$PROJECT_NAME" \
    --display-name "Python ($PROJECT_NAME)"

# --- direnv ---
# Idempotent trust of .envrc: always runs, even if previously allowed, so
# any future edits to .envrc (e.g. from a git pull) re-trigger allow.
if command -v direnv >/dev/null 2>&1 && [[ -f .envrc ]]; then
    echo "Trusting .envrc via direnv allow..."
    direnv allow
elif command -v direnv >/dev/null 2>&1; then
    echo "Note: .envrc not found — skipping direnv allow"
else
    echo
    echo "Note: direnv is not installed — UV_PROJECT_ENVIRONMENT won't be"
    echo "auto-exported when you cd into this repo in a new shell."
    echo "Install it: brew install direnv  (macOS)  /  apt install direnv  (Linux)"
    echo "Then run: direnv allow"
fi

echo
echo "Ready."
echo
echo "Interpreter (uv / shells):"
echo "  ${UV_PROJECT_ENVIRONMENT}/bin/python"
echo "Interpreter (Cursor / VS Code):"
echo "  $(pwd)/.venv/bin/python"
echo "Jupyter kernel:"
echo "  Python ($PROJECT_NAME)"
