#!/usr/bin/env bash
# ─── AI Research Assistant — Local setup script ───────────────────────────────
# Run once: bash scripts/setup.sh
set -e

echo "── AI Research Assistant Setup ──"

# 1. Check UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi
echo "✓ UV: $(uv --version)"

# 2. Create virtual environment + install dependencies
echo "Installing dependencies..."
uv sync --all-extras
echo "✓ Dependencies installed"

# 3. Copy .env.example → .env (if .env doesn't exist)
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example — fill in your API keys"
else
    echo "✓ .env already exists"
fi

# 4. Start local services
echo "Starting Postgres and Redis..."
docker compose up -d
echo "✓ Services started"

# 5. Wait for Postgres to be ready
echo "Waiting for Postgres..."
sleep 3
until docker compose exec postgres pg_isready -U postgres &> /dev/null; do
    sleep 1
done
echo "✓ Postgres ready"

# 6. Run Phase 1 tests
echo "Running Phase 1 tests..."
uv run pytest tests/test_config.py -v
echo "✓ Tests passed"

echo ""
echo "── Setup complete ──"
echo "Next: fill in API keys in .env, then run:"
echo "  uv run uvicorn app.main:app --reload"
echo "  curl http://localhost:8000/health"
