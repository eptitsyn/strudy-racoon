#!/usr/bin/env bash
# Spin up the backend (stub model) and exercise the public API end-to-end.
# Runs without Docker so it's fast enough for every CI commit.

set -euo pipefail

PORT="${PORT:-8765}"
BASE="http://127.0.0.1:${PORT}"
LOG_FILE="$(mktemp -t backend-smoke.XXXXXX.log)"
PID=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

cleanup() {
    if [[ -n "${PID}" ]]; then
        kill "${PID}" 2>/dev/null || true
        wait "${PID}" 2>/dev/null || true
    fi
    if [[ "${KEEP_LOG:-0}" != "1" && -f "${LOG_FILE}" ]]; then
        rm -f "${LOG_FILE}"
    fi
}
trap cleanup EXIT

echo "==> starting backend on :${PORT} (logs: ${LOG_FILE})"
MODELS_CONFIG_PATH="${MODELS_CONFIG_PATH:-./models.yaml}" \
LOG_LEVEL="${LOG_LEVEL:-WARNING}" \
DETECTOR_MAX_PARALLEL="${DETECTOR_MAX_PARALLEL:-2}" \
    uv run uvicorn backend.main:app --host 127.0.0.1 --port "${PORT}" --workers 1 \
    > "${LOG_FILE}" 2>&1 &
PID=$!

echo "==> waiting for /health/ready"
deadline=$(($(date +%s) + 60))
until curl -fsS "${BASE}/health/ready" >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then
        echo "!! backend never reported ready" >&2
        echo "---- backend logs ----" >&2
        cat "${LOG_FILE}" >&2
        exit 1
    fi
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "!! backend process died" >&2
        cat "${LOG_FILE}" >&2
        exit 1
    fi
    sleep 0.5
done

echo "==> GET /v1/models"
models_body=$(curl -fsS "${BASE}/v1/models")
echo "${models_body}"
echo "${models_body}" | "${PYTHON_BIN}" -c "
import json, sys
data = json.loads(sys.stdin.read())
assert data['active'] == 'stub', f\"unexpected active model: {data['active']!r}\"
names = {m['name'] for m in data['available']}
assert 'stub' in names, f'stub missing from available: {names}'
print('models endpoint OK')
"

echo "==> POST /v1/detect"
detect_body=$(curl -fsS -H 'X-Request-ID: 11111111-1111-1111-1111-111111111111' \
    -H 'Content-Type: application/json' \
    -d '{"text":"This text is being analysed by the smoke test."}' \
    "${BASE}/v1/detect")
echo "${detect_body}"
echo "${detect_body}" | "${PYTHON_BIN}" -c "
import json, sys
data = json.loads(sys.stdin.read())
assert data['verdict'] in {'ai', 'human', 'unknown'}, data
assert 0.0 <= data['ai_probability'] <= 1.0, data
assert 0.0 <= data['human_probability'] <= 1.0, data
assert data['model']['name'] == 'stub', data['model']
assert data['request_id'] == '11111111-1111-1111-1111-111111111111', data
print('detect endpoint OK')
"

echo "==> POST /v1/models/switch (idempotent on stub)"
switch_body=$(curl -fsS -H 'Content-Type: application/json' \
    -d '{"name":"stub"}' "${BASE}/v1/models/switch")
echo "${switch_body}"
echo "${switch_body}" | "${PYTHON_BIN}" -c "
import json, sys
data = json.loads(sys.stdin.read())
assert data['active']['name'] == 'stub', data
print('switch endpoint OK')
"

echo "==> ALL SMOKE CHECKS PASSED"
