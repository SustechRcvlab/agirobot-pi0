export http_proxy=http://localhost:1080
export https_proxy=http://localhost:1080
uv run python - <<'PY'
import requests
r = requests.get("https://www.google.com", timeout=5)
print(r.status_code)
PY
