#!/bin/bash
# 7 Live desktop launcher.
#
# Starts `python -m live7` if nothing is listening on 127.0.0.1:7879, then
# opens the page in a Chrome/Chromium app window with audio autoplay allowed
# (no Start click needed). Falls back to Safari if no Chrome-family browser
# is installed.
#
# Double-click it or call it from a .app wrapper; it never needs a terminal.

set -u

URL="http://127.0.0.1:7879"
PORT=7879
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$HOME/.seven"
mkdir -p "$LOG_DIR"

find_python() {
    if [ -x "$ROOT/.venv/bin/python" ]; then
        printf '%s' "$ROOT/.venv/bin/python"; return 0
    fi
    command -v python3 || command -v python
}

server_up() {
    # nc: 1 = listening, 0 = refused; lsof fallback for systems without nc.
    if command -v nc >/dev/null 2>&1; then
        nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1 && return 0
    fi
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | grep -q 127.0.0.1 && return 0
    return 1
}

if ! server_up; then
    PY="$(find_python)"
    if [ -n "$PY" ]; then
        cd "$ROOT" || exit 1
        nohup "$PY" -m live7 >>"$LOG_DIR/live7_launcher.log" 2>&1 &
        disown 2>/dev/null || true
    fi
    # give the server a moment to bind (bounded; never blocks forever)
    for _ in $(seq 1 30); do
        server_up && break
        sleep 1
    done
fi

CHROME_CANDIDATES=(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    "/Applications/Chromium.app/Contents/MacOS/Chromium"
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
)

for browser in "${CHROME_CANDIDATES[@]}"; do
    if [ -x "$browser" ]; then
        exec "$browser" \
            --app=http://127.0.0.1:7879 \
            --autoplay-policy=no-user-gesture-required \
            --no-first-run \
            --disable-session-crashed-bubble \
            >/dev/null 2>&1
    fi
done

# Fallback: Safari (autoplay needs to be allowed once for 127.0.0.1 in
# Safari > Settings > Websites > Auto-Play; a Start click works regardless).
open -a Safari "$URL"
