#!/usr/bin/env bash
# Rebuild the Eye-Q technical documentation PDF from source.
#
#   ./docs/build-pdf.sh
#
# Regenerates the figures, assembles the HTML, then renders it with headless
# Chrome. Chrome is used because it is the only print engine on the target
# machine that supports the paged-media CSS this document relies on.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT="${ROOT}/docs/Eye-Q-Technical-Documentation.pdf"

"${PY}" "${ROOT}/docs/diagrams/build.py"
"${PY}" "${ROOT}/docs/build.py"

"${CHROME}" --headless --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=15000 \
  --print-to-pdf="${OUT}" \
  "file://${ROOT}/docs/eye-q-technical-documentation.html" 2>/dev/null

echo "wrote ${OUT}"
