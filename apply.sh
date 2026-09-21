#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Apply the HED-YOLO modifications onto an installed Ultralytics package.
#
# This repository contains only the code written for this project, not the
# upstream Ultralytics sources. It is applied as an overlay on top of a
# matching upstream release.
#
# Usage:
#   ./apply.sh              # apply to the active Python environment
#   DRY_RUN=1 ./apply.sh    # show what would be copied, change nothing
# ---------------------------------------------------------------------------
set -euo pipefail

UPSTREAM_VERSION="8.4.137"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-python}"

if ! "$PY" -c "import ultralytics" 2>/dev/null; then
    echo "ERROR: ultralytics is not importable in this environment." >&2
    echo "Install the matching upstream release first:" >&2
    echo "    $PY -m pip install ultralytics==${UPSTREAM_VERSION}" >&2
    exit 1
fi

FOUND="$("$PY" -c 'import ultralytics; print(ultralytics.__version__)')"
if [ "$FOUND" != "$UPSTREAM_VERSION" ]; then
    echo "ERROR: this overlay is written for ultralytics ${UPSTREAM_VERSION}, but ${FOUND} is installed." >&2
    echo "Reinstall the matching release:" >&2
    echo "    $PY -m pip install --force-reinstall ultralytics==${UPSTREAM_VERSION}" >&2
    exit 1
fi

SITE="$("$PY" -c 'import ultralytics, os; print(os.path.dirname(os.path.dirname(ultralytics.__file__)))')"
TARGET="${SITE}/ultralytics"

echo "Upstream : ultralytics ${FOUND}"
echo "Target   : ${TARGET}"
echo "Source   : ${HERE}/ultralytics"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry run] files that would be copied:"
    (cd "${HERE}/ultralytics" && find . -type f -not -name '*.pyc' | sort)
    exit 0
fi

cp -r "${HERE}/ultralytics/." "${TARGET}/"

echo "Overlay applied."
echo
echo "Verify with:"
echo "    $PY -c \"from ultralytics.nn.Convmodules import C3k2_DB, FEM; from ultralytics.nn.SPPmodules import AIF_SPPF; print('ok')\""
