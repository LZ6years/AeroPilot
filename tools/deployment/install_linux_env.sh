#!/usr/bin/env bash
# Install the Python client/evaluation environment, never drivers or simulators.
# Usage: bash tools/deployment/install_linux_env.sh /absolute/env/bin/python
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash tools/deployment/install_linux_env.sh /absolute/env/bin/python'
  exit 0
fi
if [[ $# != 1 || $1 != /* || ! -x $1 ]]; then
  echo 'Provide one absolute executable Python path (Python 3.10).' >&2
  exit 2
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo 'This installer targets Linux x86_64 packaged simulator deployments.' >&2
  exit 2
fi
PY=$1
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
"$PY" -c 'import sys; print(sys.executable); assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for the pinned AirSim dependencies"'
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install --no-build-isolation -r requirements-airsim.txt
"$PY" -m pip install -r requirements-eval.txt 'psutil>=5.9,<8'
"$PY" -m pip check
"$PY" - <<'PY'
import importlib
import sys
print('Interpreter:', sys.executable)
for name in ('numpy', 'cv2', 'PIL', 'openai', 'airsim', 'msgpackrpc', 'unrealcv', 'psutil', 'matplotlib'):
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'import OK'))
print('Python environment ready. Simulator assets and GPU validation are separate steps.')
PY
