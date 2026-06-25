#!/usr/bin/env bash
#SBATCH --chdir=/projects/teta/kgovil/mobilitynet-analysis-scripts
#SBATCH --mail-user=kinjal.govil@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
# Execute dtw_time_threshold_sweep.ipynb. The notebook is executed
# in order, and the executed result is written to
# dtw_time_threshold_sweep.executed.ipynb so the computed outputs are preserved.

source setup/activate_conda.sh

set -euo pipefail

NOTEBOOK="dtw_time_threshold_sweep.ipynb"
OUTPUT="dtw_time_threshold_sweep.executed.ipynb"

if [[ ! -f "$NOTEBOOK" ]]; then
    echo "Notebook not found: $NOTEBOOK" >&2
    exit 1
fi

python - "$NOTEBOOK" "$OUTPUT" <<'PY'
import sys
import nbformat
from nbclient import NotebookClient

src_path, out_path = sys.argv[1], sys.argv[2]

nb = nbformat.read(src_path, as_version=4)

client = NotebookClient(nb, timeout=-1, kernel_name="python3")
client.execute()

nbformat.write(nb, out_path)
print("Executed full notebook; wrote %s" % out_path)
PY
