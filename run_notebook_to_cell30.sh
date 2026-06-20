#!/usr/bin/env bash
#
# Execute trajectory_evaluation_spatio_temporal_all.ipynb up to and including
# cell 30, then stop. The first 30 cells are sliced into a temporary notebook,
# executed in order, and the executed result is written to
# trajectory_evaluation_spatio_temporal_all.cell30.ipynb so the computed
# outputs are preserved.

set -euo pipefail

NOTEBOOK="trajectory_evaluation_spatio_temporal_all.ipynb"
N_CELLS=30
OUTPUT="trajectory_evaluation_spatio_temporal_all.cell30.ipynb"

cd "$(dirname "$0")"

if [[ ! -f "$NOTEBOOK" ]]; then
    echo "Notebook not found: $NOTEBOOK" >&2
    exit 1
fi

python - "$NOTEBOOK" "$N_CELLS" "$OUTPUT" <<'PY'
import sys
import nbformat
from nbclient import NotebookClient

src_path, n_cells, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

nb = nbformat.read(src_path, as_version=4)
nb.cells = nb.cells[:n_cells]

client = NotebookClient(nb, timeout=-1, kernel_name="python3")
client.execute()

nbformat.write(nb, out_path)
print("Executed first %d cells; wrote %s" % (n_cells, out_path))
PY
