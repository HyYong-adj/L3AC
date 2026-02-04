#!/usr/bin/env bash
# Create a temporary config derived from a base and run a short, quick training
# This copies the given config and reduces training length for a smoke test.
set -euo pipefail
BASE_CONFIG=${1:-3kbps_music}
TMP_CONFIG="tmp_quick_${BASE_CONFIG}.toml"

python - <<PY
from pathlib import Path
import tomli, tomli_w
base = Path('src/model/exp/configs') / f"{BASE_CONFIG}.toml"
conf = tomli.loads(base.read_text())
# small quick-run overrides
conf['train_epoch_num'] = 1
conf['train_data']['sample_num'] = 2
conf['train_data']['batch_size'] = 1
conf['train_data']['max_seconds'] = 2
conf['eval_data']['sample_num'] = 1

Path('%s').write_text(tomli_w.dumps(conf))
print('Wrote quick config to', '%s')
PY

echo "Launching quick training with config $TMP_CONFIG"
export WANDB_DISABLED=true
accelerate launch --num_processes=1 $(pwd)/src/main.py --config "${TMP_CONFIG%.toml}"

# cleanup
rm -f "$TMP_CONFIG"
echo "Quick run finished"