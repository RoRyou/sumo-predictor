#!/usr/bin/env bash
set -u
cd /Users/liang.lu/project/sumo_pred

# key=basho_part separated by space; basho prefix is first 6 chars of key.
PAIRS=(
  "202401_d1 202401"
  "202403_d1 202403"
  "202403_d15 202403"
  "202405_a 202405"
  "202407_d1 202407"
  "202407_d15 202407"
  "202407_wrap 202407"
  "202409_d1 202409"
  "202411_d1 202411"
  "202411_d15 202411"
)

VID_DIR=/Users/liang.lu/project/sumo_pred/data/videos/aligned
SEG_DIR=/Users/liang.lu/project/sumo_pred/reports/pose_phase2
FEAT_DIR=/Users/liang.lu/project/sumo_pred/data/processed/pose_per_segment
mkdir -p "$SEG_DIR" "$FEAT_DIR"

pids=""
for pair in "${PAIRS[@]}"; do
  key="${pair% *}"
  basho="${pair##* }"
  out_feat="$FEAT_DIR/${key}.parquet"
  if [[ -s "$out_feat" ]]; then
    echo "[skip] $key already extracted"
    continue
  fi
  (
    conda run -n sumo_pred python -m src.features.extract_bout_features run \
      --video "$VID_DIR/${key}.mp4" \
      --segments-out "$SEG_DIR/segments_${key}.json" \
      --features-out "$out_feat" \
      --align-to-basho "$basho" \
      --rikishis data/raw/rikishis.parquet \
      --bouts data/raw/bouts.parquet \
      > /tmp/extract_${key}.log 2>&1 \
      && echo "[ok] $key" || echo "[fail] $key"
  ) &
  pids="$pids $!"
done

for pid in $pids; do
  wait "$pid" || true
done

echo "all extractions finished"
ls -la "$FEAT_DIR"
