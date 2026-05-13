#!/usr/bin/env bash
# Download SUMO PRIME TIME 2024 recaps, trimmed to first ~3 min.
set -u
cd /Users/liang.lu/project/sumo_pred

# key=vid, separated by space
PAIRS=(
  "202401_d1 yIvxIjHM328"
  "202403_d1 kxyr1bAMotw"
  "202403_d15 khxY7GzHmjQ"
  "202405_a _lCrihPzLJA"
  "202407_d1 uNEarPSvubE"
  "202407_d15 0WMjApHe5kw"
  "202407_wrap sGO9P8UM6EU"
  "202409_d1 RWtV5nixL-M"
  "202411_d1 e_TYVAt1fu8"
  "202411_d15 4h8mTqm03l4"
)

OUT=/Users/liang.lu/project/sumo_pred/data/videos/aligned
mkdir -p "$OUT"

pids=""
for pair in "${PAIRS[@]}"; do
  key="${pair%% *}"
  vid="${pair##* }"
  out="$OUT/${key}.mp4"
  if [[ -s "$out" ]]; then
    echo "[skip] $key already exists"
    continue
  fi
  (
    yt-dlp \
      -f "best[ext=mp4][height<=480]/best[height<=480]/best" \
      --download-sections "*0:00-3:30" \
      --force-keyframes-at-cuts \
      --merge-output-format mp4 \
      -o "$out" \
      "https://www.youtube.com/watch?v=$vid" \
      > /tmp/yt_${key}.log 2>&1 \
      && echo "[ok] $key" || echo "[fail] $key"
  ) &
  pids="$pids $!"
done

for pid in $pids; do
  wait "$pid" || true
done
echo "all downloads finished"
ls -la "$OUT"
