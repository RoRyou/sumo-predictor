#!/usr/bin/env bash
# Phase B: build merged features parquet, run pose-only XGB, then retrain
# the v4 stack on the pose-augmented feature matrix.
set -u
cd /Users/liang.lu/project/sumo_pred

echo "=== Step 1: build features_with_pose.parquet + pose-only XGB ==="
conda run -n sumo_pred python scripts_tmp/build_features_with_pose.py \
  2>&1 | tee reports/pose_phase2/phaseB_build.log

echo
echo "=== Step 2: retrain v4 stack on features_with_pose.parquet ==="
conda run -n sumo_pred python -m src.training.train_struct run \
  --features data/processed/features_with_pose.parquet \
  --val-basho 202311 --test-start 202401 \
  --out-dir runs/struct_v14_with_pose \
  --meta xgb --calib isotonic \
  --xgb-params runs/xgb_best_params.json \
  --lgbm-params runs/lgbm_best_params.json \
  --cat-params runs/cat_best_params.json \
  2>&1 | tee reports/pose_phase2/phaseB_train.log | tail -80

echo
echo "=== final metrics ==="
cat runs/struct_v14_with_pose/metrics.json
