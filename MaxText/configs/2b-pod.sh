#!/bin/bash
set -e

RUN_NAME=$1


# Bfloat16
echo "Starting bfloat16 run"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name="$RUN_NAME-bfloat16" \
    base_emb_dim=4096 \
    per_device_batch_size=4 enable_checkpointing=false \
    steps=33500 metrics_file="bfloat16_metrics.txt"

echo "Finished bfloat16 run"
last_3_lines=$(tail -n 3 bfloat16_metrics.txt)
echo "Printing last 3 lines of metrics:"
echo "${last_3_lines}"

gsutil cp bfloat16_metrics.txt gs://mattdavidow-maxtext-br/${RUN_NAME}_metrics_bfloat16.txt

sleep 60

# AQT
echo "Starting AQT run"
export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name="$RUN_NAME-aqt" \
    base_emb_dim=4096 \
    per_device_batch_size=4 enable_checkpointing=false use_int8_training=True \
    steps=33500 metrics_file="aqt_metrics.txt"

echo "Finished aqt run"
last_3_lines=$(tail -n 3 aqt_metrics.txt)
echo "Printing last 3 lines of metrics:"
echo "${last_3_lines}"

gsutil cp aqt_metrics.txt gs://mattdavidow-maxtext-br/${RUN_NAME}_metrics_aqt.txt