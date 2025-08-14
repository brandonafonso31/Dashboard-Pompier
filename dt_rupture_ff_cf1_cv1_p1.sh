#!/bin/bash
#SBATCH --job-name=agent_run
#SBATCH --output=log/agent-%j.out
#SBATCH --error=log/agent-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mail-type=END             
#SBATCH --mail-user=michael.corbeau@irit.fr    

export WANDB_API_KEY=$(< ~/.wandb_key)


container=/apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif

script=DT_run.py


model="dt"   
years="10y"   
cf="1"
cv="1"
suffix="cf${cf}_cv${cv}_p1"
rwd="rupture_ff"


# ---------- TRAIN ----------
train_cmd="python3 -u ${script} \
  --model_name agent_${model}_${years}_${suffix}_${rwd} \
  --agent_model ${model} \
  --hyper_params dt_hyper_params.json \
  --reward_weights rw_${rwd}.json \
  --dataset df_pc_fake_${years}.pkl \
  --start 1 \
  --end 530880 \
  --eps_start 1 \
  --constraint_factor_ff ${cf} \
  --constraint_factor_veh ${cv} \
  --save_metrics_as metrics_${model}_${suffix}_${rwd} \
  --train"

echo "=== Starting training run ==="

srun singularity exec --nv ${container} bash -c "${train_cmd}"

# ---------- TEST ----------
test_cmd="python3 -u ${script} \
  --model_name agent_${model}_${years}_${suffix}_${rwd} \
  --agent_model ${model} \
  --hyper_params dt_hyper_params.json \
  --reward_weights rw_${rwd}.json \
  --dataset df_pc_real.pkl \
  --start 1 \
  --end 53088 \
  --eps_start 0 \
  --constraint_factor_ff ${cf} \
  --constraint_factor_veh ${cv} \
  --save_metrics_as metrics_${model}_${suffix}_${rwd}_test"

echo "=== Starting test run ==="
srun singularity exec --nv ${container} bash -c "${test_cmd}"
