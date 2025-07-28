import subprocess
import multiprocessing
from IPython.display import clear_output
import torch
import numpy as np
import os
import random
import random, json
from functions_utils_pomo import *
import argparse

def update_elected_agent(metrics):
    shared_path="./Data/shared_state.json"
    
    elected = random.randint(0,len(metrics)-1)
    with open(shared_path, "w") as f:
        json.dump({"elected": metrics[elected]}, f)
    print(f"[RANDOM] Agent élu: {metrics[elected]}")

def run_agent(metric, agent_id, start, end):
    """Fonction pour exécuter un agent individuel"""
    
    #print("\nBien arrivé dans la fonction run_agent !")
    
    suffix = "r100_cf3"
    model_name = f"pomo_agent_{metric}"
    cmd = [
        "python3", "agent_run.py",
        "--model_name", model_name,
        "--hyper_params", "hyper_params.json",
        "--dataset", "df_pc_real.pkl",
        "--start", str(start),
        "--end", str(end),
        "--constraint_factor_veh", "1",
        "--constraint_factor_ff", "1",
        "--save_metrics_as", f"agent_pomo_metrics_{metric}_{suffix}",
        "--train",
        "--agent_model","dqn",
        "--eps_start","1",
        "--agent_id", str(agent_id)
    ]
    
    #print(start,end)
    
    path = f"rw_{model_name}_{suffix}.json"
    if start > 1 and os.path.exists(f"./Reward_weights/{path}"):
        file_name = path
        print(f"[INFO] Fichier de reward précédemment entraîné chargé : {file_name}")
    else:
        file_name = f"rw_{metric}_{suffix}.json"
        print(f"[INFO] Fichier de reward de base chargé : {file_name}")

    cmd += ["--reward_weights", file_name]
    #print(path,os.getcwd(),os.path.exists(f"./Reward_weights/{path}"))
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    line_count = 0
    for line in process.stdout:
        line_count += 1
        if line_count % 100 == 0:
            clear_output(wait=True)

        print(line.strip())


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Agent parameters (start and end)")
    parser.add_argument("--start", type=int, default=1, help="start from num_inter")
    parser.add_argument("--end", type=int, default=40, help="end after num_inter")
    args = parser.parse_args()
    
    metrics = get_metrics(os.getcwd())
    metrics = metrics[:2]
    
    num_agents = len(metrics)
    print(f"Nombre d'agents = {num_agents}. On démarre...")

    update_elected_agent(metrics)
    
    tasks = [(metric, i, args.start, args.end) for i, metric in enumerate(metrics)]
    with multiprocessing.Pool(processes=num_agents) as pool:
        pool.starmap(run_agent, tasks)