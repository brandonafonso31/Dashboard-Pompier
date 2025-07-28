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
    
    # Créer un dictionnaire pour stocker les récompenses
    dic = {metric: None for metric in metrics}
    # Charger les récompenses depuis les fichiers
    for metric in metrics:
        try:
            with open(f"./Reward_weights/rw_mean_pomo_agent_{metric}_r100_cf3.json", "r") as f:
                dic[metric] = json.load(f)
        except FileNotFoundError:
            print(f"Fichier pour {metric} non trouvé")
            continue
        
    # Trouver la meilleure métrique et sa récompense
    best_metric = max(metrics, key=lambda m: dic.get(m, float('-inf')))
    best_reward = dic[best_metric]

    print(f"\nMeilleure récompense: {best_reward} (agent {best_metric})")

    # Écraser les fichiers avec la meilleure récompense
    for metric in metrics:
        if metric != best_metric:
            # Écraser le fichier mean
            with open(f"./Reward_weights/rw_mean_pomo_agent_{metric}_r100_cf3.json", "w") as f:
                json.dump(best_reward, f)
            
            # Écraser le fichier standard avec le contenu du meilleur agent
            best_file_content = None
            with open(f"./Reward_weights/rw_pomo_agent_{best_metric}_r100_cf3.json", "r") as best_file:
                best_file_content = json.load(best_file)
            
            with open(f"./Reward_weights/rw_pomo_agent_{metric}_r100_cf3.json", "w") as target_file:
                json.dump(best_file_content, target_file)
            
            print(f"Fichiers de {metric} écrasés avec la meilleure récompense de {best_metric}")