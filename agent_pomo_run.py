import subprocess
import multiprocessing
from IPython.display import clear_output
import torch
import numpy as np
import os
import random
import random, json, time
from threading import Thread

def update_elected_agent(metrics):
    shared_path="./Data/shared_state.json"
    while True:
        time.sleep(10)  # tous les X secondes
        elected = random.randint(0,len(metrics)-1)
        with open(shared_path, "w") as f:
            json.dump({"elected": metrics[elected]}, f)
        print(f"[PARENT] Agent élu: {metrics[elected]}")

def run_agent(metric, agent_id, elected_id):
    """Fonction pour exécuter un agent individuel"""
    
    #print("\nBien arrivé dans la fonction run_agent !")
    
    suffix = "r100_cf3"
    cmd = [
        "python3", "agent_run.py",
        "--model_name", f"pomo_agent_{metric}",
        "--hyper_params", "hyper_params.json",
        "--reward_weights", f"rw_{metric}_{suffix}.json",
        "--dataset", "df_pc_real.pkl",
        "--start", "1",
        "--end", "40",
        "--constraint_factor_veh", "1",
        "--constraint_factor_ff", "1",
        "--save_metrics_as", f"agent_pomo_metrics_{metric}_{suffix}",
        "--train",
        "--agent_model","dqn",
        "--eps_start","1",
        "--agent_id", str(agent_id), 
        "--elected_id", str(elected_id)
    ]
    #print(f"\nLa commande de l'agent pomo_agent_{metrics} est {cmd}\n")
    
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
    metrics = ['v_degraded', 'v1_not_sent_from_s1']
    # , 'v3_not_sent_from_s3', 'v_not_found_in_last_station', 'z1_VSAV_sent', 'rupture_ff']
    
    num_agents = len(metrics)
    print(f"Nombre d'agents = {num_agents}. On démarre...")

    Thread(target=update_elected_agent, args=(metrics,), daemon=True).start()
    
    tasks = [(metric, i, num_agents) for i, metric in enumerate(metrics)]
    with multiprocessing.Pool(processes=num_agents) as pool:
        pool.starmap(run_agent, tasks)