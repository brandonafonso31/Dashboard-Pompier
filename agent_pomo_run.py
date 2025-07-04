import subprocess
import multiprocessing
from IPython.display import clear_output

def run_agent(agent_id, args_template):
    """Fonction pour exécuter un agent individuel"""
    
    #print("\nBien arrivé dans la fonction run_agent !")
    
    args = args_template.copy()
    args["--model_name"] = f"{args['--model_name']}_agent_{agent_id}"
    args["--save_metrics_as"] = f"{args['--save_metrics_as']}_agent_{agent_id}"
    
    cmd = [
        "python3", "agent_run.py",
        "--model_name", args["--model_name"],
        "--hyper_params", args["--hyper_params"],
        "--reward_weights", args["--reward_weights"],
        "--dataset", args["--dataset"],
        "--start", args["--start"],
        "--end", args["--end"],
        "--constraint_factor_veh", args["--constraint_factor_veh"],
        "--constraint_factor_ff", args["--constraint_factor_ff"],
        "--save_metrics_as", args["--save_metrics_as"],
        "--train" if args["--train"] else ""
    ]
    #print(f"La commande de l'agent {args["--model_name"]} est {cmd}\n")
    
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
    
    #print("\nBien arrivé dans agent_pomo_run !\n")
    
    m = 'z1_pomo_sent_'
    suffix = "r100_cf3"
    
    args_template = {
        "--model_name": f"agent_{m}_{suffix}",
        "--hyper_params": "hyper_params.json",
        "--reward_weights": f"rw_z1_sent_{suffix}.json",
        "--dataset": "df_pc_fake_1y.pkl",
        "--start": "1",
        "--end": "53088",
        "--constraint_factor_veh": "3",
        "--constraint_factor_ff": "1",
        "--save_metrics_as": f"agent_pomo_metrics_{m}_{suffix}",
        "--train": True
    }
    
    num_agents = 5  # Nombre d'agents POMO
    
    print(f"Nombre d'agent = {num_agents}. on démarre le multi process ...")
    # Lancer les agents en parallèle
    with multiprocessing.Pool(processes=num_agents) as pool:
        pool.starmap(run_agent, [(i, args_template) for i in range(num_agents)])