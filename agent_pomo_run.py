import subprocess
import multiprocessing
from IPython.display import clear_output

def run_agent(agent_id):
    """Fonction pour exécuter un agent individuel"""
    
    #print("\nBien arrivé dans la fonction run_agent !")
    m = ['z1_sent', 'v_not_found_in_last_station']# , 'ff_sent', 'ff_skill_mean', 'v_degraded']
    # , 'function_not_found', 'skill_lvl', 'function_cancelled', 'cancelled', 'v_sent_full', 'v_sent', 'v1_not_sent_from_1st_station']
    suffix = "r100_cf3"
    
    cmd = [
        "python3", "agent_run.py",
        "--model_name", f"pomo_agent_{m[agent_id]}",
        "--hyper_params", "hyper_params.json",
        "--reward_weights", f"rw_{m[agent_id]}_{suffix}.json",
        "--dataset", "df_pc_fake_1y.pkl",
        "--start", "1",
        "--end", "53088",
        "--constraint_factor_veh", "3",
        "--constraint_factor_ff", "1",
        "--save_metrics_as", f"agent_pomo_metrics_{m[agent_id]}_{suffix}",
        "--train"
    ]
    print(f"\nLa commande de l'agent pomo_agent_{m[agent_id]} est {cmd}\n")
    
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
    num_agents = 2 # Nombre d'agents POMO
    print(f"Nombre d'agent = {num_agents}. on démarre le multi process ...")
    
    # Lancer les agents en parallèle
    with multiprocessing.Pool(processes=num_agents) as pool:
        pool.starmap(run_agent, [(i,) for i in range(num_agents)])