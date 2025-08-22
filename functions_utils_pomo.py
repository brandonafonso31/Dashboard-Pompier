import json,time,os

def get_metrics(cur_path):
    metrics_path = "Data/metrics.json"
    current_dir = os.path.basename(cur_path)
    if current_dir in ["SVG_model", "Reward_weights"]:
        metrics_path = "../" + metrics_path
    
    metrics = json.load(open(metrics_path, "r"))
    return [i for i in metrics.values()]

def get_current_elected(cur_path):
    path = "Data/shared_state.json"
    current_dir = os.path.basename(cur_path)
    if current_dir in ["SVG_model", "Reward_weights"]:
        path = "../" + path
    try:
        with open(path, "r") as f:
            data = json.load(f)
        metric = data["elected"]
        return get_metrics(os.getcwd()).index(metric)
    except Exception as e:
        print("Error reading shared_state.json:", e)
        time.sleep(0.5)
    return -1 


metric_list = ["v_required", "v_sent","v_sent_full","v_degraded","cancelled","function_not_found",
       "v1_not_sent_from_s1","v3_not_sent_from_s3","v_not_found_in_last_station",
       "ff_required","ff_sent", "rupture_ff","skill_lvl"]

metrics = {}
i = 0
for key in metric_list:
    metrics[str(i)] = key
    i+=1
with open(f"Data/metrics.json",'w') as f:
    json.dump(metrics, f,indent=4)