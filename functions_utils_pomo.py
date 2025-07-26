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