import json,time,os

def get_metrics(path):
    metrics_path = "Data/metrics.json"
    if "SVG_model" in path or "Reward_weights" in path:
        metrics_path = "../" + metrics_path
    
    metrics = json.load(open(metrics_path, "r"))
    return [i for i in metrics.values()]

def get_current_elected():
    path = "../Data/shared_state.json"
    print(os.getcwd())
    try:
        with open(path, "r") as f:
            data = json.load(f)
        metric = data["elected"]
        return get_metrics().index(metric)
    except Exception as e:
        print("Error reading shared_state.json:", e)
        time.sleep(0.5)
    return -1 