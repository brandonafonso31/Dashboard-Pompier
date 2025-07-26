import json,time

def get_metrics():
    metrics_path = "./Data/metrics.json"
    metrics = json.load(open(metrics_path, "r"))
    return [i for i in metrics.values()]

def get_current_elected():
    path = "./Data/shared_state.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
        metric = data["elected"]
        return ['v_degraded', 'v1_not_sent_from_s1'].index(metric)
    except Exception as e:
        print("Error reading shared_state.json:", e)
        time.sleep(0.5)
    return -1 