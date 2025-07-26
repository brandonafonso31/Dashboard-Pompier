import json

metrics = ['v_degraded', 
           'v1_not_sent_from_s1', 
           'v3_not_sent_from_s3', 
           'v_not_found_in_last_station', 
           'z1_VSAV_sent', 
           'rupture_ff']

shared_path="./Data/metrics.json"
    
with open(shared_path, "w") as f:
    json.dump({i: f"{metrics[i]}" for i in range(len(metrics))}, f)

print(f"c'est fait ;)")