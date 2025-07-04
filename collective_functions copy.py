import numpy as np
import re
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import json
import pickle
import torch

def get_potential_actions(state, all_ff_waiting):

    # 1st row: rl infos
    # 2nd row: idx role

    potential_actions = [99]
    skill_lvl = 0
    potential_skills = [0]
    # state = state.cpu().numpy()
    col_index = np.argmax(state[1, :] == 1) # current role
    column_values = state[2:, col_index] # ff available for a given role
    selection = column_values[column_values > 0] # ff having the skill
    if selection.size > 0: # any ff ?
        if not all_ff_waiting: # standard case
            cond_met = np.where( (column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1)) )[0] # ff having any skill lvl > 0
            potential_skills = column_values[(column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1))].tolist()
        else: # all ff waiting
            # print("selection", selection)
            skill_lvl = np.min(selection)
            cond_met = np.where( (column_values >= skill_lvl) & (state[2:, -2] == 1) )[0] # ff having the best skill lvl
            cond_met = np.array([cond_met[0]]) # first ff because all ff waiting follows an order
            potential_skills = [skill_lvl]
            # print("size", cond_met.size, "skill", skill_lvl)
            
        if cond_met.size > 0:
            potential_actions = cond_met.tolist()
        else:
            potential_skills = [0]
        
    return potential_actions, potential_skills

def get_v_availability(dic_vehicles, station):
    v_available = len(dic_vehicles[station]["available"])
    v_all = sum([len(dic_vehicles[station][x]) for x in dic_vehicles[station]])
    return v_available/v_all

def get_ff_availability(planning, station, month, day, hour):
    ff_available = len(planning[station][month][day][hour]['available'])
    ff_all = len(planning[station][month][day][hour]['planned'])
    if ff_all == 0:
        return 0
    else:
        return ff_available/ff_all

def get_neighborhood(pdd, station, num_d, n_following =5):
    try:
        idx = pdd.index(station)
        following = pdd[idx+1:idx+1+n_following]
    except:
        print(num_d, pdd)
    return following    

def get_neighborhood_availability(pdd, station, num_d, dic_vehicles, planning, month, day, hour, n_following):
    
    if num_d != 99:
        info_avail = []
        neighborhood = get_neighborhood(pdd, station, num_d, n_following)
        for s in neighborhood:
            v_avail = get_v_availability(dic_vehicles, s)
            ff_avail = get_ff_availability(planning, s, month, day, hour)
            info_avail += [v_avail, ff_avail]
        for _ in range(n_following-len(neighborhood)):
            info_avail += [0, 0]
    else:
        info_avail = [0] * n_following * 2
    return info_avail

def load_environment_variables(reward_weights, constraint_factor_veh, constraint_factor_ff, dataset, start, end):

    os.chdir('./Reward_weights')

    dic_tarif = json.load(open(reward_weights))
    print("Reward weights", dic_tarif)

    os.chdir('../Data_environment')

    df_v = pd.read_pickle("df_v.pkl")
    dic_vehicles, dic_functions = create_dic_vehicles(df_v)
    dic_vehicles = purge_dic_v(dic_vehicles)

    print("constraint factor veh is ", constraint_factor_veh)
    
    list_of_mats = dic_vehicles["TOULOUSE - VION"]["available"]
    dic_vehicles["TOULOUSE - VION"]["available"] = constrain_veh(list_of_mats, constraint_factor_veh)
    
    list_of_mats = dic_vehicles["TOULOUSE - LOUGNON"]["available"]
    dic_vehicles["TOULOUSE - LOUGNON"]["available"] = constrain_veh(list_of_mats, constraint_factor_veh)

    df_skills = pd.read_pickle("df_skills.pkl")

    df_skills = df_skills.sample(len(df_skills)//constraint_factor_ff)
    print("constraint factor ff is ", constraint_factor_ff, "Number of ff:", len(df_skills))
    
    df_roles = pd.read_pickle("df_roles.pkl")
    dic_roles_skills = generate_dic_roles_skills(df_roles, df_skills)

    df_vehicles_history = pd.read_pickle("df_vehicles_history.pkl")
    dic_roles = create_dic_roles(df_vehicles_history)
    
    with open("planning.pkl", "rb") as file:
        planning = pickle.load(file)

    df_pc = pd.read_pickle(dataset) #, sep = ';', parse_dates=["date"], converters={"PDD": ast.literal_eval, "departure": ast.literal_eval})

    dic_inter = {i:{} for i in range(1, int(len(df_pc)/2)+1)} # num_inter:station:mat_v:mat_ff
    
    dic_ff = {ff:0 for ff in df_skills.index}
    dic_indic = {'v_required': 0,
                    'v_sent': 0,
                    'v_sent_full':0,
                    'v_degraded':0,
                    'cancelled':0, #cancel departure
                    'function_not_found':0,
                    'v1_not_sent_from_1st_station':0,
                    'v_not_found_in_last_station':0,
                    'ff_required':0,
                    'ff_sent':0,
                    'ff_skill_lvl':{},
                    'ff_skill_sum':0,
                    'ff_skill_mean':0,
                    'z1_sent': 0,
                    'skill_lvl':0
                    } 
    dic_indic_old = dic_indic.copy()
    Z_1 = ['TOULOUSE - LOUGNON', 'TOULOUSE - VION']
    Z_2 = ['ST JORY', 'ROUFFIAC', 'RAMONVILLE - BUCHENS', 'COLOMIERS', 'MURET - MASSAT']
    Z_3 = ['AUTERIVE', 'ST LYS', 'GRENADE', 'FRONTON', 'VERFEIL', 'CARAMAN']
    dic_lent = {k:{} for k in Z_1} # station to, v_mat, ff_mat

    df_stations = pd.read_pickle("df_stations.pkl")
    dic_station_distance = {ville_z1: trier_villes_par_distance(df_stations, ville_z1, Z_2 + Z_3) for ville_z1 in Z_1}

    idx_start = df_pc[(df_pc["num_inter"]==start) & (df_pc["departure"]!={0: 'RETURN'})].index[0]
    idx_end = df_pc[(df_pc["num_inter"]==end) & (df_pc["departure"]=={0: 'RETURN'})].index[0]
    df_pc = df_pc[idx_start:idx_end+1]

    print("df start-end", idx_start, idx_end, len(df_pc)/2)
    
    old_date = df_pc.iloc[0, 1]
    date_reference = df_pc.iloc[0, 1]
    skills_updated = update_skills(df_skills, date_reference)

    return dic_tarif, dic_vehicles, dic_functions, df_skills, dic_roles_skills, dic_roles, planning, \
    dic_inter, dic_ff, dic_indic, dic_indic_old, Z_1, dic_lent, dic_station_distance, df_pc, \
    old_date, date_reference, skills_updated



# def gen_state(veh_depart, idx_role, ff_array, ff_existing, dic_roles, dic_roles_skills, dic_ff, df_skills):
    
#     nb_rows = 70

#     state = np.hstack(([get_roles_for_ff(veh, ff_array, dic_roles, dic_roles_skills) for veh in veh_depart]))

#     # current role to fill
#     current_role = [0]*(state.shape[1])
#     current_role[idx_role] = 1  
#     state = np.insert(state, 0, np.array(current_role), axis=0)

#     # filler row
#     filler = np.zeros((nb_rows-state.shape[0], state.shape[1])) # max 64 de base + 6 ff lent
#     state = np.append(state, filler, axis=0)

#     # filler col
#     filler = np.zeros((state.shape[0], 39 - state.shape[1]))
#     state = np.concatenate((state, filler), axis=1)

#     # resp time
#     availability = np.array([0]+[dic_ff[f] for f in df_skills.loc[ff_existing, :].index] + [0]*(nb_rows-len(ff_existing)-1)).reshape(-1, 1)  
#     state = np.concatenate((state, availability), axis=1)

#     return state

def gen_state(veh_depart, idx_role, ff_array, ff_existing, dic_roles, dic_roles_skills, dic_ff, df_skills, \
             coord_x, coord_y, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size):


    # ff skills
    state = np.hstack(([get_roles_for_ff(veh, ff_array, dic_roles, dic_roles_skills) for veh in veh_depart])).astype(float)

    state /= 8 # normalization, 8 skill lvls

    # filler row    
    if state.shape[0] > action_size:
        print(state.shape[0])
        
    filler = np.zeros((action_size-state.shape[0], state.shape[1])) # max 64 de base + 6 ff lent + 1 role to fill
    state = np.vstack((state, filler))

    # filler col
    filler = np.zeros((state.shape[0], 37 - state.shape[1]))
    state = np.concatenate((state, filler), axis=1)

    # current role to fill
    current_role = [0]*37
    current_role[idx_role] = 1  
    # state = np.insert(state, 0, np.array(current_role), axis=0)
    state = np.vstack((current_role, state))
    

    # resp time

    resp_time = np.array([dic_ff[f] for f in df_skills.loc[ff_existing, :].index])
    resp_time_norm = np.where(resp_time < 0, 0.0, resp_time/max_duration) # normalization
    mask_minus1 = (resp_time == -1) 
    mask_minus2 = (resp_time == -2)
    resp_time_all = np.stack([resp_time_norm, mask_minus1, mask_minus2], axis=1)

    zero_row = np.zeros((1, resp_time_all.shape[1])) # for current role to fill
    resp_time_all = np.vstack((zero_row, resp_time_all))
    zero_rows = np.zeros(((action_size-len(ff_existing)), resp_time_all.shape[1]))
    availability = np.vstack((resp_time_all, zero_rows))

    
    # availability = np.array([0] + resp_time_norm + [0]*(action_size-len(ff_existing)-1)).reshape(-1, 1)  
    state = np.hstack((state, availability))

    # rl_infos + position + time

    rl_infos = np.array(info_avail + [coord_x, coord_y, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos] + [0]*22)
    # print("rl_infos", rl_infos)
    state = np.vstack((rl_infos, state))

    # print("state shape final", state.shape, flush=True)

    return state

def constrain_veh(list_of_mats, factor=3, seed=42):
    random.seed(seed)
    random.shuffle(list_of_mats)
    size_subset = len(list_of_mats) // factor
    return list_of_mats[:size_subset]

def get_start_hour(df, num_inter):

    idx = df[df["num_inter"] == num_inter].index[0]
    date = df["date"].loc[idx]

    return date.month, date.day, date.hour

def reduce_dic(dic_indic):
    dic_sum = defaultdict(int)
    dic_sum['ff_skill_lvl'] = []
    
    for dic in dic_indic.values():
        for v, qty in dic.items():
            if v == 'ff_skill_lvl':
    
                for v_mat, skill_lvl in qty.items():
                    dic_sum[v] += skill_lvl
            else:
                dic_sum[v] += qty
    
    dic_sum['ff_skill_lvl'] = np.mean(dic_sum['ff_skill_lvl'])

    return dic_sum

def compute_reward(dic_indic, dic_indic_old, num_d, dic_tarif):
    reward = 0

    if num_d != 99:
        dic_reward = {key:(dic_indic[key] - dic_indic_old[key]) for key in dic_indic if key != "ff_skill_lvl"}

        for k in dic_tarif.keys():

            if k != "skill_lvl":

                reward += dic_reward[k] * dic_tarif[k]


            else:

                if dic_reward[k] != 0:

                    reward += dic_tarif[k] / dic_reward[k]

    return reward 


def step(action, idx_role, ff_existing, all_ff_waiting, current_station, Z_1, dic_lent, \
    v_mat, dic_ff, suap_lent, planning, month, day, hour, num_inter, new_required_departure, num_d, \
    list_v, num_role, mandatory, degraded, team_max, all_roles_found, \
    vehicle_found, dic_vehicles, dic_indic, skill_lvl, station_lvl):

    if action < 99:
        idx_role += 1

        ff_mat = ff_existing[action]
        # if dic_ff[ff_mat] != 0:
        #     print('ALERT', dic_ff[ff_mat], all_ff_waiting)
        dic_indic['ff_skill_lvl'][v_mat].append(skill_lvl)
        # print("before remove", ff_existing, action, ff_mat)
        
        if all_ff_waiting and (current_station in Z_1): # pompiers à rapatrier
            dic_lent[current_station][v_mat].remove(ff_mat)
            dic_ff[ff_mat] = -2 # was already in standby -1
            # print(ff_mat, "was put in standby -2", "idx role was", idx_role -1)

        else:
            stock_duration = dic_ff[ff_mat]
            dic_ff[ff_mat] = -1
            dic_indic['ff_skill_lvl'][v_mat].append(skill_lvl)

            if not (suap_lent or (current_station in Z_1)):
                try:
                    planning[current_station][month][day][hour]['available'].remove(ff_mat)
                except:
                    print(num_inter, current_station, month, day, hour, ff_mat)
                    print(action, ff_existing)
                    print(dic_lent)
                    print(planning[current_station][month][day][hour])
                    raise Exception(all_ff_waiting, ff_mat, stock_duration)
          
        planning[current_station][month][day][hour]['standby'].append(ff_mat)

    else: # aucun pompier n'a les compétences requises
        new_required_departure[num_d] = list_v # Le véhicule requis est ajouté au nouveau train
        # si le rôle est obligatoire ou que ce n'est pas le 1er véhicule
        
        if (num_role > mandatory) and (num_d == 1): # Si le rôle est facultatif et que c'est 
        # le 1er véhicule 
            if not degraded: 
                degraded = True
                # print(v_mat, "degraded")
            idx_role += 1

        else: # Si le rôle n'est pas facultatif ou que ce n'est pas le 1er véhicule    

            all_roles_found, vehicle_found, planning, dic_vehicles, \
            dic_ff = cancel_departure(all_roles_found, vehicle_found, planning, current_station, \
                                      month, day, hour, dic_vehicles, dic_ff, v_mat)  
            idx_role += team_max - (num_role -1) # le num_role est itéré une fois de plus au-delà du max
            dic_indic['cancelled'] += 1
            dic_indic['ff_skill_lvl'][v_mat] = []

    return dic_indic, all_roles_found, vehicle_found, planning, dic_vehicles, dic_ff, idx_role, degraded

def get_mandatory_max(v):
    mandatory = team_max = 2
    
    if "VSAV" in v:
        mandatory, team_max = 2, 3
    elif "FPT" in v :
        mandatory, team_max = 4, 6
    elif "EP" in v:
        mandatory, team_max = 2, 3
    elif "VSR" in v:
        mandatory, team_max = 2, 3
    elif "PCC" in v:
        mandatory, team_max = 1, 1
    # elif any(w in v for w in ["VID","VBAL","VTUTP", "VGD"]):
    #     mandatory, team_max = 2, 2
    elif v == "CCF":
        mandatory, team_max = 3, 4
    # elif "CCFL" in v:
    #     mandatory, team_max = 2, 3
       

    return mandatory, team_max

def create_dic_roles(df_vehicles_history):

    df_vehicles_history["Fonction"] = df_vehicles_history["Fonction"].fillna("")
    dic_replace = {"IMP3_CUNITE":"IMP_CU", "IMP2_SAUV":"IMP_SAUV", "COND_ENG_NAUT":"COND_BMS_EB"}
    df_vehicles_history["Fonction Occupee"] = df_vehicles_history["Fonction Occupee"].replace(dic_replace)
    dic_replace = {"XCOMPL":"COMPL"}
    df_vehicles_history["Fonction"] = df_vehicles_history["Fonction"].replace(dic_replace)
    
    dic_roles = {}
    
    for idx, row in df_vehicles_history.iterrows():
    
        tm = row["Type Matériel"]
        t = row["Type"]
        f = row["Fonction"]
        ofo = row["Ordre Fonction Occupee"]
        fo = row["Fonction Occupee"]
        if fo == "CHEF DE GROUPE":
            fo = "*CDG*"
        
        if (tm != ""):
            if tm not in dic_roles:
                dic_roles[tm] = {ofo:fo}
            elif ofo not in dic_roles[tm]:
                dic_roles[tm][ofo] = fo
        if (t != ""):
            if t not in dic_roles:
                dic_roles[t] = {ofo:fo}
            elif ofo not in dic_roles[t]:
                dic_roles[t][ofo] = fo
        if (f != ""):
            if f not in dic_roles:
                dic_roles[f] = {ofo:fo}
            elif ofo not in dic_roles[f]:
                dic_roles[f][ofo] = fo

    dic_roles = {
    fonction: {num_role: role
                  for num_role, role in valeurs.items() if num_role <= 6}
    for fonction, valeurs in dic_roles.items()
}

    dic_roles = {k: v for k, v in dic_roles.items() if not (isinstance(k, float) and np.isnan(k))}

    for veh, nb_ro in dic_roles.items():
        mandatory, team_max = get_mandatory_max(veh)
        dic_roles[veh] = {k: v for k, v in dic_roles[veh].items() if k <= team_max}

    return dic_roles

def get_potential_suap(Z_1, dic_vehicles, dic_functions):

# 2 VSAV en Z1
    suap_disp_tl = len([item for item in dic_vehicles['TOULOUSE - LOUGNON']["available"] if "VSAV" in dic_functions[item]])
    suap_disp_tv = len([item for item in dic_vehicles['TOULOUSE - VION']["available"] if "VSAV" in dic_functions[item]])
    suap_to_station = Z_1[np.argmin([suap_disp_tl, suap_disp_tv])]
    return (suap_disp_tl+suap_disp_tv), suap_to_station

def get_potential_veh(Z_1, dic_vehicles, dic_functions, v_type):

# 2 VSAV en Z1
    v_disp_tl = len([item for item in dic_vehicles['TOULOUSE - LOUGNON']["available"] if v_type in dic_functions[item]])
    v_disp_tv = len([item for item in dic_vehicles['TOULOUSE - VION']["available"] if v_type in dic_functions[item]])
    v_to_station = Z_1[np.argmin([v_disp_tl, v_disp_tv])]
    return (v_disp_tl+v_disp_tv), v_to_station

def get_potential_ginc(Z_1, dic_vehicles, dic_functions):

    # 2 FPT + 1 EPC en Z1
    fpt_disp = len([item for s in Z_1 for item in dic_vehicles[s]["available"] if any("FPT" in func for func in dic_functions[item])])
    ep_disp = len([item for s in Z_1 for item in dic_vehicles[s]["available"] if any("EP" in func for func in dic_functions[item])])

    return fpt_disp, ep_disp

def update_dict(dic, k):
    if k in dic:
        dic[k] += 1
    else:
        dic[k] = 1
    return dic

def get_role_from_skills(required_skills, ff_array):

    matches_minus_one = (required_skills == -1)[:, np.newaxis, :]  # Reshape pour broadcast
    matches_one = (required_skills == 1)[:, np.newaxis, :]
    matches_zero_or_any = (required_skills == 0)[:, np.newaxis, :]
    
    conditions_met = ((matches_minus_one & (ff_array == 0)[np.newaxis, :, :]) | 
                      (matches_one & (ff_array == 1)[np.newaxis, :, :]) | 
                      matches_zero_or_any)
    
    conditions_met = np.all(conditions_met, axis=2)
    
    first_valid_index = np.argmin(conditions_met, axis=0) +1
    
    first_valid_index[~np.any(conditions_met, axis=0)] = 0
    
    return first_valid_index

def get_roles_for_ff(vehicle, ff_array, dic_roles, dic_roles_skills):
    required_roles = dic_roles[vehicle]

    required_roles = [required_roles[k] for k in sorted(required_roles.keys())]
    required_roles = [role if role in dic_roles_skills else 'EQ_ENG_SAP' for role in required_roles]

    return np.column_stack([get_role_from_skills(dic_roles_skills[role], ff_array).reshape(-1, 1) for role in required_roles])

def distance_euclidienne(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def trier_villes_par_distance(df, ville_z1, villes_z2):

    x1, y1 = df.loc[df['Nom'] == ville_z1, ['Coordonnée X', 'Coordonnée Y']].values[0]
    

    distances = []
    for ville_z2 in villes_z2:
        x2, y2 = df.loc[df['Nom'] == ville_z2, ['Coordonnée X', 'Coordonnée Y']].values[0]
        dist = distance_euclidienne(x1, y1, x2, y2)
        dist /= 1000
        distances.append((ville_z2, int(dist)))
    
    # Trier les villes de z2 par distance croissante
    distances.sort(key=lambda x: x[1])  # Trie par la distance (le deuxième élément)
    
    # Retourner les villes triées et leurs distances
    return {ville:dist for ville, dist in distances}

def get_skill_array(plus, minus, df_skills, zeros=134):
    idx_plus = df_skills.columns.get_level_values(0).unique().get_indexer(plus)
    idx_minus = df_skills.columns.get_level_values(0).unique().get_indexer(minus)
    array = np.zeros(zeros, dtype=int)
    array[idx_plus] = 1
    array[idx_minus] = -1
    return array

def extract_skills(s):

    plus = [re.findall(r"[\w]+", s)[0]]
    plus += re.findall(r"\+\s*[\w]+", s)
    plus = [w.replace("+ ", "") for w in plus]
    minus = re.findall(r"-\s*[\w]+", s)
    minus = [w.replace("- ", "") for w in minus]
    grade = re.findall(r"s*GRADE\([A-Z]+\)", s)
    
    if grade:
        grade = grade[0][4:-1]
        sup = re.findall(r">", s)        
        if sup:            
            sup = sup[0]
        else:
            inf = re.findall(r"<", s)[0]    
    else:
        sup = ""

    return  pd.Series([plus, minus, grade, sup])

def generate_dic_roles_skills(df_roles, df_skills):
    
    df_roles["Competences"] = df_roles["Competences"].fillna("")
    
    df_roles[["Plus", "Minus", "Grade", "Sup"]] = df_roles["Competences"].apply(extract_skills)
    df_roles["Required_roles"] = df_roles.apply(lambda row: get_skill_array(row['Plus'], row['Minus'], df_skills), axis=1)
    dic_roles_skills = {}
    for fonction, group in df_roles.groupby('Fonction'):
        sorted_group = group.sort_values(by='Ordre')
        dic_roles_skills[fonction] = np.vstack(sorted_group['Required_roles'].tolist())
    return dic_roles_skills

def create_dic_vehicles(df_v):

    dic_vehicles = {}
    dic_functions = {}
    for station, group in df_v.groupby('Nom du Centre'):
        dic_vehicles[station] = {"available" : group['IU Materiel'].tolist(), "standby" : [], "inter" : [], "suap_sent":[]}

    dic_functions = dict(zip(df_v['IU Materiel'], df_v['Fonction materiel']))

    return dic_vehicles, dic_functions

def purge_dic_v(dic):
    v = []
    new_dic = {}
    
    for s, d in dic.items():
        new_dic[s] = {}
        for st, li_v in d.items():
            new_dic[s][st] = []
            for e in li_v:
                if e not in v:
                    v.append(e)
                    new_dic[s][st].append(e)

    return new_dic

def reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, dic_ff, dic_log, planning, suap_from_station, \
    suap_to_station, suap_sent, suap_returning, suap_lent, suap_to_return, month, day, hour, dic_start_time):
    
    # print("num_inter", num_inter, "RENFORT", dic_vehicles[suap_from_station]["suap_sent"])
    veh_mat = dic_vehicles[suap_from_station]["suap_sent"][0]
    dic_vehicles[suap_to_station]["available"].append(veh_mat)
    dic_vehicles[suap_from_station]["suap_sent"] = [] 
    suap_sent -= 1 

    start_month, start_day, start_hour = dic_start_time[veh_mat]
    
    if suap_returning: 
        # print("suap returning before", [f for f in dic_ff if dic_ff[f]==-1])
        suap_lent -= 1
        suap_returning = False
        for f in dic_back[veh_mat]:
            dic_ff[f] = 0
            # print("f", f, dic_ff[f])
            if (f in planning[suap_to_station][month][day][hour]["planned"]) and \
            (f not in planning[suap_to_station][month][day][hour]["available"]):
                planning[suap_to_station][month][day][hour]["available"].append(f)
                # print("ff", f, "available again", "sb:", len([f for f in dic_ff if dic_ff[f] ==-1]))

            if (f not in planning[suap_to_station][start_month][start_day][start_hour]["available"]):
                planning[suap_to_station][start_month][start_day][start_hour]["available"].append(f)
                # print("f", f, "has been added to start time on suap returning")
        
        # print("suap", veh_mat, dic_back[veh_mat], "sent back from", suap_from_station, "to", suap_to_station, "has arrived at", num_inter)
        del dic_back[veh_mat]  
        del dic_lent[suap_from_station][veh_mat]   
        del dic_log[veh_mat]

    else: 
        suap_lent += 1
        # print("suap sent before", [f for f in dic_ff if dic_ff[f]==-1])
        for f in dic_lent[suap_to_station][veh_mat]:
            dic_ff[f] = 0

        # print("suap", veh_mat, dic_lent[suap_to_station][veh_mat], "sent from", suap_from_station, "to", suap_to_station, "has arrived at", num_inter)

    suap_to_return = False

    return dic_vehicles, suap_sent, suap_lent, suap_returning, dic_ff, planning, dic_back, dic_lent, dic_log, suap_to_return

def returning(df_pc, dic_inter, num_inter, vehicle_out, dic_vehicles, dic_ff, current_ff_inter, planning, month, day, hour):

    start_month, start_day, start_hour = get_start_hour(df_pc, num_inter)

    # print("returning before", [f for f in dic_ff if dic_ff[f]==-1])
    for station, mats in dic_inter[num_inter].items(): # num_inter:station:v_mat:ff_mat          
        for veh_mat, ff_mats in mats.items():
            vehicle_out -= 1
            
            dic_vehicles[station]["inter"].remove(veh_mat)
            dic_vehicles[station]["available"].append(veh_mat)

            for f in ff_mats:
                dic_ff[f] = 0
                current_ff_inter.remove(f)
                
                if (f in planning[station][month][day][hour]["planned"]) and \
                (f not in planning[station][month][day][hour]["available"]):
                    planning[station][month][day][hour]["available"].append(f)
                    # print("f", f, "has been added in available", month, day, hour)

                if (f not in planning[station][start_month][start_day][start_hour]["available"]):
                    planning[station][start_month][start_day][start_hour]["available"].append(f)
                    # print("f", f, "has been added to start time")
         
            # print(num_inter, "vehicle in", station, veh_mat, ff_mats, vehicle_out)
                 
            dic_inter[num_inter][station][veh_mat] = []

    # print("returning after", [f for f in dic_ff if dic_ff[f]==-1])
    return vehicle_out, dic_vehicles, dic_ff, current_ff_inter, planning, dic_inter


def suap_management(suap_disp, suap_needed, suap_to_return, suap_lent, suap_to_station, \
    new_required_departure, dic_station_distance, num_inter, dic_lent, \
    dic_vehicles, already_checked, dic_ff):

    stations_suap = None

    if (suap_disp <= 2):  # demande de VSAV
        suap_needed = True
        suap_to_return = False
        new_required_departure[99] = ["VSAV"]
        # print("suap management new_required_departure", new_required_departure)
        stations_suap = iter(dic_station_distance[suap_to_station])                                                    
        # print(num_inter, "suap needed for station", suap_to_station)
        
    elif (suap_disp > 2) and suap_lent: # libération du VSAV  
        suap_needed = False                                                 
        # Mise en attente des pompiers envoyés:
        suap_to_station = [s for s, vff in dic_lent.items() if vff][0]
        stations_suap = iter([suap_to_station])
        # print(num_inter, "suap not needed anymore for station", suap_to_station)      
        for veh_mat, ff_mats in dic_lent[suap_to_station].items(): 
            if veh_mat in dic_vehicles[suap_to_station]["available"]: # and not already_checked
                suap_to_return = True
                new_required_departure[99] = ["VSAV"]
                # if num_inter==to_check:
                # print("suap available", veh_mat, new_required_departure, "at", suap_to_station)
                # print("suap available before:", [f for f in dic_ff if dic_ff[f]==-1])
                for ff in ff_mats:
                    if dic_ff[ff] == 0:
                        dic_ff[ff] = -1
                # print("suap available after:", [f for f in dic_ff if dic_ff[f]==-1])
                break
            # else:
            #     suap_to_return = False
            #     if num_inter==to_check:
            #         print("already checked - return in standby")

    return stations_suap, suap_needed, suap_to_return, new_required_departure, suap_to_station, dic_ff
            
def cancel_departure(all_roles_found, vehicle_found, planning, current_station, \
    month, day, hour, dic_vehicles, dic_ff, v_mat):
    
    all_roles_found = True
    vehicle_found = True
    # Les pompiers sortent du standby
    ff_mats = planning[current_station][month][day][hour]['standby'].copy()
    planning[current_station][month][day][hour]['standby'] = []
    # Le véhicule en standby est à nouveau disponible
    dic_vehicles[current_station]["standby"].remove(v_mat)
    dic_vehicles[current_station]["available"].append(v_mat) 
    # les pompiers sont à nouveau disponibles
    # print(v_mat, "cancel departure")
    for f in ff_mats:
        # if dic_ff[f] == -2:
        #     dic_ff[f] = -1
        #     # print(f, "is in standby again")
        # else:  
        dic_ff[f] = 0
        # print(f, "is available again") 
        if (f in planning[current_station][month][day][hour]["planned"]) and \
        (f not in planning[current_station][month][day][hour]["available"]):
            planning[current_station][month][day][hour]["available"].append(f)
        # print("f", f, "has been added in available", month, day, hour)
        

    # print("cancel departure after:", [f for f in dic_ff if dic_ff[f]<0])

    return all_roles_found, vehicle_found, planning, dic_vehicles, dic_ff

def reinforcement_returning(suap_to_station, suap_from_station, dic_log, v_mat, dic_vehicles, \
    dic_station_distance, date, df_pc, idx, dic_back, ff_to_send, suap_needed, \
    suap_sent, all_ff_waiting, v_waiting, suap_returning):
    
    suap_from_station = suap_to_station
    suap_to_station = dic_log[v_mat]
    dic_vehicles[suap_from_station]["suap_sent"].append(v_mat)
    arrival_time = date + timedelta(minutes = dic_station_distance[suap_from_station][suap_to_station] + 20)
    arrival_num = df_pc.loc[(df_pc.index >= idx) & (df_pc["date"] >= arrival_time), "num_inter"].iloc[0]
    # print("suap", v_mat, ff_to_send, "sent back from", suap_from_station, "will arrive at", arrival_num, "to", suap_to_station)
    dic_back[v_mat] = ff_to_send
    dic_log[v_mat] = suap_from_station
    # print("reinforcement_returning", dic_log)
    suap_needed = False
    suap_sent += 1
    all_ff_waiting = False
    v_waiting = False
    suap_returning = True

    return suap_from_station, suap_to_station, arrival_num, dic_back, dic_log, suap_needed, suap_sent, all_ff_waiting, v_waiting, suap_returning

def reinforcement_sending(current_station, suap_from_station, v_mat, dic_vehicles, \
    dic_station_distance, suap_to_station, date, df_pc, idx, dic_lent, \
    ff_to_send, dic_log, suap_needed, suap_sent, \
    required_departure, new_required_departure):
    
    suap_from_station = current_station
    dic_vehicles[suap_from_station]["suap_sent"].append(v_mat)
    arrival_time = date + timedelta(minutes = dic_station_distance[suap_to_station][suap_from_station] + 20)
    arrival_num = df_pc.loc[(df_pc.index >= idx) & (df_pc["date"] >= arrival_time), "num_inter"].iloc[0]
    # print("suap", v_mat, ff_to_send, "sent from", suap_from_station, "will arrive at", arrival_num, "to", suap_to_station)
    dic_lent[suap_to_station][v_mat] = ff_to_send.copy() # mise à disposition des pompiers
    dic_log[v_mat] = suap_from_station
    # print("reinforcement_sending", dic_log)

    if 99 in new_required_departure:
        del new_required_departure[99]

    suap_needed = False
    suap_sent += 1

    return suap_from_station, dic_vehicles, arrival_num, dic_lent, dic_log, new_required_departure, suap_needed, suap_sent 

def suap_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, current_station, dic_vehicles):

    # print("suap_to_return managing", current_station, dic_log, li_mat_veh)
    v_to_return = [v for v in li_mat_veh if (v in dic_log)]
    # v_to_return_now = [v for v in v_to_return if (dic_log[v] == suap_from_station)]
    if v_to_return:
        v_mat = v_to_return[0]
        v_waiting = True
        # print("suap to return", vehicle_to_find, v_mat, "found in station", current_station) 
    else:         
        v_mat = li_mat_veh[0]
        v_waiting = False
        # print("suap to return NOT found in station", current_station, "now looking for", vehicle_to_find, v_mat)
    return v_mat, v_waiting

def get_existing_ff(suap_lent, current_station, Z_1, dic_lent, ff_mats, dic_ff, v_mat, \
    suap_to_return, v_waiting, suap_to_station, all_ff_waiting):

    if suap_lent and (current_station in Z_1): # ajout des renforts à la station de Z1

        ff_lent = [f for v in dic_lent[current_station] if v in dic_lent[current_station] \
                   for f in dic_lent[current_station][v]]
        ff_mats += ff_lent
        ff_mats = list(set(ff_mats))

    ff_existing = [f for f in ff_mats if f in dic_ff].copy() # Pour éviter les pompiers manquants

    # print("ff e avec renforts", ff_existing)
    
    if suap_to_return and v_waiting and (current_station==suap_to_station):        
        ff_waiting = [f for f in ff_existing if (dic_ff[f] == -1)] # already put in standby -1
        all_ff_waiting = all(f in ff_waiting for f in dic_lent[current_station][v_mat])        
        if all_ff_waiting:
            # print("all ff waiting", [dic_ff[f] for f in dic_lent[current_station][v_mat]])
            ff_existing = dic_lent[current_station][v_mat]
            # print("ff_existing", ff_existing)
        else:
            # print("not all ff waiting")
            # ff_existing = [f for f in ff_existing if dic_ff[f] == 0].copy()
            ff_existing = [f for f in ff_existing if dic_ff[f] > -1].copy()
    else:
        # existing_ff = {ff for v_mat in dic_lent.values() for ff_existing in v_mat.values() for ff in ff_existing} # BUG
        lent_ff = [ff for v_mat in dic_lent.values() for ff_lent in v_mat.values() for ff in ff_lent]
        ff_not_lent = [num for num in ff_existing if num not in lent_ff]
        # ff_existing = [f for f in ff_not_lent if dic_ff[f] == 0].copy()
        ff_existing = [f for f in ff_not_lent if dic_ff[f] > -1].copy()

    return ff_existing, all_ff_waiting

# def get_required_roles(dic_roles, required_departure):
#     return [val for dico in [dict(sorted(dic_roles[vehicle_to_find].items())) \
#                                  for vehicle_to_find in [v[0] for k, v in required_departure.items()]] for val in dico.values()]

def gen_ff_array(df_skills, skills_updated, ff_existing):
    return skills_updated[[df_skills.index.get_loc(matricule) for matricule in ff_existing]]

# def update_duration(date, old_date, current_ff_inter, dic_ff):
#     elapsed_time = (date - old_date).total_seconds() / 60
#     for f in current_ff_inter:
#         dic_ff[f] -= int(elapsed_time)
#     return dic_ff
    
def update_duration(date, old_date, current_ff_inter, dic_ff):
    # if date > old_date:
        # elapsed_time = (date - old_date).total_seconds() / 60
    # else: # 31 dec. into 1 jan.
    #     last_time = datetime(2019, 1, 1, 0, 0)
    #     first_time = datetime(2018, 1, 1, 0, 0)
    #     t1 = (last_time - old_date).total_seconds() / 60
    #     t2 = (date - first_time).total_seconds() / 60
    #     elapsed_time = (t1+t2)
    #     print("elpased_time", old_date, date, elapsed_time)
    elapsed_time = (date - old_date).total_seconds() / 60    
    for f in current_ff_inter:
        dic_ff[f] -= int(elapsed_time)

    return dic_ff

def update_skills(df_skills, date_reference):
    condition_list = []
    for col in df_skills.columns.get_level_values(0).unique():
        deb_col = (col, 'Début')
        fin_col = (col, 'Fin')    
        condition = (df_skills[deb_col] <= date_reference) & (df_skills[fin_col] >= date_reference)
        condition_list.append(condition.rename(col))
        
    conditions = pd.concat(condition_list, axis=1)
    return np.where(conditions.fillna(False), 1, 0)

def update_dep(required_departure):
    new_d = {}
    new_k = 1
    for k, v in sorted(required_departure.items()):
        if k != 99:
            new_d[new_k] = v
            new_k +=1
        else:
            new_d[k] = v
    
    return new_d