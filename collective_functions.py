import numpy as np
import re
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import json
import pickle
import torch

def apply_logic(potential_actions, potential_skills, is_best):

    if is_best:
        action = potential_actions[potential_skills.index(min(potential_skills))]

    else:
        action = random.choice(potential_actions)
        
    skill_lvl = potential_skills[potential_actions.index(action)]

    return action, skill_lvl

def get_potential_actions(state, all_ff_waiting):

    # 1st row: rl infos
    # 2nd row: idx role

    potential_actions = [79]
    skill_lvl = 0
    potential_skills = [0]
    cond_met = np.array([])
    # state = state.cpu().numpy()
    col_index = np.argmax(state[1, :] == 1) # current role
    column_values = state[2:, col_index] # ff available for a given role

    if not all_ff_waiting: # standard case
        selection = column_values[column_values > 0] # ff having the skill
        if selection.size > 0: # any ff ?        
            cond_met = np.where( (column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1)) )[0] # ff having any skill lvl > 0
            potential_skills = column_values[(column_values > 0) & (np.all(state[2:, -3:] == 0, axis=1))].tolist()
    else: # all ff waiting
        selection = column_values # all ff to avoid the case of a ff losing his skills
        if selection.size > 0: # any ff ?
            cond_met = np.where( (state[2:, -2] == 1) )[0]                                               
            cond_met = np.array([cond_met[0]]) # first ff because all ff waiting follows an order
            
    if cond_met.size > 0:
        potential_actions = cond_met.tolist()
    else:
        potential_skills = [0]

    if potential_actions == [79]:
        assert np.all(state[-1] == 0), f"FF is on slot 79"

    if all_ff_waiting:
        assert potential_actions == [0], f"not action 0 and all_ff_waiting"

        
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

    idx = pdd.index(station)
    following = pdd[idx+1:idx+1+n_following]

    return following    

def get_neighborhood_availability(pdd, station, num_d, dic_vehicles, planning, month, day, hour, n_following):
    
    if num_d < 79:
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

def load_environment_variables(constraint_factor_veh, constraint_factor_ff, dataset, start, end):

    os.chdir('./Data_environment')

    df_stations = pd.read_pickle("df_stations.pkl")

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
                    'rupture_ff':0, #lack of ff
                    'function_not_found':0,
                    'v1_not_sent_from_s1':0,
                    'v3_not_sent_from_s3':0,
                    'v_not_found_in_last_station':0,
                    'ff_required':0,
                    'ff_sent':0,
                    'z1_VSAV_sent': 0,
                    'z1_FPT_sent': 0,
                    'z1_EPA_sent': 0,
                     'VSAV_needed':0,
                     'FPT_needed':0,
                     'EPA_needed':0,
                     'VSAV_disp':0,
                     'FPT_disp':0,
                     'EPA_disp':0,
                    'skill_lvl':0
                    } 
    dic_indic_old = dic_indic.copy()
    Z_1 = ['TOULOUSE - LOUGNON', 'TOULOUSE - VION']
    Z_2 = ['ST JORY', 'ROUFFIAC', 'RAMONVILLE - BUCHENS', 'COLOMIERS', 'MURET - MASSAT']
    Z_3 = ['AUTERIVE', 'ST LYS', 'GRENADE', 'FRONTON', 'VERFEIL', 'CARAMAN']
    Z_4 = [s for s in df_stations["Nom"] if s not in Z_1 + Z_2 + Z_3]
    print("Z_4", Z_4)
    dic_lent = {k:{} for k in Z_1} # station to, v_mat, ff_mat

    
    dic_station_distance = {ville_z1: trier_villes_par_distance(df_stations, ville_z1, Z_2 + Z_3) for ville_z1 in Z_1}

    idx_start = df_pc[(df_pc["num_inter"]==start) & (df_pc["departure"]!={0: 'RETURN'})].index[0]
    idx_end = df_pc[(df_pc["num_inter"]==end) & (df_pc["departure"]=={0: 'RETURN'})].index[0]
    df_pc = df_pc[idx_start:idx_end+1]

    print("df start-end", idx_start, idx_end)
    
    old_date = df_pc.iloc[0, 1]
    date_reference = df_pc.iloc[0, 1]
    skills_updated = update_skills(df_skills, date_reference)

    return dic_vehicles, dic_functions, df_skills, dic_roles_skills, dic_roles, planning, \
    dic_inter, dic_ff, dic_indic, dic_indic_old, Z_1, Z_4, dic_lent, dic_station_distance, df_pc, \
    old_date, date_reference, skills_updated


def gen_state(veh_depart, idx_role, ff_array, ff_existing, dic_roles, dic_roles_skills, dic_ff, df_skills, \
             coord_x, coord_y, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size):


    # ff skills
    state = np.hstack(([get_roles_for_ff(veh, ff_array, dic_roles, dic_roles_skills) for veh in veh_depart])).astype(float)

    state /= 8 # normalization, 8 skill lvls

    # filler row    
    # if state.shape[0] > action_size:
    #     print(state.shape[0])
        
    filler = np.zeros((action_size-state.shape[0], state.shape[1])) # max 74 de base + 6 ff lent + 1 role to fill
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

def compute_reward(dic_indic, dic_indic_old, num_d, dic_tarif):
    reward = 0

    if num_d < 79:

        dic_delta = {key:(dic_indic[key] - dic_indic_old[key]) for key in dic_indic if key not in ['VSAV_disp', 'FPT_disp', 'EPA_disp']}

        for m in dic_delta:

            reward += dic_delta[m] * dic_tarif[m]

        
        if dic_indic['VSAV_disp'] < 2:
            reward += dic_tarif['VSAV_disp']
    
        if dic_indic['FPT_disp'] < 2:
            reward += dic_tarif['FPT_disp']
    
        if dic_indic['EPA_disp'] < 1:
            reward += dic_tarif['EPA_disp']


    return reward


def step(action, idx_role, ff_existing, all_ff_waiting, current_station, Z_1, dic_lent, \
    v_mat, dic_ff, VSAV_lent, FPT_lent, EPA_lent, planning, month, day, hour, num_inter, new_required_departure, num_d, \
    list_v, num_role, mandatory, degraded, team_max, all_roles_found, \
    vehicle_found, dic_vehicles, dic_indic, skill_lvl, station_lvl):

    if action < 79:
        idx_role += 1

        ff_mat = ff_existing[action]
        dic_indic['skill_lvl'] += skill_lvl * 8  # was normalized in state
        
        if all_ff_waiting and (current_station in Z_1): # pompiers à rapatrier
            dic_lent[current_station][v_mat].remove(ff_mat)
            dic_ff[ff_mat] = -2 # was already in standby -1

        else:
            dic_ff[ff_mat] = -1

            if not (VSAV_lent or FPT_lent or EPA_lent or (current_station in Z_1)):
                planning[current_station][month][day][hour]['available'].remove(ff_mat)
          
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
            dic_indic['rupture_ff'] += 1
            # dic_indic['ff_skill_lvl'][v_mat] = []

    return dic_indic, dic_lent, all_roles_found, vehicle_found, planning, dic_vehicles, dic_ff, idx_role, degraded

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


def get_potential_veh(Z_1, dic_vehicles, dic_functions, v_type):

    # 2 VSAV en Z1
    # 2 FPT + 1 EPC en Z1
    v_disp_tl = len([item for item in dic_vehicles['TOULOUSE - LOUGNON']["available"] \
                     if any(v_type in func for func in dic_functions[item])]) # or func.startswith(v_type)
    v_disp_tv = len([item for item in dic_vehicles['TOULOUSE - VION']["available"] \
                     if any(v_type in func for func in dic_functions[item])]) #  or func.startswith(v_type)
    v_to_station = Z_1[np.argmin([v_disp_tl, v_disp_tv])]
    return (v_disp_tl+v_disp_tv), v_to_station

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
    # if vehicle == "EP":
    #     vehicle = "EPA"
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
        dic_vehicles[station] = {"available" : group['IU Materiel'].tolist(), 
                                 "standby" : [], 
                                 "inter" : [], 
                                 "VSAV_sent":[], 
                                 "FPT_sent":[], 
                                 "EPA_sent":[]}

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

def reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, dic_ff, dic_log, planning, v_from_station, \
    v_to_station, v_sent, v_returning, v_lent, v_to_return, month, day, hour, dic_start_time, v_type):
    
    veh_mat = dic_vehicles[v_from_station][v_type + "_sent"][0]
    dic_vehicles[v_to_station]["available"].append(veh_mat)
    dic_vehicles[v_from_station][v_type + "_sent"] = [] 
    v_sent -= 1 

    start_month, start_day, start_hour = dic_start_time[veh_mat]
    
    if v_returning: 
        v_lent -= 1
        v_returning = False
        for f in dic_back[veh_mat]:
            dic_ff[f] = 0
            if (f in planning[v_to_station][month][day][hour]["planned"]) and \
            (f not in planning[v_to_station][month][day][hour]["available"]):
                planning[v_to_station][month][day][hour]["available"].append(f)

            if (f not in planning[v_to_station][start_month][start_day][start_hour]["available"]):
                planning[v_to_station][start_month][start_day][start_hour]["available"].append(f)

        
        # print(num_inter, v_type, veh_mat, dic_back[veh_mat], "sent back from", v_from_station, "to", v_to_station, "has arrived")
        del dic_back[veh_mat]  
        del dic_lent[v_from_station][veh_mat]   
        del dic_log[veh_mat]

    else: 
        v_lent += 1
        for f in dic_lent[v_to_station][veh_mat]:
            dic_ff[f] = 0

        # print(num_inter, v_type, veh_mat, dic_lent[v_to_station][veh_mat], "sent from", v_from_station, "to", v_to_station, "has arrived")

    v_to_return = False

    return dic_vehicles, v_sent, v_lent, v_returning, dic_ff, planning, dic_back, dic_lent, dic_log, v_to_return

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

def veh_management(v_disp, v_needed, v_to_return, v_lent, v_to_station, \
    new_required_departure, dic_station_distance, num_inter, dic_lent, \
    dic_vehicles, dic_functions, dic_ff, threshold, v_type, new_num_d):

    stations_v = iter([])
    v_mat = 0
    # print(v_disp, v_disp, threshold, threshold, )
    if (v_disp < threshold):
        v_needed = True
        v_to_return = False
        new_required_departure[new_num_d] = [v_type]
        stations_v = iter(dic_station_distance[v_to_station])                                                    
        # print(num_inter, v_type, "needed for station", v_to_station)
        
    elif (v_disp > threshold) and v_lent: # libération du véhicule 
        v_needed = False                 
        flag = 0
        
        # Mise en attente des pompiers envoyés: 
        v_to_station = [s for s, vff in dic_lent.items() if vff and any(v_type in dic_functions[v] for v in vff)][0]
        # print("v_to_station", v_to_station)
        stations_v = iter([v_to_station])
        # print(num_inter, v_type, "not needed anymore for station", v_to_station)      
        for veh_mat, ff_mats in dic_lent[v_to_station].items(): 
            if veh_mat in dic_vehicles[v_to_station]["available"] and v_type in dic_functions[veh_mat]:
                # print(num_inter, v_type, veh_mat, "to return is available")
                v_to_return = True
                v_mat = veh_mat
                new_required_departure[new_num_d] = [v_type]
                for ff in ff_mats:
                    if dic_ff[ff] == 0:
                        dic_ff[ff] = -1
                        # print("ff", ff, "to return is available")
                if all(dic_ff[ff] == -1 for ff in ff_mats):
                    # print(num_inter, "all ff waiting for", v_type, veh_mat, ff_mats)
                    flag = 1
                # else:
                #     print("not all ff waiting")

            if flag:
                break

    return stations_v, v_needed, v_to_return, new_required_departure, v_to_station, dic_ff, v_mat
            
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

def reinforcement_returning(num_inter, v_to_station, v_from_station, dic_log, v_mat, dic_vehicles, \
    dic_station_distance, date, df_pc, idx, dic_back, ff_to_send, v_needed, \
    v_sent, all_ff_waiting, v_waiting, v_returning, v_type):
    
    v_from_station = v_to_station
    v_to_station = dic_log[v_mat]
    dic_vehicles[v_from_station][v_type+"_sent"].append(v_mat)
    arrival_time = date + timedelta(minutes = dic_station_distance[v_from_station][v_to_station] + 20)
    arrival_num = df_pc.loc[(df_pc.index >= idx) & (df_pc["date"] >= arrival_time), "num_inter"].iloc[0]
    # print(num_inter, v_type, v_mat, ff_to_send, "sent back from", v_from_station, "will arrive at", arrival_num, "to", v_to_station)
    dic_back[v_mat] = ff_to_send
    dic_log[v_mat] = v_from_station
    # print("reinforcement_returning", dic_log)
    v_needed = False
    v_sent += 1
    all_ff_waiting = False
    v_waiting = False
    v_to_return = False
    v_returning = True

    return v_from_station, v_to_station, arrival_num, dic_back, dic_log, v_needed, v_sent, all_ff_waiting, v_waiting, v_to_return, v_returning

def reinforcement_sending(num_inter, current_station, v_from_station, v_mat, dic_vehicles, \
    dic_station_distance, v_to_station, date, df_pc, idx, dic_lent, \
    ff_to_send, dic_log, v_needed, v_sent, \
    required_departure, new_required_departure, num_d, v_type):
    
    v_from_station = current_station
    dic_vehicles[v_from_station][v_type+"_sent"].append(v_mat)
    arrival_time = date + timedelta(minutes = dic_station_distance[v_to_station][v_from_station] + 20)
    arrival_num = df_pc.loc[(df_pc.index >= idx) & (df_pc["date"] >= arrival_time), "num_inter"].iloc[0]
    # print(num_inter, v_type, v_mat, ff_to_send, "sent from", v_from_station, "will arrive at", arrival_num, "to", v_to_station)
    dic_lent[v_to_station][v_mat] = ff_to_send.copy() # mise à disposition des pompiers
    dic_log[v_mat] = v_from_station
    # print("reinforcement_sending", dic_log)

    if num_d in new_required_departure:
        del new_required_departure[num_d]

    v_needed = False
    v_sent += 1

    return v_from_station, dic_vehicles, arrival_num, dic_lent, dic_log, new_required_departure, v_needed, v_sent 

def v_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, current_station, dic_vehicles, v_type, v_mat_to_return):

    v_to_return = [v for v in li_mat_veh if (v in dic_log)]
    if v_to_return and v_mat_to_return != 0:
        # v_mat = v_to_return[0]
        v_mat = v_mat_to_return
        v_waiting = True
        # print("managing", vehicle_to_find, "to return", v_mat, "found in station", current_station) 
    else:         
        v_mat = li_mat_veh[0]
        v_waiting = False
        # print("managing", v_type, "to return NOT found in station", current_station, "now looking for", vehicle_to_find, v_mat)
    return v_mat, v_waiting

def adding_lent_ff(VSAV_lent, FPT_lent, EPA_lent, current_station, Z_1, dic_lent, ff_mats, dic_ff):

    if (VSAV_lent or FPT_lent or EPA_lent) and (current_station in Z_1): # ajout des renforts à la station de Z1

        ff_lent = [f for v in dic_lent[current_station] if v in dic_lent[current_station] \
                   for f in dic_lent[current_station][v]]
        ff_mats += ff_lent
        ff_mats = list(set(ff_mats))

    return [f for f in ff_mats if f in dic_ff].copy() # Pour éviter les pompiers manquants


def are_all_ff_waiting(ff_existing, current_station, dic_lent, dic_ff, v_mat):
        
    ff_waiting = [f for f in ff_existing if (dic_ff[f] == -1)] # already put in standby -1
    return all(f in ff_waiting for f in dic_lent[current_station][v_mat])   

    # else:
    #     lent_ff = [ff for v_mat in dic_lent.values() for ff_lent in v_mat.values() for ff in ff_lent]
    #     ff_not_lent = [num for num in ff_existing if num not in lent_ff]
    #     ff_existing = [f for f in ff_not_lent if dic_ff[f] > -1].copy()





# def get_required_roles(dic_roles, required_departure):
#     return [val for dico in [dict(sorted(dic_roles[vehicle_to_find].items())) \
#                                  for vehicle_to_find in [v[0] for k, v in required_departure.items()]] for val in dico.values()]

def gen_ff_array(df_skills, skills_updated, ff_existing):
    return skills_updated[[df_skills.index.get_loc(matricule) for matricule in ff_existing]]

    
def update_duration(date, old_date, current_ff_inter, dic_ff):

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
        if k < 99:
            new_d[new_k] = v
            new_k += 1
        else:
            new_d[k] = v
    
    return new_d