import numpy as np
import pandas as pd
from agent import *
from collective_functions import *

from datetime import datetime, timedelta
import torch
import json
import argparse
import re
import ast
import os
import pickle
from IPython.display import clear_output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Agent parameters")
    parser.add_argument("--hyper_params", type=str, help="Agent hyper parameters")
    parser.add_argument("--dataset", type=str, default="df_pc_fake.pkl", help="Name of dataset")
    parser.add_argument("--eps_fixed", action='store_true', help="Is epsilon fixed")
    parser.add_argument("--train", action='store_true', help="Train mode")
    # parser.add_argument("--eps_update", type=int, default=2000, help="Update epsilon every")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--start", type=int, default=1, help="start from num_inter")
    parser.add_argument("--end", type=int, default=159264, help="end after num_inter")
    parser.add_argument("--reward_weights", type=str, help="JSON file with reward weights")
    parser.add_argument("--save_metrics_as", type=str, default="dic_indic_agent", help="save metrics as")
    parser.add_argument("--constraint_factor_veh", type=int, default=1, help="size of available vehicles in Z1. factor 1 is 100%, factor 3 is 33%")
    parser.add_argument("--constraint_factor_ff", type=int, default=1, help="size of available firefighters. factor 1 is 100%, factor 3 is 33%")

    args = parser.parse_args()

    os.chdir('./Data')

    hyper_params = json.load(open(args.hyper_params, "r"))
    
    eps_fixed = args.eps_fixed
    if hyper_params["layer_type"] == "noisy":
        eps_fixed = True
    eps = int(not eps_fixed)
    
    compute = False

    device = torch.device(hyper_params["device"])
    torch.autograd.set_detect_anomaly(True)
    hyper_params["max_train_steps"] = (args.end-args.start) * 5 # (approx. 5 actions by intervention)
    print("max_train_steps", hyper_params["max_train_steps"])
    agent = DQN_Agent(**hyper_params)
    print("Agent initialized", flush=True)

    if not args.train:
        os.chdir('../SVG_model')
        if hyper_params["lr_dec"] == 0:
            agent.optimizer.eval()
        agent.qnetwork_local.load_state_dict(torch.load(args.model_name, weights_only=True))
        agent.qnetwork_local.eval()
        
        print("Eval mode - weights loaded", flush=True)

    else:
        if hyper_params["lr_dec"] == 0:
            agent.optimizer.train()
        agent.qnetwork_local.train()
        print("Train mode", flush=True)

    os.chdir('../')

    ### LOAD ENVIRONMENT VARIABLES ###

    dic_tarif, dic_vehicles, dic_functions, df_skills, dic_roles_skills, dic_roles, planning, dic_inter, \
    dic_ff, dic_indic, dic_indic_old, Z_1, dic_lent, dic_station_distance, df_pc, old_date, date_reference, \
    skills_updated = load_environment_variables(args.reward_weights, args.constraint_factor_veh, args.constraint_factor_ff, args.dataset, args.start, args.end)
    
    suap_sent, ginc_sent, suap_lent, vehicle_out, num_d, score, action_num = 0, 0, 0, 0, 42, 0, 0
    suap_needed, suap_to_return, all_ff_waiting, v_waiting, \
    already_checked, suap_returning, following_depart = False,False,False,False,False,False,False
    suap_to_station, suap_from_station = "", ""
    vehicle_evo, reward_evo, current_ff_inter = [], [], []
    dic_log, dic_back, dic_start_time, dic_veh_typ = {}, {}, {}, {}    

    eps_update = (args.end-args.start) // 21 # approx. 21 iterations to reach 10% of original lr
    d = 1
    print("eps_fixed", eps_fixed, "eps", eps, "eps_update", eps_update, flush=True)

    max_duration = df_pc["Duration"].max()
    action_size = hyper_params["action_size"] # idx role + rl infos

    os.chdir('../SVG_model')

    # for idx, inter in tqdm(df_pc.iloc[:-20].iterrows(), total=len(df_pc.iloc[:-20])):
    for idx, inter in df_pc.iloc[:-20].iterrows():

        # time sin/cos, x, y
    
        num_inter, date, pdd, required_departure, zone, duration, month, day, hour, minute, \
        coord_x, coord_y, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos = inter
        
        dic_ff = update_duration(date, old_date, current_ff_inter, dic_ff)

        # ff_inter = [f for f in dic_ff if dic_ff[f] > 0]
        # if ff_inter:
        #     for f in ff_inter:
        #         print(num_inter, f, dic_ff[f])
    
        if date > date_reference:
            date_reference = date
            skills_updated = update_skills(df_skills, date_reference)
                
        if (suap_sent | ginc_sent) and (num_inter == arrival_num) :  # ARRIVEE DES RENFORTS
            dic_vehicles, suap_sent, suap_lent, suap_returning, \
            dic_ff, planning, dic_back, dic_lent, dic_log, \
            suap_to_return = reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, \
                                                    dic_ff, dic_log, planning, suap_from_station, suap_to_station, suap_sent, \
                                                    suap_returning, suap_lent, suap_to_return, month, day, hour, dic_start_time)
    
        if (required_departure == {0:"RETURN"}):  # RETOUR     
            vehicle_out, dic_vehicles, dic_ff, current_ff_inter, planning, \
            dic_inter = returning(df_pc, dic_inter, num_inter, vehicle_out, dic_vehicles, \
                                  dic_ff, current_ff_inter, planning, month, day, hour)
                     
        else: # INTERVENTION      
            veh_depart = [v[0] for k, v in sorted(required_departure.items())]

            dic_indic['v_required'] += len(required_departure)

            if len(veh_depart) > 5:
                new_veh_depart = veh_depart[5:].copy()
                veh_depart = veh_depart[:5]  
                required_departure = {i + 1: [v] for i, v in enumerate(veh_depart)}
                following_depart = True
                # print("following_depart")

            new_required_departure = {}
            stations = iter(pdd)            
            inter_done = False
            already_checked = False
            suap_to_return = False
            station_lvl = 0
            idx_role = 0
            
            vehicle_evo.append([num_inter, vehicle_out])
    
            while not inter_done: # Tant que l'intervention n'est pas finie   
                
                current_station = next(stations, False) # On va dans la plus proche caserne

                # print(num_inter, required_departure)
                
                if ((not suap_to_return) or (num_d != 99)) and (current_station not in dic_inter[num_inter]): 
                    # Il s'agit d'une intervention et non d'un renfort
                    dic_inter[num_inter][current_station] = {}
                    # print("station created for inter", num_inter, current_station)
                    
                station_lvl += 1
    
                if not current_station: # S'il n'y a plus de plus proche caserne, l'intervention est terminée
                    inter_done = True
                    departure_done = True         
                    dic_indic['v_not_found_in_last_station'] += len(required_departure)
                    # print("v_not_found_in_last_station")
                    
                #-------------------------------------------------------------------------------------------------------------------------------#    
                #- Tentative de punission si v_not_found_in_last_station : ---------------------------------------------------------------------#
                #-------------------------------------------------------------------------------------------------------------------------------#
                    if len(required_departure) > 0:
                        penality = np.log(len(required_departure)) + 1
                        reward -= penality
                        print(f"Véhicules non trouvés à la dernière station. Pénalité: {-penality}")
                #-------------------------------------------------------------------------------------------------------------------------------#    
    
                else: # Sinon on cherche les véhicules requis
                    departure_done = False
                    required_vehicles = iter(sorted(required_departure.items()))

                # print(num_inter, required_departure)
        
                while not departure_done: # Tant que tous les véhicules n'ont pas été envoyés
                    
                    num_d, list_v = next(required_vehicles, (0, [])) # On cherche les véhicules requis dans le train initial
                    # print("start", "num_d", num_d, "list_v", list_v)
                    if station_lvl > 1 and num_d == 1:
                        dic_indic['v1_not_sent_from_1st_station'] += 1
                        # print("v1_not_sent_from_1st_station", station_lvl)
                    
                    # if list_v:
                    #     mandatory, team_max = get_mandatory_max(list_v[0])
                    
                    if not list_v: # S'il n'y a plus de véhicules requis dans le train initial    
                        departure_done = True # On a envoyés tous les véhicules possibles depuis cette caserne 
                        # print("no more v to send from this station")
                        if new_required_departure: # S'il reste des véhicules à faire partir dans le nouveau train  
                            required_departure = new_required_departure
                            required_departure = update_dep(required_departure) # to test
                            veh_depart = [v[0] for k, v in sorted(required_departure.items())]
                            new_required_departure = {}
                            idx_role = 0
                            # print("new_required_departure", veh_depart)

                        else: # S'il ne reste pas de véhicules à faire partir
                            inter_done = True # L'intervention est terminée
                            if following_depart:
                                inter_done = False
                                stations = iter(pdd)
                                station_lvl = 0
                                required_departure = {i + 1: [v] for i, v in enumerate(new_veh_depart)}
                                required_departure = update_dep(required_departure)
                                veh_depart = new_veh_depart
                                new_required_departure = {}
                                idx_role, num_d = 0, 1
                                following_depart = False
                                      
                    else: # S'il y a des véhicules requis dans le train initial  

                        vehicles_to_find = iter(list_v)   
                        vehicle_found = False
                        vehicle_lvl = 1

                        mandatory, team_max = get_mandatory_max(list_v[0])
    
                        if (num_d == 99):
                            current_station = next(stations_suap, False)
                        # print(num_inter, "current_station", current_station, "num_d", num_d, )

                        if not current_station:
                            vehicle_found = True
                        
                        while not vehicle_found: # Tant qu'on a pas trouvé le véhicule requis
    
                            vehicle_to_find = next(vehicles_to_find, False) # On cherche la prochaine fonction à faire partir  
                            
                            if not vehicle_to_find: # S'il n'y a plus de fonction à faire partir
                                
                                idx_role += team_max 
                                vehicle_found = True
                                new_required_departure[num_d] = list_v # Le véhicule requis est ajouté au nouveau train 
                                dic_indic['function_not_found'] += 1
                                # print("function not found")

                            else: # S'il y a une fonction à faire partir depuis cette caserne
                                # On cherche les véhicules disponibles dans la caserne actuelle correspondant à la fonction requise  

                                # print("num_d", num_d, "vehicle to find:", vehicle_to_find)
                                
                                v_mats = dic_vehicles[current_station]["available"].copy()                             
                                li_mat_veh = [v_mat for v_mat in v_mats if vehicle_to_find in dic_functions[v_mat]]
    
                                if li_mat_veh: # Si des véhicules ont la fonction requise   
    
                                    if suap_to_return and (current_station == suap_to_station):
                                        v_mat, v_waiting = suap_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, \
                                                                                   current_station, dic_vehicles)

                                    else: 
                                        v_mat = li_mat_veh[0] 
                                        
                                    if num_d < 99:
                                        veh_depart[num_d-1] = vehicle_to_find

                                    # print(num_inter, "v_mat", v_mat, "num_d", num_d, "veh_depart", veh_depart, "vehicle_to_find", vehicle_to_find)
                                        
                                    dic_indic['ff_skill_lvl'][v_mat] = []
    
                                    mandatory, team_max = get_mandatory_max(vehicle_to_find)
                                    
                                    # On met le véhicule en standby
                                    dic_vehicles[current_station]["available"].remove(v_mat)
                                    dic_vehicles[current_station]["standby"].append(v_mat) 
                                    vehicle_found = True
    
                                    # On cherche les rôles à pourvoir
                                    roles = dic_roles[vehicle_to_find]
                                    # print(roles)
                                    role_number = iter(range(1, len(roles) + 1))
                                    all_roles_found = False
                                    degraded = False
    
                                    while not all_roles_found: # Tant que tous les rôles ne sont pas pourvus
    
                                        num_role = next(role_number, (0))  
    
                                        if num_role > team_max: # Pour limiter les rôles
                                            num_role = 0
                                                           
                                        if not num_role: # S'il n'y a plus de rôles à pourvoir   
                                            # print("no more role to fill")
                                            all_roles_found = True
                                            ff_to_send = planning[current_station][month][day][hour]['standby'].copy()
                                            planning[current_station][month][day][hour]['standby'] = []    
                                            dic_vehicles[current_station]["standby"].remove(v_mat)
    
                                            if suap_needed and (num_d == 99): # départ en renfort
                                                dic_start_time[v_mat] = (month, day, hour)
                                                suap_from_station, dic_vehicles, arrival_num, dic_lent, \
                                                dic_log, new_required_departure, suap_needed, \
                                                suap_sent  = reinforcement_sending(current_station, suap_from_station, v_mat, \
                                                                                   dic_vehicles, dic_station_distance, \
                                                                                   suap_to_station, date, df_pc, idx, \
                                                                                   dic_lent, ff_to_send, dic_log, suap_needed, \
                                                                                   suap_sent, required_departure, \
                                                                                   new_required_departure)

                                                # dic_indic['z1_sent'] += 1
    
                                            elif suap_to_return and v_waiting and all_ff_waiting: # retour de renfort
                                                suap_from_station, suap_to_station, arrival_num, dic_back, dic_log, \
                                                suap_needed, suap_sent, all_ff_waiting, v_waiting, \
                                                suap_returning = reinforcement_returning(suap_to_station, suap_from_station, dic_log, \
                                                                                         v_mat, dic_vehicles, dic_station_distance, date, \
                                                                                         df_pc, idx, dic_back, ff_to_send, suap_needed, \
                                                                                         suap_sent, all_ff_waiting, v_waiting, \
                                                                                         suap_returning)
                                              
                                            elif suap_to_return and v_waiting and not all_ff_waiting: 
                                                # retour impossible, pompiers indisponibles
                                                dic_vehicles[current_station]["available"].append(v_mat)
                                                planning[current_station][month][day][hour]['available'] += ff_to_send
    
                                            else: # départ de véhicule en inter
    
                                                dic_inter[num_inter][current_station][v_mat] = ff_to_send
                                                dic_vehicles[current_station]["inter"].append(v_mat)
                                                
                                                current_ff_inter += ff_to_send
                                                for f in ff_to_send:
                                                    dic_ff[f] = duration
                                                    # print(num_inter, current_station, f, duration)
                                                
                                                vehicle_out += 1

                                                # print(num_inter, "vehicle_out", v_mat, ff_to_send, veh_depart, vehicle_to_find)

                                                dic_indic['v_sent'] += 1
                                                if degraded: 
                                                    dic_indic['v_degraded'] += 1 
                                                else:
                                                    dic_indic['v_sent_full'] += 1 
                                                dic_indic['ff_sent'] += len(ff_to_send)

                                                dic_veh_typ = update_dict(dic_veh_typ, vehicle_to_find) # metrique
    
                                                if not suap_sent and (current_station in Z_1): # s'il n'y a pas de renfort en route
                                                    suap_disp, suap_to_station = get_potential_suap(Z_1, dic_vehicles, dic_functions) 
                                                    
                                                    stations_suap, suap_needed, suap_to_return, new_required_departure, suap_to_station, \
                                                    dic_ff = suap_management(suap_disp, suap_needed, suap_to_return, suap_lent, \
                                                                             suap_to_station, new_required_departure, dic_station_distance,\
                                                                             num_inter, dic_lent, dic_vehicles, already_checked, dic_ff)
                                                    if suap_needed:
                                                        dic_indic['z1_sent'] += 1
    
                                        else: # S'il y a un rôle à pourvoir

                                            info_avail = get_neighborhood_availability(pdd, current_station, num_d, dic_vehicles, \
                                                                                       planning, month, day, hour, 5)

    
                                            ff_mats = planning[current_station][month][day][hour]["planned"].copy()

                                            ff_existing, all_ff_waiting = get_existing_ff(suap_lent, current_station, Z_1, dic_lent, \
                                                                                          ff_mats, dic_ff, v_mat, suap_to_return, \
                                                                                          v_waiting, suap_to_station, all_ff_waiting)  

                                            if not all_ff_waiting:
                                                for f in ff_existing:
                                                    
                                                    if f not in planning[current_station][month][day][hour]["available"] and \
                                                    dic_ff[f] == 0 and \
                                                    not (suap_lent or (current_station in Z_1)):
                                                        print(planning[current_station][month][day][hour])
                                                        print(ff_existing)
                                                        print(dic_lent)
                                                        raise Exception("ERROR", num_inter, current_station, month, day, hour, f, dic_ff[f])
                                                        
                                            
                                            if len(ff_existing) > 70:
                                                print(num_inter, current_station)

                                            ff_array = gen_ff_array(df_skills, skills_updated, ff_existing)

                                            

                                            state = gen_state(veh_depart, idx_role, ff_array, ff_existing, \
                                                              dic_roles, dic_roles_skills, dic_ff, df_skills, \
                                                              coord_x, coord_y, month_sin, month_cos, day_sin, \
                                                              day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size)

                                            # if all_ff_waiting:
                                            #     print(state[0:6, np.r_[0:5, -3:0]])

                                            # if num_inter % 500 == 0:
                                                # print("start", state[0, :15])
                                                # print("end", state[2:17, -3])
    
                                            if compute and args.train:
                                                agent.step(old_state, action, reward, state, inter_done)
                                                
     
                                            action, skill_lvl = agent.act(state, all_ff_waiting, eps)

                                            dic_indic['skill_lvl'] += skill_lvl * 8 # denormalize
    
                                            dic_indic, all_roles_found, vehicle_found, planning, dic_vehicles, dic_ff, idx_role, \
                                            degraded = step(action, idx_role, ff_existing, all_ff_waiting, current_station, Z_1, dic_lent, \
                                                            v_mat, dic_ff, suap_lent, planning, month, day, hour, num_inter, \
                                                            new_required_departure, num_d, list_v, num_role, mandatory, degraded, team_max, \
                                                            all_roles_found, vehicle_found, dic_vehicles, dic_indic, \
                                                            skill_lvl, station_lvl)

                                            # REWARD

                                            reward = compute_reward(dic_indic, dic_indic_old, num_d, dic_tarif)
                                            dic_indic_old = dic_indic.copy()
                                            
                                            score += reward
                                            # print(f"\r", num_inter, vehicle_out, reward, end="", flush=True)
                                            # print(num_inter, vehicle_out, "reward :", reward)
                                            action_num += 1
                                            reward_evo.append([action_num, reward])
                                            old_state = state
                                            compute = True

                                # else: # si aucun véhicule n'a la fonction requise
                                #     print(num_inter, veh_depart, vehicle_to_find, "no vehicule found")

        old_date = date       
                                    
        if num_inter % 100 == 0 and required_departure == {0:"RETURN"}:
            # print(f"\r", num_inter, vehicle_out, "score:", score, end="", flush=True)
            rwd_mean = np.mean([row[1] for row in reward_evo[-100:]])
            lr = agent.optimizer.param_groups[0]['lr']
            print(f"{num_inter} v_out: {vehicle_out} rwd_mean: {rwd_mean:.2f} per ff: {score/dic_indic['ff_sent']:.2f} act: {action_num} per act.: {(score/action_num):.5f} v_not_found_ls: {dic_indic['v_not_found_in_last_station']} deg: {dic_indic['v_degraded']} z1_sent: {dic_indic['z1_sent']} eps: {eps:.2f} lr: {lr:.5f}", flush=True)
            
        # elif num_inter % 100 == 0 and required_departure == {0:"RETURN"} and eps_fixed:
        #     rwd_mean = np.mean([row[1] for row in reward_evo[-100:]])
        #     print(f"{num_inter} v_out: {vehicle_out} rwd_mean: {rwd_mean:.2f} score: {score} act: {action_num} ratio:, {-(score/action_num):.5f} v_not_found_ls: {dic_indic['v_not_found_in_last_station']} cancel_dep: {dic_indic['cancelled']} eps:, {eps:.2f}", flush=True)
            

        if not eps_fixed and num_inter % eps_update == 0 and required_departure == {0:"RETURN"}:
            eps = eps * 0.99**d
            eps = max(0.05, eps)
            d+=1
        # if num_inter % 99 == 0:
        #     clear_output(wait=True)

        if num_inter % 10000 == 0 and args.train and required_departure == {0:"RETURN"}:   
            torch.save(agent.qnetwork_local.state_dict(), args.model_name)
            print(num_inter, "Agent saved as", args.model_name, flush=True)

    if args.train:

        os.chdir('../SVG_model')   
        if hyper_params["lr_dec"] == 0:
            agent.optimizer.eval() # eval avant save pour schedule free
        torch.save(agent.qnetwork_local.state_dict(), args.model_name)
        print("Agent saved as", args.model_name, flush=True)
        print()
        os.chdir('../Plots')
        np.save(args.save_metrics_as + "_train_reward_" + args.dataset[6:10] +".npy", reward_evo)

    else:

        os.chdir('../Plots')
        
        np.save(args.save_metrics_as + "_vehicle_" + args.dataset[6:10] +".npy", vehicle_evo)
        np.save(args.save_metrics_as + "_reward_" + args.dataset[6:10] +".npy", reward_evo)    
        pickle.dump(dic_indic, open(args.save_metrics_as + ".pkl", "wb"))
    
        os.chdir('../')

        print("Metrics saved", flush=True)
        print()


