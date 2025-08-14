import pandas as pd
import numpy as np
import os
import re
import geopandas as gpd
import random
from collections import defaultdict
from shapely.geometry import Point
from datetime import datetime, timedelta
from sklearn.neighbors import KDTree
import pickle
import argparse

def reorg_dates(df):
    df['departure_sort'] = df['departure'].apply(lambda x: 0 if x == {0: 'RETURN'} else 1)
    return df.sort_values(by=['date', 'departure_sort']).drop(columns='departure_sort')

def generate_stations(df_stations):
    df_stations["Type"] = df_stations["Type"].fillna("")
    df_stations_u = df_stations[df_stations["Type"].str.startswith('CS') 
    & ~df_stations["Nom"].str.startswith(('X','Z')) 
    & (df_stations["Etat"] == "Disponible") 
    & (df_stations["Identifiant unique du centre"].str.startswith('B'))]
    df_stations_u = df_stations_u[['Type', 'Nom', 'Coordonnée X', 'Coordonnée Y']].reset_index(drop=True)
    df_stations_u.loc[df_stations_u["Nom"] == "VION", "Nom"] = "TOULOUSE - VION"
    df_stations_u.loc[df_stations_u["Nom"] == "LOUGNON", "Nom"] = "TOULOUSE - LOUGNON"
    df_stations_u.loc[df_stations_u["Nom"] == "BUCHENS", "Nom"] = "RAMONVILLE - BUCHENS"
    df_stations_u.loc[df_stations_u["Nom"] == "MURET", "Nom"] = "MURET - MASSAT"
    df_stations_u.loc[df_stations_u["Nom"] == "ST LYS", "Nom"]
    df_stations_u["Coordonnée X"] = df_stations_u["Coordonnée X"].str.replace(',', '.', regex=False)
    df_stations_u["Coordonnée Y"] = df_stations_u["Coordonnée Y"].str.replace(',', '.', regex=False)
    df_stations_u["Coordonnée X"] = pd.to_numeric(df_stations_u["Coordonnée X"], errors='coerce')
    df_stations_u["Coordonnée Y"] = pd.to_numeric(df_stations_u["Coordonnée Y"], errors='coerce')
    return df_stations_u

def generate_vehicles(df_vehicles, df_stations_u):
    df_veh_u = df_vehicles[df_vehicles["Nom du Centre"].isin(df_stations_u["Nom"])]
    return df_veh_u[["Nom du Centre", 
                      "IU Materiel", 
                      "Type materiel", 
                      "Fonction materiel"]].groupby(["Nom du Centre", 
                                                     "Type materiel", 
                                                     "IU Materiel"]).agg(lambda x: list(set(x))).reset_index()


def generate_firefighters(df_firefighters):
    df_firefighters = df_firefighters.rename(columns={'Compétence - Centre': 'Centre', 
                                  'Compétence - Nom': 'Compétence', 
                                  'Compétence - Date et heure de début' : 'Début',
                                  'Compétence -Date et heure de fin' : "Fin"})
    df_firefighters['Début'] = df_firefighters['Début'].apply(lambda x : x[:10])
    df_firefighters['Début'] = pd.to_datetime(df_firefighters['Début'].str.split().str[0], format='%d/%m/%Y')
    df_firefighters['Début'] = df_firefighters.groupby(['Matricule', "Compétence"])['Début'].transform('min')
    end_fallback_date = "01/01/2100"
    df_firefighters['Fin'] = df_firefighters['Fin'].fillna(end_fallback_date)
    df_firefighters['Fin'] = pd.to_datetime(df_firefighters['Fin'].str.split().str[0], format='%d/%m/%Y')
    df_firefighters['Fin'] = df_firefighters.groupby(['Matricule', "Compétence"])['Fin'].transform('max')
    df_firefighters = df_firefighters.drop_duplicates()
    df_firefighters = df_firefighters.drop_duplicates(subset=['Matricule', 'Compétence'], keep='first')
    
    df_comp = df_firefighters.pivot_table(index=['Matricule'], columns='Compétence', values=['Début', 'Fin'], aggfunc=lambda x: x)
    df_comp = df_comp.swaplevel(0, 1, axis=1).sort_index(axis=1)
    df_comp[('TOUTES', 'Début')] = df_comp[[col for col in df_comp.columns if "Début" in col]].min(axis=1)
    df_comp[('TOUTES', 'Fin')] = np.datetime64('2100-01-01T00:00:00')
    df_comp = df_comp.apply(pd.to_datetime)
    return df_comp, df_firefighters


def get_number_v(status):
    global dic_vehicles
    ss = dic_vehicles.keys()
    return sum(len(dic_vehicles[s][status]) for s in ss)

def get_stations(x, y, df_pdd, stations_u):
    point = Point([x, y])
    pdd = df_pdd.loc[:, df_pdd.columns.str.startswith('cis')].iloc[np.where(df_pdd.geometry.contains(point))].values[0]
    return [x for x in pdd if x in stations_u]

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
    
    distances.sort(key=lambda x: x[1])
    return {ville:dist for ville, dist in distances}

def get_zone(pdd, Z_1, Z_2, Z_3):
    if pdd[0] in Z_1:
        return "Z_1"
    elif pdd[0] in Z_2:
        return "Z_2"
    elif pdd[0] in Z_3:
        return "Z_3"
    else:
        return "Z_4"


def precompute_pdd(df_pdd, df_sample, stations_u):
    colonnes_a_garder = [col for col in df_pdd.columns if col.startswith('cis')] + ['geometry']
    df_pdd = df_pdd[colonnes_a_garder]
    
    df_sample["PDD"] = df_sample.apply(lambda row: get_stations(row['Coord X'], 
                                                                row['Coord Y'], 
                                                                df_pdd, 
                                                                stations_u),
                                       axis=1)
    return df_sample

def precompute_zone(df_stations, df_sample, Z_1, Z_2, Z_3):
    pdd_z_1 = {ville_z1: trier_villes_par_distance(df_stations, ville_z1, Z_2 + Z_3) for ville_z1 in Z_1}
    df_sample["zone"] = df_sample["PDD"].apply(get_zone, args=(Z_1, Z_2, Z_3))
    return df_sample

def precompute_incident(df_rank_incident, df_sample):

    dict_result = df_rank_incident.set_index('rank')['sin'].to_dict()
    # df_sample['Incident'] += 1 #for tabddpm, min incident was 0 instead of 1
    df_sample['incident_name'] = df_sample['Incident'].map(dict_result)
    return df_sample

def precompute_area_type(df_xy, df_lieu, df_nom_commune, df_commune, df_secteur, df_sample):

    df_xy = df_xy.dropna(subset=["Coordonnées X", "Coordonnées Y"])
    data = df_xy[["Coordonnées X", "Coordonnées Y"]].to_numpy()
    tree = KDTree(data, leaf_size=40)
    
    df_lieu["EST_SITUE_SUR_NUMERO_LIEU"] = df_lieu["EST_SITUE_SUR_NUMERO_LIEU"].fillna(0)
    # df_lieu = df_lieu.iloc[:, :4]
    
    df_sample['area_type'] = df_sample.apply(lambda row: get_area_type(row['Coord X'], 
                                                                       row['Coord Y'], 
                                                                       df_xy, df_lieu, df_nom_commune, df_commune, df_secteur,
                                                                       data, 
                                                                       tree), 
                                             axis=1)    

    return df_sample

def precompute_prob_dict(df_inter_clean):

    nested_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for _, row in df_inter_clean.iterrows():
        incident = row["incident_name"]
        area = row["area_type"]
        list_of_strings = tuple(row["real_func"])
        nested_dict[incident][area][list_of_strings] += 1
    
    nested_dict = {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in nested_dict.items()}
    
    prob_dict = defaultdict(lambda: defaultdict(dict))
    
    for incident, areas in nested_dict.items():
        for area, string_counts in areas.items():
            total = sum(string_counts.values())
            prob_dict[incident][area] = {strings: count / total for strings, count in string_counts.items()}
    
    prob_dict = {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in prob_dict.items()}

    return prob_dict


def prob_departure(area_type, inc_name, prob_dict):

    if area_type in prob_dict[inc_name]:
        choices = list(prob_dict[inc_name][area_type].keys())  
        weights = list(prob_dict[inc_name][area_type].values()) 
        dep_tuple = random.choices(choices, weights=weights, k=1)[0]
        departure = {i+1: [val] for i, val in enumerate(dep_tuple)}
    elif "*" in prob_dict[inc_name]:
        choices = list(prob_dict[inc_name]["*"].keys())  
        weights = list(prob_dict[inc_name]["*"].values()) 
        dep_tuple = random.choices(choices, weights=weights, k=1)[0]
        departure = {i+1: [val] for i, val in enumerate(dep_tuple)}

    return departure
        


def precompute_prob_departure(df_sample, prob_dict):

    df_sample['departure'] = df_sample.apply(lambda row: prob_departure(row['area_type'], 
                                                                       row['incident_name'], 
                                                                       prob_dict), 
                                             axis=1)    

    return df_sample

def precompute_departure(df_sample, dic_inc_ar_mat):
    df_sample['departure'] = df_sample.apply(lambda row: get_departure(row['area_type'], 
                                                                       row['incident_name'], 
                                                                       dic_inc_ar_mat), 
                                             axis=1)    

    return df_sample

def get_departure(area_type, inc_name, dic_inc_ar_mat):
    if area_type in dic_inc_ar_mat[inc_name].keys():
        departure = dic_inc_ar_mat[inc_name][area_type]
    elif "*" in dic_inc_ar_mat[inc_name].keys():
        departure = dic_inc_ar_mat[inc_name]["*"]
    else:
        new_aera_dic = {k:v for k, v in dic_inc_ar_mat[inc_name].items() if v != ""}.values()
        departure = next(iter(new_aera_dic))

    return departure
    

def get_area_type(x, y, df_xy, df_lieu, df_nom_commune, df_commune, df_secteur, data, tree):

    point = np.array([[x, y]])
    dist, ind = tree.query(point, k=1)
    nearest_point = data[ind[0][0]]
    X, Y = nearest_point
    id_lieu = df_xy[(df_xy["Coordonnées X"] == X) & (df_xy["Coordonnées Y"] == Y)]["Identifiant unique du lieu"].iloc[0]

    if id_lieu in df_lieu:

        while id_lieu:
    
            part_of = df_lieu[df_lieu["NUMERO_LIEU"] == id_lieu]["EST_SITUE_SUR_NUMERO_LIEU"].iloc[0]
            area_number = df_lieu[df_lieu["NUMERO_LIEU"] == id_lieu]["NUMERO_TYPE_SECTEUR"].iloc[0]
            id_lieu = int(part_of)

    else:
        nom_com = df_xy[ df_xy["Identifiant unique du lieu"] == id_lieu]["Nom de la commune"].iloc[0]
        num_com = df_nom_commune[df_nom_commune["NOM_COMMUNE"]==nom_com]["NUMERO_COMMUNE"].iloc[0]
        num_secteur = df_commune[df_commune["NUMERO_COMMUNE"]==num_com]["NUMERO_TYPE_SECTEUR"].iloc[0]
        

    return df_secteur[df_secteur["NUMERO_TYPE_SECTEUR"] == num_secteur]["TYPE_SECTEUR"].iloc[0]



def precompute_date(df_sample, start_year, seed=42):
    np.random.seed(seed)
    vc = df_sample.Day.value_counts().sort_index()
    print(vc.idxmin(), vc.idxmax())
    print(vc)
    if "Minute" not in df_sample.columns:
        df_sample["Minute"] = np.random.randint(0, 60, size=len(df_sample))
    df_sample['date'] =  datetime(start_year, 1, 1) \
    + pd.to_timedelta(df_sample['Day'] -1, unit='D') \
    + pd.to_timedelta(df_sample['Hour'], unit='h') \
    + pd.to_timedelta(df_sample['Minute'], unit='m')
    # df_sample[['Month', 'Day', 'Hour', "Minute"]] = df_sample['date'].apply(lambda x: pd.Series([x.month, x.day, x.hour, x.minute]))
    df_sample["Month"] = df_sample["date"].dt.month
    df_sample["Day"] = df_sample["date"].dt.day
    df_sample["Hour"] = df_sample["date"].dt.hour
    df_sample["Minute"] = df_sample["date"].dt.floor('s').dt.minute

    return df_sample

def precompute_returns(df_sample, start_inter, end_inter, is_fake):
    df_sample['delta'] = pd.to_timedelta(df_sample['Duration'], unit='m')
    df_sample['date_return'] = df_sample['date'] + df_sample['delta']
    df_sample_sorted = df_sample.sort_values(by='date').reset_index(drop=True)
    df_sample_sorted["num_inter"] = range(start_inter, end_inter+1)
    df_sample_short = df_sample_sorted.copy()
    df_sample_short = df_sample_sorted[["num_inter", "date", "PDD", "departure", "zone", "Duration", "Month", "Day", "Hour", "Minute"]]
    # # if is_fake:
    df_sample_short.loc[:,["Coord X", "Coord Y", "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]] = \
        df_sample_sorted[["Coord X", "Coord Y", "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]]
    # else:
    #     df_sample_short[["Coord X", "Coord Y", "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]] = \
    #     0, 0, 0, 0, 0, 0, 0, 0
    df_return = pd.DataFrame()
    df_return["num_inter"] = df_sample_sorted["num_inter"]
    df_return["date"] = df_sample_sorted["date_return"]
    df_return["PDD"] = [[] for _ in range(len(df_sample_sorted))]
    df_return["departure"] = [{0:"RETURN"} for _ in range(len(df_sample_sorted))]
    df_return["zone"] = ""
    df_return["Duration"] = 0
    df_return[['Month', 'Day', 'Hour', "Minute"]] = df_return['date'].apply(lambda x: pd.Series([x.month, x.day, x.hour, x.minute]))
    df_return[["Coord X", "Coord Y", "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]] = 0, 0, 0, 0, 0, 0, 0, 0
    
    df_combined = pd.concat([df_return, df_sample_short], ignore_index=True) # il faut mettre le return en 1er pour régler le bug de simultanéité départ/retour
    df_combined_sorted = df_combined.sort_values(by='date', kind='mergesort').reset_index(drop=True)
    return df_combined_sorted

def create_responses(df_responses, df_rank_incident):

    dic_replace = {"CCF DEGRAD":"CCF", 
                   "XCOMPL":"COMPL", 
                   "VPL":"VSN", 
                   "EMB":"VEMB", 
                   "CEIN":"VSAV", 
                   'EPA DEGRAD':"EPC18",
                   'VL[X-GPT OPERATION]':'VL', 
                   'VSN[TOULOUSE - LOUGNON]':'VSN',  
                   'VFT[ST GAUDENS]':"VFT", 
                   'VSR[COLOMIERS]':"VSR", 
                   'VL[X-CTA CODIS]':"VL",
                   "CESDMF":"CESD",
                    'FPT   MPR':"FPT",
                    'VLHR[Z CRS LUCHON]':"VLHR",
                    'VPL':"VSN",
                   'PSECINC    MPR':"PSECINC",
                    'PSECINC2    MPR':"PSECINC2"
                  }
    df_responses["Materiel"] = df_responses["Materiel"].replace(dic_replace)
    df_responses = df_responses.dropna(subset=['Materiel']).reset_index(drop=True)
    dict_result = df_rank_incident.set_index('rank')['sin'].to_dict()
    valeurs_a_chercher = list(dict_result.values())
    df_responses_short =  df_responses[df_responses['Nom'].isin(valeurs_a_chercher)].reset_index(drop = True)
    dic_inc_ar_mat = {}

    for _, row in df_responses_short.iterrows():
        nom = row["Nom"]
        secteur = row["Secteur"]
        materiel = row["Materiel"]
        
        materiel_dict = extract_bracket_number_and_clean(materiel)
        
        if nom not in dic_inc_ar_mat:
            dic_inc_ar_mat[nom] = {}
        
        if secteur not in dic_inc_ar_mat[nom]:
            dic_inc_ar_mat[nom][secteur] = {}
        
        for num, materials in materiel_dict.items():
            
            dic_inc_ar_mat[nom][secteur][num] = materials

    return dic_inc_ar_mat

def extract_bracket_number_and_clean(materiel):
    # Dictionnaire où chaque numéro est associé à une liste de matériels
    result = {}

    # On sépare les matériels par numéro entre crochets
    split_by_number = re.split(r'(\[\d+\])', materiel)

    current_num = None
    for part in split_by_number:
        # Si la partie est un numéro entre crochets
        if re.match(r'\[\d+\]', part):
            current_num = int(part.strip('[]'))  # On extrait le numéro et le convertit en entier
            result[current_num] = []  # On initialise une liste pour ce numéro
        elif current_num is not None:
            # On divise les matériels par " |-- ou " et conserve les mots avec les espaces
            materials = re.split(r' \|-- ou ', part)
            for material in materials:
                # Extraction des mots en majuscules, en conservant les espaces entre eux
                # et en incluant [], _, et - s'ils sont présents
                combined_words = ' '.join(re.findall(r'[A-Z\[\]_\- ]+\d*', material))
                if combined_words.strip():
                    result[current_num].append(combined_words.strip())  # On ajoute les mots avec les espaces

    return result

def create_dic_planning(chemin_dossier):
    
    fichiers_csv = [f for f in os.listdir(chemin_dossier) if f.endswith('.csv')]
    
    # Initialiser une liste pour stocker chaque DataFrame
    liste_df = []
    
    # Lire chaque fichier CSV et l'ajouter à la liste des DataFrames
    for fichier in fichiers_csv:
        chemin_fichier = os.path.join(chemin_dossier, fichier)
        df = pd.read_csv(chemin_fichier, sep=";")
        liste_df.append(df)
    
    # Concaténer tous les DataFrames en un seul
    df_plan = pd.concat(liste_df, ignore_index=True)

    df_plan['Date Heure de début de tranche planning'] = pd.to_datetime(df_plan['Date Heure de début de tranche planning'], format="%d/%m/%Y %H:%M:%S")

    # Extraire uniquement la date (sans heure) de la colonne Date Heure de début de tranche planning
    df_plan['Mois'] = df_plan['Date Heure de début de tranche planning'].dt.month
    df_plan['Jour'] = df_plan['Date Heure de début de tranche planning'].dt.day
    
    planning = {}
    
    # Itérer sur chaque centre, jour et heure uniques
    for (centre, mois, jour, heure), group in df_plan.groupby(['Nom Centre', 'Mois', 'Jour', 'Heure']):
        
        if centre not in planning:
            planning[centre] = {}
    
        if mois not in planning[centre]:
            planning[centre][mois] = {}
        
        if jour not in planning[centre][mois]:
            planning[centre][mois][jour] = {}
        
        # Ajouter les matricules uniques pour chaque jour et heure
        planning[centre][mois][jour][heure] = {"planned": group['Matricule'].unique().tolist(), "available":group['Matricule'].unique().tolist(), "standby":[]}
    
    for centre in planning.keys():
        for mois in planning[centre]:  
            for jour in range(1, 32):
                if jour not in planning[centre][mois]:
                    planning[centre][mois][jour] = {}
                for heure in range(0, 24):
                    if heure not in planning[centre][mois][jour]:
                        planning[centre][mois][jour][heure] = {"planned":[], "available":[], "standby":[]}

    return planning

#########################################################################################

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Environment params")
    parser.add_argument("--prob_dep", action='store_true', help="if departure is computed from probabilities")
    parser.add_argument("--sample_list", nargs='+', help="List of samples to use")
    parser.add_argument("--save_as", type=str, help="List of samples to use")
    args = parser.parse_args()

    print("is probabilistic departure", args.prob_dep)

    Z_1 = ['TOULOUSE - LOUGNON', 'TOULOUSE - VION']
    Z_2 = ['ST JORY', 'ROUFFIAC', 'RAMONVILLE - BUCHENS', 'COLOMIERS', 'MURET - MASSAT']
    Z_3 = ['AUTERIVE', 'ST LYS', 'GRENADE', 'FRONTON', 'VERFEIL', 'CARAMAN']

    os.chdir('./Data')

    # df_sorties = pd.read_csv("sorties_2018.csv", sep=";")
    df_firestations = pd.read_csv("firestations.csv", sep = ';')
    df_materiel = pd.read_csv("materiel_2018.csv", sep=";")
    df_comp = pd.read_csv("comp_2018.csv", sep=";")
    df_roles = pd.read_csv("roles_competences.csv", sep=";")
    df_vehicles_history = pd.read_csv("df_vehicles_history.csv", sep=";")

    df_xy = pd.read_csv("X-Y-lieu.csv", sep=";")
    df_lieu = pd.read_csv("dbo.LIEU.csv", sep=";")
    df_secteur = pd.read_csv("dbo.SECTEUR.csv", sep=";")
    df_commune = pd.read_csv("dbo.COMMUNE.csv", sep=";")
    df_nom_commune = pd.read_csv("dbo.NOM_COMMUNE.csv", sep=";")
    df_responses = pd.read_csv("responses_by_incident.csv", sep = ';')
    df_pdd = gpd.read_file("pdd.geojson")

    df_stations = generate_stations(df_firestations)
    stations_u = sorted(x for x in df_materiel['Nom du Centre'].unique() if not x.startswith('X') and not x.startswith('Z'))    
    df_v = generate_vehicles(df_materiel, df_stations)    
    df_skills, df_firefighters = generate_firefighters(df_comp)

    
    os.chdir('../Data_environment')    
    
    df_stations.to_pickle("df_stations.pkl")
    df_v.to_pickle("df_v.pkl")
    df_skills.to_pickle("df_skills.pkl")
    df_roles.to_pickle("df_roles.pkl")
    df_vehicles_history.to_pickle("df_vehicles_history.pkl")

    os.chdir('../Data_preprocessed')

    df_prob_dep = pd.read_pickle("df_prob_dep.pkl")
    df_rank_incident = pd.read_pickle("df_rank_incident.pkl")

    dic_inc_ar_mat = create_responses(df_responses, df_rank_incident)

    # REAL

    os.chdir('../Data_trained')
    
    is_fake = False
    df_pc_real = pd.read_pickle("df_real.pkl")
    window = len(df_pc_real)
    print("window real:", window)
    df_pc_real = precompute_pdd(df_pdd, df_pc_real, stations_u)
    df_pc_real = precompute_zone(df_stations, df_pc_real, Z_1, Z_2, Z_3)
    df_pc_real = precompute_incident(df_rank_incident, df_pc_real)
    df_pc_real = precompute_area_type(df_xy, df_lieu, df_nom_commune, df_commune, df_secteur, df_pc_real)
    

    if args.prob_dep:
        df_pc_real = pd.concat([df_pc_real, df_prob_dep], axis=1)
        prob_dict = precompute_prob_dict(df_pc_real)
        print("prob_dict computed")           
        df_pc_real = precompute_prob_departure(df_pc_real, prob_dict)
    else:
        df_pc_real = precompute_departure(df_pc_real, dic_inc_ar_mat)

    start_year = 2018
    start_inter = 1
    end_inter = window

    print("start_year", start_year, "start_inter", start_inter, "end_inter", end_inter)
        
    df_pc_real = precompute_date(df_pc_real, start_year)
    df_pc_real = precompute_returns(df_pc_real, start_inter, end_inter, is_fake)
    print('real', len(df_pc_real), "done")

    cols_to_norm = ["Coord X", "Coord Y", 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'Hour_sin', 'Hour_cos']

    for col in cols_to_norm:
        min_val = df_pc_real[col].min()
        max_val = df_pc_real[col].max()
        if min_val != max_val:
            df_pc_real[col] = (df_pc_real[col] - min_val) / (max_val - min_val)
        else:
            df_pc_real[col] = 0.0

    os.chdir('../Data_environment')
    df_pc_real.to_pickle("df_pc_real.pkl")
    
    
    # FAKE

    os.chdir('../Data_sampled')
    is_fake = True
    dfs_to_concat = []
    start_year = 2018
    start_inter = 1
    window = 0
    end_inter = 0

    for sample_file in args.sample_list:

        start_inter += window
        window = len(pd.read_pickle(sample_file))
        print("window fake:", window)
        end_inter += window


        print("start_year", start_year, "start_inter", start_inter, "end_inter", end_inter)

        df_pc_fake = pd.read_pickle(sample_file) # "df_fake.pkl"
        df_pc_fake = precompute_pdd(df_pdd, df_pc_fake, stations_u)
        df_pc_fake = precompute_zone(df_stations, df_pc_fake, Z_1, Z_2, Z_3)
        df_pc_fake = precompute_incident(df_rank_incident, df_pc_fake)
        df_pc_fake = precompute_area_type(df_xy, df_lieu, df_nom_commune, df_commune, df_secteur, df_pc_fake)

        print(df_pc_fake.shape)
    
        if args.prob_dep:          
            df_pc_fake = precompute_prob_departure(df_pc_fake, prob_dict)
        else:
            df_pc_fake = precompute_departure(df_pc_fake, dic_inc_ar_mat)
            print(df_pc_fake.shape)

        df_pc_fake = precompute_date(df_pc_fake, start_year)

        

        
        print(df_pc_fake.shape)
        df_pc_fake = precompute_returns(df_pc_fake, start_inter, end_inter, is_fake)
        print(df_pc_fake.shape)
        print(sample_file, len(df_pc_fake), "done")
        dfs_to_concat.append(df_pc_fake)
        start_year += 1



    df_pc_fake = pd.concat(dfs_to_concat, ignore_index=True)

    df_pc_fake = reorg_dates(df_pc_fake)    
    print(df_pc_fake.columns)

    cols_to_norm = ["Coord X", "Coord Y", 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'Hour_sin', 'Hour_cos']

    for col in cols_to_norm:
        min_val = df_pc_fake[col].min()
        max_val = df_pc_fake[col].max()
        if min_val != max_val:
            df_pc_fake[col] = (df_pc_fake[col] - min_val) / (max_val - min_val)
        else:
            df_pc_fake[col] = 0.0

    
    
    os.chdir('../Data_environment')
    df_pc_fake.to_pickle(args.save_as)
    print("global fake done", len(df_pc_fake))

    os.chdir('../Data')

    path = "./Planning/"
    planning = create_dic_planning(path) 

    os.chdir('../Data_environment')

    pickle.dump(planning, open("planning.pkl", "wb"))

    print("Planning done")

    os.chdir('../')