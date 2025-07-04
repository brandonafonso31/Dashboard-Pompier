import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

import geopandas as gpd
from shapely import MultiPolygon, unary_union, Point
from random import getrandbits, uniform
from data import *
from ddpm import DDPM
import pickle
import argparse
import os

def quantile_inverse_transform(df_res, normalizer_ddpm):

    cols = ["Coord X", "Coord Y", "Duration"]
    df_ddpm = pd.DataFrame(columns=cols)    
    normalizer_ddpm = pickle.load(open('normalizer_ddpm.pkl', 'rb'))    
    df_ddpm[cols] = normalizer_ddpm.inverse_transform(df_res[cols].values)
    return df_ddpm

def decode_periodic(sin_value, cos_value, period):
    angle = np.arctan2(sin_value, cos_value)
    if angle < 0:
        angle += 2 * np.pi  # Ensure angle is in [0, 2*pi) range
    value = (angle * period) / (2 * np.pi)
    return round(value)

def sincos_inverse_transform(df_sample, df_res, y_gen):

    df_sample["Day"] = df_res.apply(lambda row: decode_periodic(row['Day_sin'], row['Day_cos'], 365), axis=1)
    # df_sample["Day_week"] = df_res.apply(lambda row: decode_periodic(row['Day_week_sin'], row['Day_week_cos'], 7), axis=1)
    # df_sample["Day_month"] = df_res.apply(lambda row: decode_periodic(row['Day_month_sin'], row['Day_month_cos'], 31), axis=1)
    df_sample["Month"] = df_res.apply(lambda row: decode_periodic(row['Month_sin'], row['Month_cos'], 12), axis=1)
    df_sample["Hour"] = df_res.apply(lambda row: decode_periodic(row['Hour_sin'], row['Hour_cos'], 24), axis=1)

    # df_sample.loc[df_sample["Month"] > 11, "Month"] = 0
    # # df_sample.loc[df_sample["Day"] > 364, "Day"] = 0
    # df_sample.loc[df_sample["Hour"] > 23, "Hour"] = 0
    # df_sample.loc[df_sample["Day_week"] > 6, "Day_week"] = 0
    # df_sample.loc[df_sample["Day_month"] > 30, "Day_month"] = 0

    df_sample["Incident"] = y_gen + 1
    df_sample["Duration"] = df_sample["Duration"].apply(lambda x : x if x > 10 and x < 20*60 else df_sample["duree"].median()).astype(int)
    
    return df_sample

def clean_and_shift(row, cis_cols, li_new_cis):
    
    filtered_values = [val for val in row[cis_cols] if not (str(val).startswith("Z") or str(val).startswith("X")) and not val in li_new_cis]
    new_values = filtered_values + [""] * (len(cis_cols) - len(filtered_values))
    row.update(pd.Series(new_values, index=cis_cols))
    return row

def union_iris(df_pdd, cis, li_new_cis):

    df_pdd = df_pdd[~df_pdd[cis].str.startswith("Z")]
    cis_cols = [col for col in df_pdd.columns if col.startswith("cis")]
    df_pdd = df_pdd.apply(clean_and_shift, args=(cis_cols, li_new_cis,), axis=1)

    area_name_unique = sorted(df_pdd[cis].unique())
    areas_geo = [MultiPolygon( list(df_pdd[df_pdd[cis] == nom].geometry) ) for nom in area_name_unique]

    areas = gpd.GeoDataFrame({'sector' : area_name_unique, 'geometry' : areas_geo} , crs=df_pdd.crs)
    areas.geometry = areas.geometry.to_crs(2154)
    
    areas.geometry = areas.geometry.apply(lambda x : unary_union(x))

    return areas

def get_point_in_area(point, zones):
    res = zones["sector"][zones.contains(point)].values
    if res.size > 0:
        return res[0]
    else:
        return ""

def create_df_new_samples(df_raw, col, var=0.02):

    df_test = pd.DataFrame(df_raw[col].value_counts().reset_index())
    # df_test = df_test.rename(columns={"index" : "area_name", "area_name":"count"})
    df_test.loc[:, "new_samples"] = df_test["count"].apply(gen_num_samples, args=(var,))
    df_test.loc[:, "perc."] = df_test["new_samples"] / df_test["count"] * 100 - 100
    df_test.loc[:, "delta"] = df_test["new_samples"] - df_test["count"]
    df_test.loc[df_test["delta"].idxmax(), "new_samples"] -= df_test["delta"].sum()
    df_test.loc[:, "perc."] = df_test["new_samples"] / df_test["count"] * 100 - 100
    df_test.loc[:, "delta"] = df_test["new_samples"] - df_test["count"]
    return df_test

def gen_num_samples(num_samples, var):

    sign = getrandbits(1)
    
    if sign:
        x = uniform(1, 1+var)
    else:
        x = uniform(1-var, 1)

    return int(round(num_samples * x))

def new_df_sample(df_test, col, df_oversampled):

    list_of_df = []
    for idx, row in df_test.iterrows():
                
        if len(df_oversampled[df_oversampled[col] == row[col]]) >= row["new_samples"]:
            list_of_df.append(df_oversampled[df_oversampled[col] == row[col]].sample(int(row["new_samples"])))
            
        else:            
            list_of_df.append(df_oversampled[df_oversampled[col] == row[col]])

    return pd.concat(list_of_df, ignore_index=True)

def post_process(df):
    df.loc[df["Month"] < 1 , "Month"] = 12
    df.loc[df["Day"] < 1, "Day"] = 1
    df.loc[df["Hour"] > 23, "Hour"] = 0
    # df.loc[df["Day_month"] > 30, "Day_month"] = 0
    return df

def distance_days(a, b, nb_days=365):
    return min(abs(a - b), nb_days - abs(a - b))

def harmonize(df1, to_keep = 10, value_span = 25, nb_days = 365):
    vc1 = df1.Day.value_counts().iloc[to_keep:value_span+to_keep]
    top_days = vc1.index.tolist()
    vc2 = df1.Day.value_counts().iloc[-value_span:]
    flop_days = vc2.index.tolist()

    for ref_day in top_days:

        nearest_day = min(flop_days, key=lambda d: distance_days(d, ref_day, nb_days))
        ref_inter_1 = df1[df1.Day == ref_day]
        ref_inter_2 = df1[df1.Day == nearest_day]
        mu = (len(ref_inter_1) + len(ref_inter_2)) // 2
        to_move = len(ref_inter_1) - mu
        inter_sampled = ref_inter_1.sample(n=to_move)
        df1.loc[inter_sampled.index,"Day"] = nearest_day        
        flop_days.remove(nearest_day)
    
    return df1

#############################################################################################""

if __name__ == "__main__":

    os.chdir('./Data_preprocessed')

    df_real = pd.read_pickle("df_real.pkl")

    os.chdir('../Data_trained')
    
    with open("dataset.pkl", "rb") as file:
        dataset = pickle.load(file)
    
    parser = argparse.ArgumentParser(description="Train_params")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--layers", type=int, default=1024, help="Size of layers")
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate")
    parser.add_argument("--dim_t", type=int, default=128, help="Timestep embedding dimensions")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay")
    parser.add_argument("--model_name", type=str, default="mlp", help="Model name")
    parser.add_argument("--gaussian_loss_type", type=str, default="mse", help="Gaussian loss type : mse or kl")
    parser.add_argument("--multinomial_loss_type", type=str, default="vb_stochastic", \
                        help="Multinomial loss type : vb_stochastic or vb_all")
    parser.add_argument("--parametrization", type=str, default="x0", help="Parametrization : x0 or direct")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler : cosine or linear")
    parser.add_argument("--is_y_cond", type=bool, default=True, help="Is target to predict")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("--model_path", type=str, default="./model_ddpm.pt", help="Model path")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--save_as", type=str, default="dqn", help="Save model in path_to_file")
    parser.add_argument("--load_as", type=str, default="dqn", help="Load model from path_to_file")
    parser.add_argument("--os_factor", type=int, default=3, help="Oversampling factor")
    parser.add_argument("--to_keep", type=int, default=10, help="nb of highest values to keep by harmonization")
    parser.add_argument("--value_span", type=int, default=25, help="nb of interventions to permute by harmonization")    
    parser.add_argument("--sample_batch_size", type=int, default=8192, help="Batch size of samples")
    parser.add_argument("--variability", type=float, default=0.02, help="Tolerated variability")
    parser.add_argument("--save_sample_as", type=str, default="df_fake.pkl", help="Save sample in path_to_file")

    args = parser.parse_args()

    print("os factor:", args.os_factor, "sample batch size:", args.sample_batch_size, "variability:", args.variability, flush=True)

    os.chdir('../')

    ddpm = DDPM(lr = args.lr, layers = args.layers, num_timesteps = args.num_timesteps, \
                model_name = args.model_name, dim_t = args.dim_t, gaussian_loss_type = args.gaussian_loss_type, \
                multinomial_loss_type = args.multinomial_loss_type, parametrization = args.parametrization, \
                scheduler = args.scheduler, is_y_cond = args.is_y_cond, weight_decay = args.weight_decay, \
                batch_size=args.batch_size, log_every=100,  verbose=args.verbose, epochs=args.epochs, device=args.device, \
                save_as = args.save_as, load_as=args.load_as)

    num_samples = len(df_real) * args.os_factor

    X_gen, y_gen = ddpm.sample(dataset, num_samples, args.sample_batch_size)

    os.chdir('./Data_trained')

    df_quantile = pickle.load(open("df_quantile.pkl", "rb"))

    normalizer_ddpm = pickle.load(open('normalizer_ddpm.pkl', 'rb')) 

    cols = list(df_quantile.columns)
    cols.remove("Incident") # for same size dataframe
    df_res = pd.DataFrame(data = X_gen, columns = cols)

    df_sample = quantile_inverse_transform(df_res, normalizer_ddpm)
    df_oversampled = sincos_inverse_transform(df_sample, df_res, y_gen)
    # df_oversampled[["Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]] = \
    # df_res[["Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]]

    

    df_oversampled[["Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]] = \
    df_res[["Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]]

    os.chdir('../Data')

    filename = "pdd.geojson"
    df_pdd = gpd.read_file(filename)
    
    zones = union_iris(df_pdd, "cis1" , ['MONTGISCARD', 'AUSSONNE', 'TOULOUSE - ATLANTA', 'TOULOUSE - CARSALADE', 'TOULOUSE - DELRIEU'])
    
    gdf_real = gpd.GeoDataFrame(df_real, geometry=gpd.points_from_xy(df_real['Coord X'], df_real['Coord Y']), crs="2154")
    df_real['area_name'] = gdf_real["geometry"].apply(get_point_in_area, args=(zones,))
    
    gdf_fake = gpd.GeoDataFrame(df_oversampled, geometry=gpd.points_from_xy(df_oversampled['Coord X'], df_oversampled['Coord Y']), crs="2154")
    df_oversampled['area_name'] = gdf_fake["geometry"].apply(get_point_in_area, args=(zones,))
    
    df_new_samples = create_df_new_samples(df_real, "area_name", args.variability)
    print(df_new_samples.columns, flush=True)
    if df_new_samples.delta.sum() == 0:
        print("new samples OK", flush=True)
    else:
        print("new samples NOT OK", df_new_samples.delta.sum(), flush=True)
    
    df_fake = new_df_sample(df_new_samples, "area_name", df_oversampled)

    df_fake = post_process(df_fake)

    os.chdir('../Data_sampled')

    df_fake.to_pickle('df_fake_woh.pkl')

    df_fake = harmonize(df_fake, args.to_keep, args.value_span, 365)

    df_fake.to_pickle(args.save_sample_as)

    print(df_fake.shape, flush=True)
    print(df_fake.columns, flush=True)

    

    df_fake.to_pickle(args.save_sample_as)

    print("dataset sampled", flush=True)

    os.chdir('../')