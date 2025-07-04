import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from data import *
from ddpm import DDPM
import pickle
import argparse

import os

def quantile_transform(df_sincos):
    cols = ["Coord X", "Coord Y", "Duration"] 
    to_QT = df_sincos[cols].values
    
    normalizer_ddpm = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=1000, # max(min(to_QT.shape[0] // 30, 1000), 10),
            subsample=1000000000,
            random_state=42)    
    df_sincos.loc[:, cols] = normalizer_ddpm.fit_transform(to_QT)
    
    
    return df_sincos, normalizer_ddpm

def encode_periodic(value, period):
    angle = (2 * np.pi * value) / period
    return np.sin(angle), np.cos(angle)

def sincos_transform(df_raw):
    df_raw[["Month_sin", "Month_cos"]] = df_raw["Month"].apply(lambda x: pd.Series(encode_periodic(x, 12)))
    df_raw[["Day_sin", "Day_cos"]] = df_raw["Day"].apply(lambda x: pd.Series(encode_periodic(x, 365)))
    df_raw[["Hour_sin", "Hour_cos"]] = df_raw["Hour"].apply(lambda x: pd.Series(encode_periodic(x, 24)))
    # df_raw[["Day_week_sin", "Day_week_cos"]] = df_raw["Day_week"].apply(lambda x: pd.Series(encode_periodic(x, 7)))
    # df_raw[["Day_month_sin", "Day_month_cos"]] = df_raw["Day_month"].apply(lambda x: pd.Series(encode_periodic(x, 31)))
    # return df_raw[['Coord X', 'Coord Y', "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos", 'Duration', 'Incident']]
    return df_raw[["Month_sin", "Month_cos", "Hour_sin", "Hour_cos", "Day_sin", "Day_cos"]]

######################################################################################################################

if __name__ == "__main__":

    os.chdir('./Data_preprocessed')    
    df_real = pd.read_pickle("df_real.pkl")

    os.chdir("../Data_trained/")

    # df_real["Incident"] -= 1 #for tabddpm, min incident should be 0 not 1
    
    df_sincos = sincos_transform(df_real.copy())

    df_real_temp = df_sincos.copy()
    df_real_temp[['Coord X', 'Coord Y', 'Duration', 'Incident', "Month", "Day", "Hour"]] = df_real[['Coord X', 'Coord Y', \
                                                                         'Duration', 'Incident', "Month", "Day", "Hour"]].copy()
    print(df_real_temp.columns)
    df_real_temp.to_pickle("df_real.pkl")

    df_sincos[['Coord X', 'Coord Y', 'Duration', 'Incident']] = df_real[['Coord X', 'Coord Y', \
                                                                         'Duration', 'Incident']].copy()

    print(df_sincos.columns)

    df_sincos["Incident"] -= 1 #for tabddpm, min incident should be 0 not 1
    
    df_quantile, normalizer_ddpm = quantile_transform(df_sincos)

    df_quantile.to_pickle("df_quantile.pkl")

    pickle.dump(normalizer_ddpm, open('normalizer_ddpm.pkl', 'wb'))
    
    dataset = raw_dataset_from_df(df_quantile, [], dummy = False, col = "Incident")
    
    pickle.dump(dataset, open("dataset.pkl", "wb"))
    
    parser = argparse.ArgumentParser(description="Train_params")
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--layers", type=int, default=1024, help="Size of layers")
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate")
    parser.add_argument("--dim_t", type=int, default=128, help="Timestep embedding dimensions")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay")
    parser.add_argument("--model_name", type=str, default="mlp", help="Model type : mlp or resnet")
    parser.add_argument("--gaussian_loss_type", type=str, default="mse", help="Gaussian loss type : mse or kl")
    parser.add_argument("--multinomial_loss_type", type=str, default="vb_stochastic", \
                        help="Multinomial loss type : vb_stochastic or vb_all")
    parser.add_argument("--parametrization", type=str, default="x0", help="Parametrization : x0 or direct")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler : cosine or linear")
    parser.add_argument("--is_y_cond", action='store_true', help="Is target to predict")
    parser.add_argument("--verbose", action='store_true', help="Verbose")
    parser.add_argument("--save_as", type=str, default="dqn", help="Save model in file")
    parser.add_argument("--load_as", type=str, default="dqn", help="Load model from file")
    parser.add_argument("--device", type=str, default="cuda", help="device")

    args = parser.parse_args()

    os.chdir('../')

    ddpm = DDPM(lr = args.lr, layers = args.layers, num_timesteps = args.num_timesteps, \
                model_name = args.model_name, dim_t = args.dim_t, gaussian_loss_type = args.gaussian_loss_type, \
                multinomial_loss_type = args.multinomial_loss_type, parametrization = args.parametrization, \
                scheduler = args.scheduler, is_y_cond = args.is_y_cond, weight_decay = args.weight_decay, \
                batch_size=args.batch_size, log_every=100,  verbose=args.verbose, epochs=args.epochs, device=args.device, \
                save_as = args.save_as, load_as=args.load_as)
    
    ddpm.fit(dataset)
    
    # os.chdir('../')
    
    print("Model trained")

