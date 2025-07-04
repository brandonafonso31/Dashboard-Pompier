import numpy as np
import pandas as pd

def encode_periodic(value, period):
    angle = (2 * np.pi * value) / period
    return np.sin(angle), np.cos(angle)

def sincos_transform(df_raw):
    df_raw[["Month_sin", "Month_cos"]] = df_raw["Month"].apply(lambda x: pd.Series(encode_periodic(x, 12)))
    df_raw[["Day_sin", "Day_cos"]] = df_raw["Day"].apply(lambda x: pd.Series(encode_periodic(x, 365)))
    df_raw[["Hour_sin", "Hour_cos"]] = df_raw["Hour"].apply(lambda x: pd.Series(encode_periodic(x, 24)))
    return df_raw[["Month_sin", "Month_cos", "Hour_sin", "Hour_cos", "Day_sin", "Day_cos"]]
