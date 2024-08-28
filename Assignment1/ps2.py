import numpy as np
import pandas as pd

df = pd.read_csv("landslide_data_miss.csv")

missing_values = df[df["stationid"].isna()]

#print(missing_values)

df = df.dropna(subset=["stationid"])

df.to_csv("values_deleted.csv",index = False)