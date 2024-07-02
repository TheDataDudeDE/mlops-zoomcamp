import pandas as pd
from datetime import datetime
import batch
import pandas as pd
import pickle


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


options = {"client_kwargs": {"endpoint_url": "http://localhost:4566"}}


data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]

df = pd.DataFrame(data, columns=columns)
categorical = ["PULocationID", "DOLocationID"]
df = batch.prepare_data(df=df, categorical=categorical)
with open("model.bin", "rb") as f_in:
    dv, lr = pickle.load(f_in)
df_result = batch.predict_duration(df, categorical, 2023, 1, dv, lr)
batch.save_data(df_result, "s3://nyc-duration/prediction_2023_01.parquet", options)
