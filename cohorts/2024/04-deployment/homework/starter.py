#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import argparse


def read_data(filename):
    categorical = ["PULocationID", "DOLocationID"]
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def main(year, month):
    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = read_data(filename)

    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    df_result = pd.DataFrame({"ride_id": df["ride_id"], "predictions": y_pred})
    print(
        f"Mean predicted duration {year:04d}/{month:02d} : {df_result.predictions.mean()}"
    )
    output_file = f"predictions_{year:04d}_{month:02d}.parquet"
    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for taxi data.")
    parser.add_argument("year", type=int, help="Year of the data")
    parser.add_argument("month", type=int, help="Month of the data")

    args = parser.parse_args()
    main(args.year, args.month)
