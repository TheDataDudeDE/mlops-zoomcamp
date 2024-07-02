#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os


def prepare_data(df, categorical):
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


def read_data(filename):
    df = pd.read_parquet(filename)
    return df


def predict_duration(df, categorical, year, month, dv, lr):
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.mean())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred
    return df_result


def save_data(df, filename, options):
    df.to_parquet(filename, engine="pyarrow", storage_options=options)
    return "File saved"


def get_input_path(year, month):
    default_input_pattern = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = (
        "s3://nyc-duration/predictions_{year:04d}_{month:02d}.parquet"
    )
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def set_environment_variables(year, month):
    input_file_pattern = f"s3://nyc-duration/{year:04d}-{month:02d}.parquet"
    output_file_pattern = f"s3://nyc-duration/{year:04d}-{month:02d}.parquet"

    os.environ["INPUT_FILE_PATTERN"] = input_file_pattern
    os.environ["OUTPUT_FILE_PATTERN"] = output_file_pattern

    print(f"Set INPUT_FILE_PATTERN to {input_file_pattern}")
    print(f"Set OUTPUT_FILE_PATTERN to {output_file_pattern}")


def main(year, month):
    options = {"client_kwargs": {"endpoint_url": "http://localhost:4566"}}
    input_file = get_input_path(year=year, month=month)
    output_file = get_output_path(year=year, month=month)

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)
    categorical = ["PULocationID", "DOLocationID"]

    df = read_data(filename=input_file)
    df = prepare_data(df, categorical=categorical)

    df_result = predict_duration(df, categorical, year, month, dv, lr)

    save_data(df_result, output_file, options=options)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    if len(sys.argv) > 3:
        if sys.argv[3].lower() in ("true", "1", "yes", "y"):
            env = True
        else:
            env = False
    else:
        env = False

    if env:
        set_environment_variables(year=year, month=month)

    main(year=year, month=month)
