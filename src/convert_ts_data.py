import os
import pandas as pd
import argparse
import json

def convert_ts_data(df: pd.DataFrame) -> dict:
    # Convert the DataFrame to a time series format
    ts_data = df.to_dict(orient= "records")
    return ts_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to time series dictionary format.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
    parser.add_argument("output_file", type=str, help="Path for the output file (JSON)")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    ts_data = convert_ts_data(df)

    with open(args.output_file, "w") as f:
        json.dump(ts_data, f, indent=4)