import pandas as pd
import argparse

def repeat_parquet_rows(input_path, output_path, repeat_times=32):
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows")

    df_repeated = df.loc[df.index.repeat(repeat_times)].reset_index(drop=True)

    print(f"After repeat x{repeat_times}: {len(df_repeated)} rows")

    df_repeated.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input parquet file")
    parser.add_argument("output", help="output parquet file")
    parser.add_argument("--repeat", type=int, default=32)

    args = parser.parse_args()

    repeat_parquet_rows(args.input, args.output, args.repeat)
