import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", required=True)
args = parser.parse_args()

df = pd.read_csv(args.file, index_col=0)
df.to_csv(args.file, index=False)
