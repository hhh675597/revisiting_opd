import pyarrow.parquet as pq

file_path = "/data/home/zdhs0086/hhh/verl-agent/data/math/dapo-math-17k.parquet"
table = pq.read_table(file_path)
num_rows = table.num_rows
print(f"Number of rows: {num_rows}")