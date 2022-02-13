import pandas as pd
from pathlib import Path

import pandas_profiling

data_path = Path("..", "data", "census.csv")

df = pd.read_csv(data_path)
profile = pandas_profiling.ProfileReport(df, title="PandasProfilingReport_Census", explorative=True)
profile.to_file("PandasProfilingReport_Census.html")
