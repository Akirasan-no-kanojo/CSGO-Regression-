import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv("csgo.csv")
profile= ProfileReport(data, title="CSGO Report")
profile.to_file("CSGO Report.html")