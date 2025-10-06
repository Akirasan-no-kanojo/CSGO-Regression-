import pickle
import pandas as pd


with open("csgo_model.pkl", "rb") as f:
    model = pickle.load(f)


data = pd.read_csv("csgo.csv")

cols_to_drop = ["date", "day", "month", "year", "wait_time_s"]
x = data.drop(columns=["points"] + cols_to_drop)
y_pred = model.predict(x)

print("Complete prediction:")
print(y_pred[:10])  # In thử 10 kết quả đầu