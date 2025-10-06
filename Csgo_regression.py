import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

target= "points"
data = pd.read_csv("csgo.csv")
cols_to_drop = ["date", "day", "month", "year", "wait_time_s"] 
x = data.drop(columns=[target] + cols_to_drop)
y= data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 80)
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
Result = [
    "Win",
    "Tie",
    "Lost"
]
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[Result]))
])
nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformer = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["match_time_s", "team_a_rounds", "team_b_rounds", "ping", "kills", "assists", "deaths", "mvps", "hs_percent"]),
    ("ord_features", ord_transformer, ["result"]),
    ("nom_features", nom_transformer, ["map"]),
])

params = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__criterion": ["squared_error", "absolute_error"],
    "regressor__max_depth": [None, 2, 5, 10],
    "regressor__min_samples_split": [2, 5],
    "preprocessor__num_features__imputer__strategy": ["mean", "median"]
}
model = Pipeline(steps=[
    ("preprocessor", transformer),
    ("regressor", RandomForestRegressor(random_state=100))
])

model_gs = GridSearchCV(
    estimator=model,
    param_grid=params,
    refit=True,
    scoring="r2",
    cv=6,
    verbose=2,
    n_jobs=-1
)
model_gs.fit(x_train, y_train)
y_predict = model_gs.predict(x_test)
print(model_gs.best_params_)
print(model_gs.best_score_)
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))
with open("csgo_model.pkl", "wb") as f:
      pickle.dump(model_gs.best_estimator_, f)