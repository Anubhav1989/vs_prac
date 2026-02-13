import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:\\Users\\anubh\\OneDrive\\Desktop\\prtc\\covid_toy - covid_toy.csv")

df = df.dropna()

lb = LabelEncoder()
for col in ["gender", "cough", "city"]:
    df[col] = lb.fit_transform(df[col])

x = df.drop(columns=["has_covid"])
y = df["has_covid"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

print(x_train.head(2))
