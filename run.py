# Solution code for the Iris Dataset Homework (run.py)

import pandas as pd
from scipy.stats import zscore

# Question 1: Pre-process the data
def preprocess_data(input_filename):
    data = pd.read_csv(input_filename)
    data.columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
    z_score_length = (data["SepalLengthCm"] - data["SepalLengthCm"].mean()) / data["SepalLengthCm"].std()
    z_score_width = (data["SepalWidthCm"] - data["SepalWidthCm"].mean()) / data["SepalWidthCm"].std()
    data = data[(abs(z_score_length) <= 2) & (abs(z_score_width) <= 2)]
    data["ID"] = range(1, len(data) + 1)
    return data


# Question 2: Descriptive Statistics Functions
def species_count(data):
    df = preprocess_data(data)
    print(df["Species"].value_counts())
    return df["Species"].value_counts().to_dict()

def average_sepal_length(data):
    df = preprocess_data(data)
    return round(df["SepalLengthCm"].mean(), 1)

def max_petal_width(data):
    df = preprocess_data(data)
    return round(df["PetalWidthCm"].max(), 1)

def min_petal_length(data):
    df = preprocess_data(data)
    return round(df["PetalLengthCm"].min(), 1)

def count_sepal_length_above_5(data):
    df = preprocess_data(data)
    return len(df[df["SepalLengthCm"] > 5])

# Question 3: Analysis Functions
def count_petal_length_below_2(data):
    df = preprocess_data(data)
    return len(df[df["PetalLengthCm"] < 2])

def get_sepal_width_above_3_5(data):
    df = preprocess_data(data)
    return list(df[df["SepalWidthCm"] > 3.5]["ID"].values)

def species_count_petal_width_above_1_5(data):
    df = preprocess_data(data)
    return df[df["PetalWidthCm"] > 1.5]["Species"].value_counts().to_dict()

def get_virginica_petal_length_above_6(data):
    df = preprocess_data(data)
    return list(df[(df["PetalLengthCm"] > 6) & (df["Species"] == "Iris-virginica")]["ID"].values)
    
def get_largest_sepal_width(data):
    df = preprocess_data(data)
    return df[df["SepalWidthCm"] == df["SepalWidthCm"].max()]["ID"].iloc[0]
