from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
sex_encoder = joblib.load("sex_encoder.pkl")
embarked_encoder = joblib.load("embarked_encoder.pkl")
title_encoder = joblib.load("title_encoder.pkl")
deck_encoder = joblib.load("deck_encoder.pkl")
age_medians = joblib.load("age_medians.pkl")

title_map = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
}

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "AgeBin",
    "SibSp",
    "Parch",
    "Fare",
    "FareBin",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
    "Deck",
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    df = pd.DataFrame([data])

    title = df.loc[0, "Name"].split(",")[1].split(".")[0].strip()

    title = title_map.get(title, title)

    if title not in ["Mr", "Mrs", "Miss", "Master", "Rare"]:
        title = "Rare"

    df["Title"] = title

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    cabin = str(df.loc[0, "Cabin"])

    if cabin == "" or cabin == "nan":
        deck = "U"
    else:
        deck = cabin[0]

    df["Deck"] = deck

    df["Embarked"] = df["Embarked"].fillna("S")

    if pd.isna(df.loc[0, "Age"]):
        t = df.loc[0, "Title"]
        p = int(df.loc[0, "Pclass"])
        df.loc[0, "Age"] = age_medians.get((t, p), 30)
        
    df["Fare"] = df["Fare"].fillna(32)

    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 30, 45, 60, 100],
        labels=False
    )

    df["FareBin"] = pd.cut(
        df["Fare"],
        bins=[-1, 8, 15, 31, np.inf],
        labels=False
    )

    df["Sex"] = sex_encoder.transform(df["Sex"].astype(str))

    df["Embarked"] = embarked_encoder.transform(df["Embarked"].astype(str))

    df["Title"] = title_encoder.transform(df["Title"].astype(str))

    df["Deck"] = deck_encoder.transform(df["Deck"].astype(str))

    X = df[FEATURES]

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    return jsonify({
        "survived": bool(prediction),
        "probability": round(float(probability), 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)