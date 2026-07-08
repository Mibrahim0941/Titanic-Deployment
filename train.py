import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

train = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")

print("\n Train Set Null Values ")
print(train.isnull().sum())
print("\n Test Set Null Values ")
print(test.isnull().sum())

test_passenger_ids = test["PassengerId"]
train["is_train"] = 1
test["is_train"] = 0
full = pd.concat([train, test], sort=False, ignore_index=True)

full["Title"] = full["Name"].str.extract(r",\s*([^\.]+)\.")

title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}

full["Title"] = full["Title"].replace(title_map)
full.loc[~full["Title"].isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Title"] = "Rare"

full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

full["Deck"] = full["Cabin"].str[0].fillna("U")
 
full["Fare"] = full.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.median())
)
full["FareBin"] = pd.qcut(full["Fare"], 4, labels=False, duplicates="drop")

full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

age_medians = full.groupby(["Title", "Pclass"])["Age"].median().to_dict()
joblib.dump(age_medians, "age_medians.pkl")

full["Age"] = full.groupby(["Title", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
full["Age"] = full["Age"].fillna(full["Age"].median()) 

full["AgeBin"] = pd.cut(
    full["Age"],
    bins=[0, 12, 18, 30, 45, 60, 100],
    labels=False
)

sex_encoder = LabelEncoder()
full["Sex"] = sex_encoder.fit_transform(full["Sex"].astype(str))

embarked_encoder = LabelEncoder()
full["Embarked"] = embarked_encoder.fit_transform(full["Embarked"].astype(str))

title_encoder = LabelEncoder()
full["Title"] = title_encoder.fit_transform(full["Title"].astype(str))

deck_encoder = LabelEncoder()
full["Deck"] = deck_encoder.fit_transform(full["Deck"].astype(str))

joblib.dump(sex_encoder, "sex_encoder.pkl")
joblib.dump(embarked_encoder, "embarked_encoder.pkl")
joblib.dump(title_encoder, "title_encoder.pkl")
joblib.dump(deck_encoder, "deck_encoder.pkl")
 
features = [
    "Pclass", "Sex", "Age", "AgeBin", "SibSp", "Parch",
    "Fare", "FareBin", "Embarked", "Title", "FamilySize",
    "IsAlone", "Deck",
]
 
train_df = full[full["is_train"] == 1]
test_df = full[full["is_train"] == 0]
 
X = train_df[features]
y = train_df["Survived"].astype(int)
X_test = test_df[features]

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_split=6,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X, y)
predictions = model.predict(X_test).astype(int)

submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": predictions
})

submission.to_csv("Dataset/submission.csv", index=False)
print("Saved submission.csv")

joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns),"columns.pkl")



