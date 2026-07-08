import requests
import json
import traceback

data = {
    "Name": "Braund, Mr. Owen Harris",
    "Pclass": 1,
    "Sex": "male",
    "Age": None,
    "SibSp": 1,
    "Parch": 0,
    "Fare": None,
    "Embarked": "S",
    "Cabin": None
}

try:
    from app import app
    with app.test_client() as client:
        res = client.post('/predict', json=data)
        print("Status code:", res.status_code)
        if res.status_code != 200:
            print(res.get_data(as_text=True))
        else:
            print(res.json)
except Exception as e:
    traceback.print_exc()
