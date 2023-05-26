from predictapi import index, predict
# import pandas as pd
# import requests

def test_index():
#     api_url = 'http://localhost:5000/'
#     response = requests.get(api_url)
#     assert response.status_code == 200
#     assert response.text == "Hello, world!"
    assert index() == "Hello, world!"


# 
# def test_predict():
    # api_url = 'http://localhost:5000/predict'
    # 
    # client = pd.read_csv("df_credit_dash_score.csv", nrows=1).iloc[:, 2:-2]
    # data = {'client_feat': client.to_json(orient="records")}
    # 
    # response = requests.get(api_url, json=data)  
    # 
    # assert response.status_code == 200
    # result = response.json()
    # prediction = result['prediction']
    # assert prediction in [0, 1]
    # prediction_proba = result['prediction_proba']
    # assert 0 <= prediction_proba <= 1
    