from flask import Flask, jsonify, request
import pandas as pd
import pickle
import shap
import json

app = Flask(__name__)

# Charger le modèle de classifieur
classifier = pickle.load(open('lgbm_model.pkl', 'rb'))
@app.route('/')
def index():
    return "Hello, world!"

@app.route('/predict', methods=['GET'])
def predict():
    
    # Récupérer les données envoyées en tant que JSON
    data = request.get_json() 
    X = pd.read_json(data["client_feat"])
    
    # Effectuer la prédiction
    prediction = classifier.predict(X)[0]
    prediction_proba = round(classifier.predict_proba(X)[0][0], 2)

    # Shap values
    explainer = shap.Explainer(classifier["clf"])
    X_ = pd.DataFrame(classifier['scaler'].transform(X), columns = X.columns)
    shap_values = explainer(X_)[:, :, 1]
    shap_values.data = X.values
    sh = {"values": shap_values[0].values.tolist(),
          "base_values": shap_values[0].base_values.tolist(),
          "data": shap_values[0].data.tolist()}
    sh_json = json.dumps(sh)
    
    # Renvoyer la réponse sous forme de JSON
    response = {'prediction': prediction,
                'prediction_proba': prediction_proba, # Récupérer le score 
                'shap_values': sh_json}  
    
    return jsonify(response)
    

if __name__ == '__main__':
    app.run()     
    

    
    
