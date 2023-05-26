import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import shap
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
st.balloons()

#---------------------------------#
@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data("df_credit_dash_score.csv")
features = list(df.iloc[:, 2:-2].columns)

#---------------------------------#
# Sidebar
col1 = st.sidebar
image = Image.open('pretadepenser.png')
col1.image(image, width = 200, use_column_width=True)

#col1.header('  Identifiant Client')
selected_client = col1.selectbox('Identifiant Client', list(df['SK_ID_CURR']))
selected_features = col1.multiselect('Variables', features)

#---------------------------------#
# Main panel
col2, col3 = st.columns((2,1))

col31, col32 = col3.columns((1,3))
image = Image.open('OC.png')
col31.image(image, width = 100, use_column_width=True)
image = Image.open('credit.png')
col32.image(image, width = 400, use_column_width=True)

col2.write("<span style='font-size: 50px; text-align: center;'> <b>Gestionnaire de crédit </b></span>", unsafe_allow_html=True)
col2.markdown(""" 
Ce tableau de bord prédit la probabilité qu'un client rembourse son crédit et classe sa demande en crédit accordé/refusé. 
""")

#-----------------------------------#
# Informations du client sélectionné
X = df.loc[df['SK_ID_CURR'] == selected_client, features]

# Call API model for prediction
api_url = 'http://localhost:5000/predict'

data = {'client_feat': X.to_json(orient="records")}

response = requests.get(api_url, json=data)

if response.status_code == 200:
    result = response.json()
    prediction = result['prediction']
    prediction_proba = result['prediction_proba']
    shap_values = json.loads(result['shap_values'])
    print('Score de prédiction pour client1 :', prediction_proba * 100)
else:
    print("Erreur lors de l'appel de l'API")
    
#-----------------------------------#
# Probabilité de remboursement + état de la demande
decision = "crédit accordé" if prediction == 0 else "crédit refusé" 
proba_remboursement = round(prediction_proba * 100)

col21, col22 = col2.columns((1,1))

col21.write("<span style='font-size: 24px'> <b>Client : <span style='color: gray; font-size:1.1em'> {} </span> </b></span>".format(str(selected_client)), unsafe_allow_html=True)
if prediction == 1 :
    col21.write("<span style='font-size: 24px'> <b>Probabilité de remboursement : \n  <span style='color: red; font-size:1.2em'> {} % </span> </b></span>".format(str(proba_remboursement)), unsafe_allow_html=True)
    col21.write("<span style='font-size: 24px'> <b> Etat de la demande : <span style='color: red; font-size:1.2em'> {} </span> </b></span>".format(decision), unsafe_allow_html=True) 
if prediction == 0 :
    col21.write("<span style='font-size: 24px'> <b>Probabilité de remboursement : <span style='color: blue; font-size:1.2em'> {} % </span> </b></span>".format(str(proba_remboursement)), unsafe_allow_html=True)
    col21.write("<span style='font-size: 24px'> <b> Etat de la demande : <span style='color: blue; font-size:1em'> {} </span> </b></span>".format(decision), unsafe_allow_html=True) 

#-------------------
# la bare du score
fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = proba_remboursement,
        number = {'suffix': "%"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 50, 'position': "top"},
        title = {'text': "Score client"},
        gauge = {
            #'shape': "bullet",
            'axis': {'range': [None, 100]},
            'threshold': {'line': {'color': "red", 'width': 0},
                          'thickness': 0.75,
                          'value': 50},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}],
            'bar': {'color': "red" if prediction == 1 else "blue" ,  'thickness': 0.5}}))
fig.update_layout(width=350, height=350)
col22.plotly_chart(fig)
st.write('---')

#-------------------------------#
col2, col3 = st.columns((2,1))

# Profil client
col2.header('Profil client: ' + str(selected_client))
col2.markdown(""" 
* <span style="color:red">Les variables en rouge **défavorisent** l'accord du crédit</span>.  
* <span style="color:blue"> Les variables en bleu  **favorisent** l'accord du crédit</span>. 
""", unsafe_allow_html=True)

explanation = shap.Explanation(values=np.array(shap_values["values"]),
                               base_values=shap_values["base_values"],
                               data=np.array(shap_values["data"]))
explanation.feature_names = features

fig, ax = plt.subplots()
ax.set_title('Feature importance based on SHAP values')
shap.plots.waterfall(explanation)
col2.pyplot(fig, bbox_inches='tight')

#---------------------------
# Feature importance globale
col3.header('Feature importance globale')
col3.markdown(""" Les variables les plus importantes pour le modèle de prédiction sont: 
* **Ext_source_1, Ext_source_2, Ext_source_3**: 3 scores normalisés d'une souce extérieur
* **DAYS_BIRTH** : - l'âge du client calculé en jour 
* **CLOSED_DAYS_CREDIT**: Combien de jours avant la demande actuelle le client a-t-il demandé un crédit au bureau de crédit
* **PREV_DAYS_DECISION_MIN**: Combien de jours avant la décision concernant les demandes précédentes
* **PREV_NAME_PRODUCT_TYPE_XNA_MEAN**: 
""", unsafe_allow_html=True)

image = Image.open('lgbm_importances.png')
col3.image(image, width = 200, use_column_width=True)
st.write('---')

#----------------------------------------
# Positionnement du client par rapport à l'ensemble de clients
col2.header("Positionnement du client par rapport à l'ensemble de clients")
col2.markdown("La ligne noire corréspond à la valeur du client séléctionné")

#---------------------
# histogramme des variables sélectionnées : Rouge pour les clients à risque et bleu pour les clients fiables
for feature in selected_features: 
    # The individual's value stored in a variable called "individual_value"
    individual_value = X[feature].values[0]

    # Create the histogram with marginal box plots
    fig = px.histogram(df,
                       x=feature,
                       color="TARGET_",
                       marginal="box",  # can be `box`, `violin`
                       hover_data=df.columns,
                       labels={"TARGET_": "Client"},
                       color_discrete_sequence=['rgba(255, 0, 0, 0.8)', 'rgba(0, 0, 255, 0.8)'])

    # Add a vertical line for the individual value
    fig.add_shape(
        type="line",
        x0=individual_value,
        x1=individual_value,
        yref='paper',  # Set yref to 'paper' to span the entire height of the figure
        y0=0,
        y1=1,
        line=dict(color="black", width=4, dash="dash")
    )

    # Update layout for better visibility of the vertical line
    fig.update_layout(
        showlegend=True,
        title_text="Client sélectionné",
        xaxis_title=feature,
        yaxis_title="Count",
        bargap=0.1
    )
    
    # Show the figure
    col2.plotly_chart(fig)

#--------------------------------
# plot 2D entre deux variables sélectionnées en fonction du score client du rouge au bleu.
# Le noir pour le client sélectionné. 

# Parcourir la liste pour générer tous les couples de features selectionnées
couples_feat = []
for i in range(len(selected_features)):
    for j in range(i+1, len(selected_features)):
        couple_feat = (selected_features[i], selected_features[j])
        couples_feat.append(couple_feat)
        
for feat in couples_feat: 
    fig = go.Figure(data=[go.Scatter(
        x=df[feat[0]], #
        y=df[feat[1]],
        mode='markers',
        marker=dict(
            color=df["score"],
            size=10,
            showscale=True,
            colorscale="RdBu",
            colorbar=dict(
                title="Score",
                tickmode="array",
                ticks="outside",
            ))
    )])
    
    fig.add_trace(go.Scatter(
    x=X[feat[0]],  # Coordonnée x du point noir
    y=X[feat[1]],  # Coordonnée y du point noir
    mode='markers',
    marker=dict(
        color='black',  # Couleur du point noir
        size=20),
    ))
    
    fig.update_layout(
        title="Scatter Plot",
        xaxis=dict(title=feat[0]),
        yaxis=dict(title=feat[1]),
        showlegend=False,
    )
    # Show the figure
    col2.plotly_chart(fig)
#----------------------------------------------

