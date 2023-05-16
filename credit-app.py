import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import shap
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
#st.title('Tableau de bord de gestion de crédit')
st.write("<span style='font-size: 50px; text-align: center;'> <b>Gestionnaire de crédit </b></span>", unsafe_allow_html=True)
st.markdown(""" 
Ce tableau de bord prédit la probabilité qu'un client rembourse son crédit puis classe sa demande en crédit a accordé ou crédit refusé. 
""")
#---------------------------------#
# Page layout (continued)
## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((1,1))
#---------------------------------#
df = pd.read_csv("data/traited/df_credit_dash.csv")
features = list(df.iloc[:, 2:].columns)
#---------------------------------#
# Sidebar + Main panel
image = Image.open('pretadepenser.png')
col1.image(image, width = 200, use_column_width=True)

#col1.header('  Identifiant Client')
selected_client = col1.selectbox('Identifiant Client', list(df['SK_ID_CURR']))
selected_features = col1.multiselect('Variables', features)

#---------------------------------#

# Reads in saved classification model
load_clf = pickle.load(open('lgbm_model.pkl', 'rb'))

# Apply model to make predictions
X = df.loc[df['SK_ID_CURR'] == selected_client, df.columns[2:]]
prediction = load_clf.predict(X)
prediction_proba = load_clf.predict_proba(X)

decision = "crédit accordé" if prediction == 0 else "crédit refusé" 
proba_remboursement = round(prediction_proba[0][0] * 100)

col2.write("<span style='font-size: 24px'> <b>Client : <span style='color: gray; font-size:1.1em'> {} </span> </b></span>".format(str(selected_client)), unsafe_allow_html=True)

if prediction == 1 :
    col2.write("<span style='font-size: 24px'> <b>Probabilité de remboursement : <span style='color: red; font-size:1.5em'> {} % </span> </b></span>".format(str(proba_remboursement)), unsafe_allow_html=True)
    col2.write("<span style='font-size: 24px'> <b> Etat de la demande : <span style='color: red; font-size:1.5em'> {} </span> </b></span>".format(decision), unsafe_allow_html=True) 
if prediction == 0 :
    col2.write("<span style='font-size: 24px'> <b>Probabilité de remboursement : <span style='color: green; font-size:1.5em'> {} % </span> </b></span>".format(str(proba_remboursement)), unsafe_allow_html=True)
    col2.write("<span style='font-size: 24px'> <b> Etat de la demande : <span style='color: green; font-size:1.5em'> {} </span> </b></span>".format(decision), unsafe_allow_html=True) 


#-------------------------------#
# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.Explainer(load_clf["clf"])
X_ = pd.DataFrame(load_clf['scaler'].transform(X), columns = X.columns)
shap_values = explainer(X_)[:, :, 1]
shap_values.data = X.values
col2.header('Feature Importance')
col2.markdown(""" 
* <span style="color:red">Les variables en rouge **défavorisent** l'accord du crédit</span>.  
* <span style="color:blue"> Les variables en bleu  **favorisent** l'accord du crédit</span>. 
""", unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.set_title('Feature importance based on SHAP values')
#shap.summary_plot(shap_values, X)
shap.plots.waterfall(shap_values[0])
col2.pyplot(fig, bbox_inches='tight')
col2.write('---')

#-------------------
# la bare du score
fig = go.Figure(go.Indicator(
    mode = "number+gauge+delta", value = proba_remboursement,
    number = {'suffix': "%"},
    domain = {'x': [0, 1], 'y': [0, 1]},
    delta = {'reference': 50, 'position': "top"},
    title = {'text':"<b>Profit</b><br><span style='color: gray; font-size:0.8em'>U.S. $</span>", 'font': {"size": 14}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 100]},
        'threshold': {
            'line': {'color': "red", 'width': 0},
            'thickness': 0.75, 'value': 50},
        'bgcolor': "white",
        'steps': [
            {'range': [0, 50], 'color': "lightgray"},
            {'range': [50, 100], 'color': "gray"}],
        'bar': {'color': "red" if prediction == 1 else "green" ,  'thickness': 0.5}}))
fig.update_layout(width=400, height=300)
col3.plotly_chart(fig)

#----------------------------------------
# Positionnement du client par rapport à l'ensemble de clients
col2.header("Positionnement du client par rapport à l'ensemble de clients")
col2.markdown("La ligne noire corréspond à la valeur du client séléctionné")

df["TARGET_"] = df["TARGET"].apply(lambda x: "Client à Risque" if x else "Client fiable")


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
X_ = df.iloc[:, 2:-1]
df['score'] = load_clf.predict_proba(X_)[:, 0]

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
        size=10),
    ))
    
    fig.update_layout(
        title="Scatter Plot",
        xaxis=dict(title=feat[0]),
        yaxis=dict(title=feat[1]),
        showlegend=False,
    )
    # Show the figure
    col2.plotly_chart(fig)

