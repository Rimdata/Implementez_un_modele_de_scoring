import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import shap
import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.subplots import make_subplots
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
#st.title('Tableau de bord de gestion de crédit')
st.write("<span style='font-size: 50px; text-align: center;'> <b>Tableau de bord de gestion de crédit </b></span>", unsafe_allow_html=True)
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
shap_values = explainer(X_)[:, :, 0]
shap_values.data = X.values
col2.header('Feature Importance')

fig, ax = plt.subplots()
ax.set_title('Feature importance based on SHAP values')
#shap.summary_plot(shap_values, X)
shap.plots.waterfall(shap_values[0])
col2.pyplot(fig, bbox_inches='tight')
col2.write('---')


#-------------------
import plotly.graph_objects as go


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
