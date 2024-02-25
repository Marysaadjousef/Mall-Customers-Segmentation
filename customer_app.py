from sklearn import preprocessing
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

filename = 'Mall Customers.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    '<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Prediction")

with st.form("my_form"):
    Gender = st.number_input(label='Gender', step=0.001, format="%.6f")
    Age = st.number_input(label='Age', step=0.001, format="%.6f")
    Annual_Income = st.number_input(
        label='Annual Income (k$)', step=0.01, format="%.2f")
    Spending_Score = st.number_input(
        label='Spending Score (1-100)', step=0.01, format="%.2f")

    data = [[Gender, Age, Annual_Income, Spending_Score]]

    submitted = st.form_submit_button("Submit")

if submitted:
    clust = loaded_model.predict(data)[0]
    print('Data Belongs to Cluster', clust)

    cluster_df1 = df[df['Cluster'] == clust]
    plt.rcParams["figure.figsize"] = (20, 3)
    for c in cluster_df1.drop(['Cluster'], axis=1):
        fig, ax = plt.subplots()
        grid = sns.FacetGrid(cluster_df1, col='Cluster')
        grid = grid.map(plt.hist, c)
        plt.show()
        st.pyplot(figsize=(5, 5))