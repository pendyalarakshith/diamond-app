import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

st.title("WebApp Using Streamlit")
st.image("streamlit.png",width=300)
st.title("Case study on diamond dataset")
data = pd.read_csv("Diamonds.csv")
st.write("Shape of Dataset",data.shape)
menu = st.sidebar.radio("Menu",["Home","Prediction Price"])
if menu == "Home":
    st.image("diamonds.png",width=500)
    st.header("Tabular Data of a sales")
    if st.checkbox("Tabular Data"):
        st.table(data.head())
    st.header("Statistical summary of a dataframe")
    if st.checkbox("Statistics"):
        st.table(data.describe())
    if st.header("Coorelation Graph"):
        fig,ax = plt.subplots(figsize=(5,2.5))
        numeric_data = data.select_dtypes(include=["float", "int"])
        sns.heatmap(numeric_data.corr(),annot=True,cmap="coolwarm")
        st.pyplot(fig)
    st.title("Graphs")
    graph = st.selectbox("Different types of graphs",["Scatter plot","Bar Graph","Histogram"])
    if graph=="Scatter plot":
        value = st.slider("Filter data using carat",0,6)
        data = data.loc[data["carat"]>=value]
        fig,ax = plt.subplots(figsize=(10,5))
        sns.scatterplot(data=data,x="carat",y="price",hue="cut")
        st.pyplot(fig)
    if graph == "Bar Graph":
        fig,ax = plt.subplots(figsize=(3.5,2))
        sns.barplot(x="cut",y=data.cut.index,data=data)
        st.pyplot(fig)
    if graph == "Histogram":
        fig,ax = plt.subplots(figsize=(5,3))
        sns.histplot(data.price,kde=True)
        st.pyplot(fig)
if menu == "Prediction Price":
    st.title("Prediction price of a diamond")

    from sklearn.linear_model import LinearRegression
    lr=LinearRegression()
    x=np.array(data["carat"]).reshape(-1,1)
    y=np.array(data["price"]).reshape(-1,1)
    lr.fit(x,y)
    value = st.number_input("Carat",0.20,5.01,step=0.15)
    value = np.array(value).reshape(1,-1)
    prediction = lr.predict(value)[0]
    if st.button("Price Prediction($)"):
        st.write(f"{prediction}")