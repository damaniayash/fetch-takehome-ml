from linear_regression import LinearRegression
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from plotly import graph_objs as go
import plotly

def handle_input(csv_name):
    df = pd.read_csv(csv_name)
    df.columns = ['date','value']
    df.date = pd.to_datetime(df.date, format = '%Y-%m-%d')
    df['month'] = pd.DatetimeIndex(df.date).month
    df['day'] = pd.DatetimeIndex(df.date).day
    df['Time'] = np.arange(1,len(df.index)+1)
    df['Lag_1'] = df['value'].shift(1)
    df = df.fillna(0)
    return df

def plot_data(df):
    fig = px.line(df,x='date',y='value',markers=True)
    return fig

def pre_processing(df):
    X = np.array(df.Time).reshape(-1,1)
    y = np.array(df.value)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=3456
        )
    return X_train, X_test, y_train, y_test, X, y

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


def plot_regression_line(X_train, X_test, y_train, y_test, X, regressor):
    y_pred_line = regressor.predict(X)
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    return fig


def make_prediction(regressor,df):
    y_pred_line = regressor.predict(np.array(list(range(0,730))).reshape(-1,1))
    pred = pd.DataFrame(y_pred_line[365:730],pd.date_range(start='01/01/2022', end='31/12/2022'))
    pred = pred.reset_index()
    pred.columns = ['date', 'value']
    monthly_mean = pred.resample(rule='M', on='date')['value'].mean()
    monthly_mean21 = df.resample(rule='M', on='date')['value'].mean()
    monthly_sum = pred.resample(rule='M', on='date')['value'].sum()
    monthly_sum21 = df.resample(rule='M', on='date')['value'].sum()
    return monthly_mean,monthly_mean21,monthly_sum,monthly_sum21,y_pred_line

def plot_predicted_mean(X_train, X_test, y_train, y_test, regressor, df):
    monthly_mean,_,_,_,y_pred_line = make_prediction(regressor,df)
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    plt.plot(np.array(list(range(0,730))), y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.scatter(list(range(365,730,31)),monthly_mean,color='green')
    return fig

def plot_monthly_sum(regressor, df):
    monthly_mean,monthly_mean21,monthly_sum,monthly_sum21,y_pred_line = make_prediction(regressor,df)
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(list(range(365,730,31)),monthly_sum,color='blue')
    plt.scatter(list(range(0,365,31)),monthly_sum21,color='red')

    df21 = pd.DataFrame(pd.date_range(start='1/1/2021', periods=12, freq='M'), monthly_sum21).reset_index()

    df22 = pd.DataFrame(pd.date_range(start='1/1/2022', periods=12, freq='M'), monthly_sum).reset_index()
    df21.columns = ['value','time']
    df22.columns = ['value','time']
    fig1 = px.scatter(df21, x='time', y='value')
    fig2 = px.scatter(df22, x='time', y='value').update_traces(marker=dict(color='red'))

    fig = go.Figure(data = fig1.data + fig2.data)
    #st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    return fig, df21, df22
    