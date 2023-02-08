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
    df["weekend"] = df.date.dt.day_name().isin(['Saturday', 'Sunday'])
    df = df.fillna(0)
    return df

def plot_data(df):
    fig = px.line(df,x='date',y='value',markers=True)
    return fig

def plot_monthly_data(df):
    fig = px.line(df,x='day',y='value',color='month')
    return fig

def plot_data_weekends(df):
    fig = px.line(df,x='date',y='value',color='weekend',markers=True)
    return fig

def pre_processing(df):
    X = np.array(df.Time).reshape(-1,1)
    y = np.array(df.value)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=3456, shuffle=False
        )
    return X_train, X_test, y_train, y_test, X, y

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


def plot_regression_line(X_train, X_test, y_train, y_test, X, regressor):
    y_pred_line = regressor.predict(X)
    train = pd.DataFrame(X_train,y_train).reset_index()
    test = pd.DataFrame(X_test,y_test).reset_index()
    pred = pd.DataFrame(X,y_pred_line).reset_index()
    train.columns = ['value','time']
    test.columns = ['value', 'time']
    pred.columns = ['value', 'time']
    fig1 = px.scatter(train, x='time', y='value')
    fig2 = px.scatter(test, x='time', y='value').update_traces(marker=dict(color='orange'))
    fig3 = px.line(pred, x='time',y='value').update_traces(line_color='black', line_width=5)
    fig = go.Figure(data = fig1.data + fig2.data + fig3.data).update_layout(
        xaxis_title="Time Step", yaxis_title="No of receipts")
    #plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    return fig


def plot_calculations(regressor,df):
    y_pred_line = regressor.predict(np.array(list(range(0,730))).reshape(-1,1))
    pred = pd.DataFrame(y_pred_line[365:730],pd.date_range(start='01/01/2022', end='31/12/2022'))
    pred = pred.reset_index()
    pred.columns = ['date', 'value']
    monthly_mean = pred.resample(rule='M', on='date')['value'].mean()
    monthly_mean21 = df.resample(rule='M', on='date')['value'].mean()
    monthly_sum = pred.resample(rule='M', on='date')['value'].sum()
    monthly_sum21 = df.resample(rule='M', on='date')['value'].sum()
    df21 = pd.DataFrame(pd.date_range(start='1/1/2021', periods=12, freq='M'), monthly_sum21).reset_index()
    df22 = pd.DataFrame(pd.date_range(start='1/1/2022', periods=12, freq='M'), monthly_sum).reset_index()
    df21.columns = ['value','time']
    df22.columns = ['value','time']
    df22['month'] = pd.DatetimeIndex(df22.time).month
    df22 = df22[['month','value','time']]
    return monthly_mean,monthly_mean21,monthly_sum,monthly_sum21,y_pred_line,df21,df22

# def plot_predicted_mean(X_train, X_test, y_train, y_test, X, regressor, df):
#     monthly_mean,_,_,_,y_pred_line,_,_ = plot_calculations(regressor,df)
#     fig = plt.figure(figsize=(8, 6))
#     plt.scatter(X_train, y_train)
#     plt.scatter(X_test, y_test)
#     plt.plot(np.array(list(range(0,730))), y_pred_line, color="black", linewidth=2, label="Prediction")
#     plt.scatter(list(range(365,730,31)),monthly_mean,color='green')
#     return fig

def plot_predicted_mean(X_train, X_test, y_train, y_test, X, regressor, df):
    fig1 = plot_regression_line(X_train, X_test, y_train, y_test, X, regressor)
    monthly_mean = plot_calculations(regressor,df)[0]
    pred = pd.DataFrame(list(range(365,730,31)), monthly_mean).reset_index()
    pred.columns = ['value','time']
    print(pred)
    fig2 = px.scatter(pred, x='time', y='value').update_traces(marker=dict(color='green'))
    fig = go.Figure(data = fig1.data + fig2.data).update_layout(
        xaxis_title="Time Step", yaxis_title="No of receipts")
    return fig


def plot_monthly_sum(regressor, df):
    _,_,_,_,_,df21,df22 = plot_calculations(regressor,df)
    fig1 = px.scatter(df21, x='time', y='value')
    fig2 = px.scatter(df22, x='time', y='value').update_traces(marker=dict(color='red'))

    fig = go.Figure(data = fig1.data + fig2.data)
    #st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    del df22['time']
    df22 = df22.style.background_gradient(axis=0)
    
    return fig, df21, df22
    