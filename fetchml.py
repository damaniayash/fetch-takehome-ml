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
from helper import *


st.title('Predicting number of receipts using Linear Regression')

st.subheader("Lets Import the data first")
st.markdown('We have added the `month`, `day of the month`,`weekend` and `time step` columns to help better understand the data')
df = handle_input('data_daily.csv')
st.write(df,use_container_width=True)

st.subheader('Let us plot the data to see any visual trends')
st.plotly_chart(plot_data(df), use_container_width=True)
st.markdown(
    'Looking at the plot we can clearly see a linear relation between the `number of scanned receipts` and `time`. \
    We can also see daily fluctuation in the number of receipts, let us plot the month over month daily number of receiept\
    to see if we have some sort of pattern.'
    )
st.plotly_chart(plot_monthly_data(df), use_container_width=True)
st.markdown('Although there seems to be some relation between the same days over the month but it is not significant.\
            We will also test to see if more more upload receipts over the weekend vs normal weekdays')
st.plotly_chart(plot_data_weekends(df),use_container_width=True)
st.markdown('Again, we do not visual see any trend that more people are uploading receipts on the weekends compared\
            to week days.')
st.subheader('Model Finalization')
st.markdown(f'We belive that Linear Regression would best fit this data, coupled with the fact we are only expected to predict\
            the total monthly sales for the year 2022, the daily flucuations are not that important compared to the overall\
            linear trend. For the independent variable we cannot use date directly, so we will use time steps as \
            the independent variable. We will make a 80/20 split for training and test samples. After training the model on\
            the training data, we can visualize the regression line.')
X_train, X_test, y_train, y_test, X, y = pre_processing(df)

regressor = LinearRegression(learning_rate=0.000035, n_iters=1000000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

st.markdown(f'After training, we can use the trained weights `y = m*x + b` where `m` is the weight and `b` is the bias term\
            the values for m and b are `{regressor.weights[0]}`,`{round(regressor.bias)}`')


make_prediction(regressor,df)
st.plotly_chart(plot_regression_line(X_train, X_test, y_train, y_test, X, regressor))

st.markdown('We can then extrapolate the regression line to calculate the mean of each month for the year `2022` \
            represented by green marker')

st.plotly_chart(plot_predicted_mean(X_train, X_test, y_train, y_test, X,regressor, df))

st.plotly_chart(plot_monthly_sum(regressor, df)[0], theme="streamlit", use_container_width=True)
st.subheader('Predict Total Number of scanned receipts for each monnth in 2022')
st.write(plot_monthly_sum(regressor, df)[2])


