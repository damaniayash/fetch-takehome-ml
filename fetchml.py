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
st.markdown('We have added the `month`, `day of the month` and `time step` columns to help better understand the data')
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

X_train, X_test, y_train, y_test, X, y = pre_processing(df)

regressor = LinearRegression(learning_rate=0.000035, n_iters=1000000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
accu = r2_score(y_test, predictions)
print("Accuracy:", accu)

make_prediction(regressor,df)
st.pyplot(plot_regression_line(X_train, X_test, y_train, y_test, X, regressor))

st.pyplot(plot_predicted_mean(X_train, X_test, y_train, y_test, regressor, df))

#st.pyplot(plot_monthly_sum(regressor, df))
st.plotly_chart(plot_monthly_sum(regressor, df)[0], theme="streamlit", use_container_width=True)

st.write(plot_monthly_sum(regressor, df)[2])


