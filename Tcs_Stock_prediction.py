import numpy as np
import pandas as pd

data_set=pd.read_csv("TCS_stock.csv")
data_set.dropna(inplace=True)

data_set["Date"]=pd.to_datetime(data_set["Date"])
data_set.set_index("Date",inplace=True)

x=data_set.loc[:,["Open","High","Low","Close","Volume"]].values
y=data_set["Adj Close"].values
y=y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

accuracy=lin_reg.score(x,y)
print(accuracy)