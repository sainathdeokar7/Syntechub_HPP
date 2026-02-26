# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = pd.read_csv("housing.csv")
# data.dropna(inplace=True)
# data.info()

# from sklearn.model_selection import train_test_split

# x = data.drop(['median_house_value'],axis=1)
# y = data['median_house_value']

# X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# train_data = X_train.join(y_train)
# train_data

# train_data.hist(figsize = (15, 8))
# sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")

# train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
# train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
# train_data['population'] = np.log(train_data['population'] + 1)
# train_data['households'] = np.log(train_data['households'] + 1)

# train_data .hist(figsize=(15,8))
# # train_data.ocean_proximity.value_count()
# train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
# train_data

# plt.figure(figsize=(15, 8))
# sns.scatterplot(x="lattitude", y="longitude", data=train_data, hue="median_house_value", palettee="coolwarm")

# train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
# train_data['household_rooms'] = train_data['total_rooms'] / train_data['household']

# from sklearn.liner_model import LinerRegression

# X_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']

# reg = LinerRegression()

# reg.fit(X_train, y_train)

# LinerRegression()

# test_data = X_test.join(y_test)

# test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
# test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
# test_data['population'] = np.log(test_data['population'] + 1)
# test_data['households'] = np.log(test_data['households'] + 1)

# test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)

# test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
# test_data['household_rooms'] = test_data['total_rooms'] / test_data['household']

# X_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']

# reg.score(X_test, y_test)

# 1️⃣ Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 2️⃣ Load dataset
data = pd.read_csv("housing.csv")  # path to your CSV

# 3️⃣ Initial exploration
print(data.head())
print(data.info())
print(data.describe())

# 4️⃣ Correlation heatmap (only numeric)
plt.figure(figsize=(12,8))
sns.heatmap(data.select_dtypes(include="number").corr(), annot=True, cmap="YlGnBu")
plt.title("Numeric Feature Correlation")
plt.show()

# 5️⃣ Preprocessing
# Drop rows with missing target (if any)
data = data.dropna(subset=["median_house_value"])

# For simplicity, drop rows with any missing values
data = data.dropna()

# Convert categorical columns using one-hot encoding
data = pd.get_dummies(data)

# 6️⃣ Train-test split
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lr = lin_reg.predict(X_test)
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

# 8️⃣ Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# 9️⃣ Compare predicted vs actual
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted House Value")
plt.title("Random Forest Predictions vs Actual")
plt.show()