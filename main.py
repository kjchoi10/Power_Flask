import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

# Data load
p1_power = pd.read_csv ('Plant_1_Generation_Data.csv')
p1_weather = pd.read_csv ('Plant_1_Weather_Sensor_Data.csv')
p2_power = pd.read_csv ('Plant_2_Generation_Data.csv')
p2_weather = pd.read_csv ('Plant_2_Weather_Sensor_Data.csv')

# Data cleaning
p1_power['DATE_TIME'] = pd.to_datetime(p1_power['DATE_TIME'])
p1_weather['DATE_TIME'] = pd.to_datetime(p1_weather['DATE_TIME'])

p1_combined = pd.merge(p1_power, p1_weather, on=['DATE_TIME', 'PLANT_ID'], how='inner')

p2_power['DATE_TIME'] = pd.to_datetime(p2_power['DATE_TIME'])
p2_weather['DATE_TIME'] = pd.to_datetime(p2_weather['DATE_TIME'])

p2_combined = pd.merge(p2_power, p2_weather, on=['DATE_TIME', 'PLANT_ID'], how='inner')

# Multi-linear regression
model = LinearRegression()
x, y = p1_combined[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']], p1_combined['DC_POWER']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=50)

results = model.fit(x_train,y_train)

prediction = results.predict(x_test)

print('Coefficients: \n', results.coef_)

print('Mean Square error: %.2f' % mean_squared_error(y_test, prediction))

print('R^2: %.2f' % r2_score(y_test, prediction))

# Manual Ridge
ridge_regr = Ridge(alpha=0.01)
result_ridge = ridge_regr.fit(x_train, y_train)
ridge_prediction = result_ridge.predict(x_test)

print('Coefficients: \n', result_ridge.coef_)

print('Mean Square error: %.2f' % mean_squared_error(y_test, ridge_prediction))

print('R^2: %.2f' % r2_score(y_test, ridge_prediction))

# CV Ridge
cv_regr = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10, 100, 1000], store_cv_values=True).fit(x_train, y_train)
cv_prediction = cv_regr.predict(x_test)

print('Alpha: \n', cv_regr.alpha_)

print('Mean Square error: %.2f' % mean_squared_error(y_test, cv_prediction))

print('R^2: %.2f' % r2_score(y_test, cv_prediction))





