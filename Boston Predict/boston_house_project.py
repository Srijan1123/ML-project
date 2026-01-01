import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_csv(r"C:\Users\Acer\Desktop\AIML projects\CSV\boston_housing.csv")
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())


for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['MEDV'].fillna(df['MEDV'].mean(), inplace=True) 

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

df.hist(figsize=(10,6), bins=20)
plt.show()

X = df.drop('MEDV', axis=1)
y = df['MEDV']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)



dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)


param_grid = {
    'max_depth': [2, 4, 6, 8, None],
    'min_samples_leaf': [1, 2, 4],
    'n_estimators': [50, 100, 200]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='r2'
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print('\nBest hyperparameters:', grid.best_params_)


y_pred = best_model.predict(X_test)
print('\nR2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE:', np.sqrt(np.mean((y_test - y_pred)**2)))


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.show()


importance = best_model.feature_importances_
features = X.columns
sns.barplot(x=importance, y=features)
plt.title('Feature Importance')
plt.show()


joblib.dump(best_model, 'boston_rf_model.pkl')
joblib.dump(scaler, 'boston_scaler.pkl')


loaded_model = joblib.load('boston_rf_model.pkl')
loaded_scaler = joblib.load('boston_scaler.pkl')


#------- let's take an  Example:RM=6, LSTAT=12,  PTRATIO=18 ------------
new_data = pd.DataFrame([[6, 12, 18]], columns=X.columns)
new_data_scaled = loaded_scaler.transform(new_data)
predicted_price = loaded_model.predict(new_data_scaled)
print('Predicted House Price:', round(predicted_price[0], 2))
