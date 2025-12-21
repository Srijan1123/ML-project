import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv(r"C:\Users\Acer\OneDrive\Pictures\Documents\AIML projects\car_details.csv")
print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.head())

df['km_driven'] = df['km_driven'].fillna(df['km_driven'].mean())
df['price'] =  df['price'].fillna(df['price'].mean())

df = pd.get_dummies(df, columns=['fuel','transmission','owner'], drop_first=True)


x = df.drop('price', axis=1)
y = df['price']

model_columns = x.columns
joblib.dump(model_columns,'car_model_columns.pkl')


x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print('---Linear regression model')
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)


print('----Random forest model----')
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)

result = pd.DataFrame({
    'Model':['linear regression','Random forest'],
    'MAE':[
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred)
    ],
    
    'RMSE':[
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    ],
    
    'R2':[
        r2_score(y_test,lr_pred),
        r2_score(y_test, rf_pred)
    ]
    
})

print(result)


#---Let's save the model here---
joblib.dump(rf,'car_price_model.pkl')
print("Car price model saved")

# --- let's load model and user input
loaded_model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('car_model_columns.pkl')

year = int(input("Enter car year: "))
km = float(input("Enter car kilometer: "))
fuel = input("Enter fuel (Petrol / Diesel): ").title()
transmission = input("Transmission (Manual / Automatic): ").title()
owner = input("Owner (First / Second / Third): ").title()


user_df = pd.DataFrame(0, index=[0], columns=model_columns)

user_df['year'] = year
user_df['km_driven'] = km

if f'fuel_{fuel}' in user_df.columns:
    user_df[f'fuel_{fuel}'] = 1
    
if f'transmission_{transmission}' in user_df.columns:
    user_df[f'transmission_{transmission}'] = 1
    
if f'owner_{owner}' in user_df.columns:
    user_df[f'owner_{owner}'] = 1
    

prediction = loaded_model.predict(user_df)
print(f"\nPredicted car price: NPR {prediction[0]:,.2f}")


plt.scatter(y_test, lr_pred)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual price VS predicted price")
plt.show()

plt.scatter(y_test, rf_pred)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual price VS predicted price")
plt.show()
