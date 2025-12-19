import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_csv(r"C:\Users\Acer\OneDrive\Pictures\Documents\AIML projects\House prediction.csv")

print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.head())

x = df[['sqft', 'rooms']]
y = df['price']

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
print('---Linear regression model---')
lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
lr_pred = lr_model.predict(x_test)

print('---Random forest model---')
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(x_train,y_train)
rf_pred = rf_model.predict(x_test)


results = pd.DataFrame({
    'Model':['Linear regression', 'Random forest'],
    'MAE':[
         mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred)
    ],
    
    'RMSE':[
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    ],
    
    'R2':[
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred)
    ]
})


plt.scatter(y_test, rf_pred)
plt.xlabel("Actual price")
plt.ylabel("predicted price")
plt.title("Actual price VS predicted price")
plt.show()


sqft = float(input("Enter square feet: "))
rooms = int(input("Enter number of rooms: "))

user_input = np.array([[sqft,rooms]])
predicted_price = rf_model.predict(user_input)
print(f"\nPredicted house price: NPR {predicted_price[0]:,.2f}")



joblib.dump(rf_model,'house_price_model.pkl')
print("\nModel saved as house_price_model.pkl")


loaded_model = joblib.load("house_price_model.pkl")
test_prediction = loaded_model.predict([[1200,3]])


print("\nTest Prediction (1200 sqft, 3 rooms):", test_prediction)