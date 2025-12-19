import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\Acer\OneDrive\Pictures\Documents\AIML projects\salary_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())


df['city'] = df['city'].fillna(df['city'].mode()[0])
df['skills_score'] = df['skills_score'].fillna(df['skills_score'].mean())
df['salary'] = df['salary'].fillna(df['salary'].mean())
df = pd.get_dummies(df, columns=['education', 'city'], drop_first=True)


x = df.drop('salary', axis=1)
y = df['salary']


# --- save training columns ---
model_columns = x.columns
joblib.dump(model_columns, 'model_columns.pkl')


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print('----Linear regression----')
lr = LinearRegression()
lr_model = lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

print("----Random forest model----")
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf = rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)


result = pd.DataFrame({
    'Model': ['Linear regression', 'Random forest'],
    'MAE': [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    ],
    'R2': [
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred)
    ]
})

print(result)

#---let's save the model here----
joblib.dump(rf, 'salary_model.pkl')
print("Model saved")

#---let's Load model and takes the user input here---
loaded_model = joblib.load('salary_model.pkl')
model_columns = joblib.load('model_columns.pkl')

experience = int(input("Enter your year of experience: "))
skills = float(input("Enter skills score: "))
education = input("Education (Bachelor / +2 / Masters): ")
city = input("Enter city (Kathmandu / Pokhara / Bhaktapur): ")

education = education.title()
city = city.title()

# --- create empty dataframe with trained columns ---
user_df = pd.DataFrame(0, index=[0], columns=model_columns)

user_df['experience'] = experience
user_df['skills_score'] = skills

if f'education_{education}' in user_df.columns:
    user_df[f'education_{education}'] = 1

if f'city_{city}' in user_df.columns:
    user_df[f'city_{city}'] = 1


prediction = loaded_model.predict(user_df)
print(f"\nPredicted salary: NPR {prediction[0]:,.2f}")

# ----let's plot here----------
#  for ------Linear Regression model ---------
plt.scatter(y_test, lr_pred, alpha=0.7)
plt.xlabel("Actual salary")
plt.ylabel("Predicted salary")
plt.title("Linear Regression: Actual vs Predicted Salary")
plt.show()

#-----------For Random Forest model ---------
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.xlabel("Actual salary")
plt.ylabel("Predicted salary")
plt.title("Random Forest: Actual vs Predicted Salary")
plt.show()
