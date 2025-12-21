import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r"C:\Users\Acer\OneDrive\Pictures\Documents\AIML projects\loan.csv")

print(df.head())
print(df.isnull().sum())
print(df.info())


df['income'] = df['income'].fillna(df['income'].mean())
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].mean())

df = pd.get_dummies(df, columns=['employment', 'city'], drop_first=True)


X = df.drop('loan_amount', axis=1)
y = df['loan_amount']


model_columns = X.columns
joblib.dump(model_columns, 'loan_model_columns.pkl')


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print('---Linear Regression---')
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)


print('---Random Forest---')
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
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

print(results)


joblib.dump(rf, 'loan_amount_model.pkl')
print("Loan model saved")


loaded_model = joblib.load('loan_amount_model.pkl')
model_columns = joblib.load('loan_model_columns.pkl')

age = int(input("Enter age: "))
income = float(input("Enter monthly income: "))
years = int(input("Loan duration (years): "))
credit = float(input("Credit score: "))
employment = input("Employment (Salaried / Self-employed): ").title()
city = input("City (Kathmandu / Pokhara / Bhaktapur): ").title()


user_df = pd.DataFrame(0, index=[0], columns=model_columns)

user_df['age'] = age
user_df['income'] = income
user_df['loan_years'] = years
user_df['credit_score'] = credit

if f'employment_{employment}' in user_df.columns:
    user_df[f'employment_{employment}'] = 1

if f'city_{city}' in user_df.columns:
    user_df[f'city_{city}'] = 1


prediction = loaded_model.predict(user_df)
print(f"\nPredicted Loan Amount: NPR {prediction[0]:,.2f}")


plt.scatter(y_test, lr_pred)
plt.xlabel("Actual Loan Amount")
plt.ylabel("Predicted Loan Amount")
plt.title("Linear Regression: Actual vs Predicted Loan")
plt.show()

plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Loan Amount")
plt.ylabel("Predicted Loan Amount")
plt.title("Random Forest: Actual vs Predicted Loan")
plt.show()



