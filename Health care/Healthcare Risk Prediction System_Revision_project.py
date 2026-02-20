import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras 

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error,confusion_matrix, classification_report, mean_squared_error,r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
n = 100

df = pd.DataFrame({
 'Age':np.random.randint(18,70,n),
 'BMI': np.random.normal(27,5,n),
 'BloodPressure':np.random.normal(120,15,n),
 'Cholestrol':np.random.normal(200,20,n),
 'Smoking':np.random.choice(['Yes','No'],n,p=[0.3,0.7]),
 'PhysicalActivity':np.random.randint(1,5,n),
 'StressLevel':np.random.randint(1,10,n)
})

df['MedicalCost'] = (
 df['Age'] * 200 + 
 df['BMI'] * 300 +
 df['BloodPressure'] * 50 +
 df['Cholestrol'] * 20 +
 np.random.normal(0,1000,n)
)

risk_score = (
 df['BMI'] * 0.3 + 
 df['Smoking'].map({'Yes':1,"No":0}) * 5 + 
 df['StressLevel'] * 0.5 + 
 np.random.normal(0,2,n)
)

df['DiseaseRisk'] = (risk_score > risk_score.mean()).astype(int)

for col in ['BMI','BloodPressure', 'Cholestrol']:
 df.loc[df.sample(frac=0.5).index, col] = np.nan
 
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())

sns.countplot(x='DiseaseRisk', data = df)
plt.title("Disease Risk Distribution")
plt.show()

sns.histplot(df['MedicalCost'], kde=True)
plt.title("Medical cost distribution")
plt.show()

plt.figure(figsize=(80,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='d')
plt.show()

df.fillna(df.median(numeric_only=True), inplace=True)
df = pd.get_dummies(df, drop_first=True)

x_reg = df.drop(['MedicalCost','DiseaseRisk'], axis= 1)
y_reg = df['MedicalCost']

x_clf = df.drop(['MedicalCost','DiseaseRisk'])
y_clf = df['DiseaseRisk']

xr_train,xr_test,yr_train,yr_test = train_test_split(x_reg,y_reg, test_size=0.2, random_state= 42)
xc_train,xc_test,yc_train,yc_test = train_test_split(x_clf,y_clf, test_size=0.2, random_state= 42)

scaler = StandardScaler()
xr_train = scaler.fit_transform(xr_train)
xr_test = scaler.transform(xr_test)

xc_train = scaler.fit_transform(xc_train)
xc_test = scaler.transform(xc_test)

print("---- Regression models -----")

print("----Linear Regression ----")
lr = LinearRegression()
lr.fit(xr_train,yr_train)
lr_pred= lr.predict(xr_test)

print("MAE:",mean_squared_error(yr_test, lr_pred))
print("R2:",r2_score(yr_test,lr_pred))


param_gird = {
 'n_estimators':[100,200],
 'max_depth':[None,10]
}
grid = GridSearchCV(
 RandomForestRegressor(),
 param_gird,
 cv = 5
)

grid.fit(xr_train,yr_train)
print("Best params:",grid.best_params_ )
best_rf_reg = grid.best_estimator_

importance_reg = best_rf_reg.feature_importances_
features_name = x_reg.columns

importance_df_reg = pd.DataFrame({
 'Fetaures':features_name,
 'Importance': importance_reg
}).sort_values(by='Importance', ascending=False)
print(importance_df_reg)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Features', data = importance_df_reg)
plt.title("Fetaures Importance --- Regression")
plt.show()





print("----- Classifcations ---")

print("------ Logistic Rgression -------")
lr = LogisticRegression()
lr.fit(xc_train, yc_train)
lr_pred = lr.predict(xc_test)

print("Accuracy:", accuracy_score(yc_test, lr_pred))
print(confusion_matrix(yc_test, lr_pred))

print("--- Random Forest Classifier -----")
rf = RandomForestClassifier()
rf.fit(xc_train, yc_train)
rf_pred = rf.predict(xc_test)
print("Accuracy:", accuracy_score(yc_test, rf_pred))

importance_clf = rf.feature_importances_
importance_df_clf = pd.DataFrame({
 'Fetaures':x_clf.columns,
 'Importance':importance_clf

}).sort_values(by='Importance', ascending=False)
print(importance_df_clf)
plt.figure(figsize=(8,5))
sns.barplot(x= 'Importance', y = 'Fetaures', data = importance_df_clf)
plt.title("Features Importance --- clalssification")
plt.show()


model_reg = Sequential([
 Dense(64, activation = 'relu', input_shape = (xr_train.shape[1],)),
 BatchNormalization(),
 Dropout(0.3),
 
 Dense(32, activation = 'relu'),
 BatchNormalization(),
 Dropout(0.2),
 
 Dense(16, activation  ='softmax'),
 BatchNormalization(),
 Dropout(0.1),
 
 Dense(1)
])

model_reg.compile(
 optimizer = 'adam',
 loss = 'mse',
 metrics = ['accuracy']
)

model_reg.fit(
 xr_train,yr_train,
 epoches = 15,
 validation_split = 0.2
)


print("--- ANN for Classification -------")
model_clf = Sequential([
 Dense(64, activation = 'relu', input_shape = (xr_train.shape[1],)),
 BatchNormalization(),
 Dropout(0.3),
 
 Dense(32, activation = 'relu'),
 BatchNormalization(),
 Dropout(0.2),
 
 Dense(16, activation  ='softmax'),
 BatchNormalization(),
 Dropout(0.1),
 
 Dense(1, activation = 'sigmoid')
])

model_clf.compile(
 optimizer = 'adam',
 loss = 'binary_crossentropy',
 metrics = ['accuracy']
)

model_clf.fit(
 xc_train,yc_train,
 epoches = 15,
 validation_split = 0.2
)


joblib.dump(lr, 'linear_regression_model.pkl')
joblib.dump(best_rf_reg,'rf_regressor_model.pkl')
joblib.dump(rf, "rf_classifier_model.pkl")
joblib.dump(scaler,'scaler.pkl')

model_reg.save = ("ann_regression_model.h5")
model_clf.save = ("ann_classification_model.h5")
print("All model saved succesfully")










