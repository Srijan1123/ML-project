import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping



df = pd.read_csv(r"C:\Users\Acer\Desktop\AIML projects\CSV\Real Estate + Fraud detection.csv")
print(df.info())
print(df.head())
print(df.describe())
print(df.isnull().sum())


for col in ['Size_sqft', 'CrimeRate']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

df['Price'] = df['Price'] + np.random.normal(0, 2000, size=len(df))

print(df.isnull().sum())


for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())


df['FraudFlag'] = df['FraudFlag'].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)


x_reg = df.drop(['Price', 'FraudFlag'], axis=1)
y_reg = df['Price']

x_clf = df.drop(['Price', 'FraudFlag'], axis=1)
y_clf = df['FraudFlag']

xr_train, xr_test, yr_train, yr_test = train_test_split(x_reg, y_reg, test_size=0.2, random_state=42)
xc_train, xc_test, yc_train, yc_test = train_test_split(x_clf, y_clf, test_size=0.2, random_state=42)



scaler = StandardScaler()

xr_train = scaler.fit_transform(xr_train)
xr_test = scaler.transform(xr_test)

xc_train = scaler.fit_transform(xc_train)
xc_test = scaler.transform(xc_test)

print("\n--- Linear Regression ---")

lr = LinearRegression()
lr.fit(xr_train, yr_train)
lr_pred = lr.predict(xr_test)

print("MAE:", mean_absolute_error(yr_test, lr_pred))
print("R2:", r2_score(yr_test, lr_pred))

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

rf_reg = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(rf_reg, param_grid, cv=5)
grid_rf.fit(xr_train, yr_train)

best_rf = grid_rf.best_estimator_
pred_rf = best_rf.predict(xr_test)

print("Best RF Params:", grid_rf.best_params_)


print("\n--- Logistic Regression ---")
log = LogisticRegression(max_iter=1000)
log.fit(xc_train, yc_train)

log_pred = log.predict(xc_test)

print("Accuracy:", accuracy_score(yc_test, log_pred))
print(confusion_matrix(yc_test, log_pred))

print("\n--- Random Forest Classifier ---")
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(xc_train, yc_train)

rf_pred = rf_clf.predict(xc_test)
print("Accuracy:", accuracy_score(yc_test, rf_pred))


print("\n--- ANN Regression ---")
model_reg = Sequential([
    Dense(128, activation='relu', input_shape=(xr_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1)
])

model_reg.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

early = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_reg.fit(
    xr_train, yr_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early]
)



print("\n--- ANN Classification ---")
model_clf = Sequential([
    Dense(64, activation='relu', input_shape=(xc_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model_clf.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_clf.fit(
    xc_train, yc_train,
    epochs=30,
    validation_split=0.2,
    callbacks=[early]
)

importance = best_rf.feature_importances_

plt.figure(figsize=(8,6))
plt.barh(x_reg.columns, importance)
plt.title("Feature Importance")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()
