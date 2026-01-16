import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv(r"C:\Users\Acer\Desktop\ML_API_Projects\NN csv\customer_chrun.csv")
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())

sns.histplot(df['tenure_months'])
plt.title("Tenure months")
plt.show()

sns.histplot(df['monthly_spend'])
plt.title("Monthly spend")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Feature collection")
plt.show()

df['age'].fillna(df['age'].mean(), inplace=True)
df['total_charges'].fillna(df['total_charges'].mode()[0], inplace=True)

for col in ['gender', 'contract_type', 'payment_method', 'internet_service']:
    df[col].fillna(df[col].mode()[0], inplace=True)

df = pd.get_dummies(
    df,
    columns=['gender', 'contract_type', 'payment_method', 'internet_service'],
    drop_first=True
)

x = df.drop(['churn', 'monthly_spend'], axis=1)
y_chrun = df['churn']
y_spend = df['monthly_spend']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y_chrun,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("Logistic Reegression")
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("Random forest classifier")
rf = RandomForestClassifier(
    random_state=42,
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5
)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

param_grid = {
    "max_depth": [2, 4, 6],
    "min_samples_leaf": [2, 5, 10],
    "n_estimators": [100, 200]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(x_train, y_train)
print("Best param:", grid.best_params_)

importance = rf.feature_importances_
features = x.columns

feat_imp = pd.DataFrame({
    'Features': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x='Importance', y='Features', data=feat_imp.head(10))
plt.title("Top factor affecting chrun")
plt.show()

joblib.dump(rf, "chrun_model.pkl")
joblib.dump(scaler, "scaler.pkl")

model = Sequential([
    Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


earlystop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True
)
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.title("Loss curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label="validation_accuracy")
plt.title("Accuracy curve")
plt.legend()
plt.show()

model.save("chrun_nn_model.h5")
loaded_model = tf.keras.models.load_model('chrun_nn_model.h5')

new_customer = np.array([x_test[0]])
new_customer_scaled = scaler.transform(new_customer)

chrun_prob = loaded_model.predict(new_customer_scaled)[0][0]
print("chrun probability:", chrun_prob)
