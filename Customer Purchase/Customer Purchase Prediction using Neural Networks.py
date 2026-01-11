import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv(r"C:\Users\Acer\Desktop\ML_API_Projects\NN csv\Customer purchase.csv")

print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())




num_cols = ['age', 'annual_income', 'time_on_site', 'avg_session_time']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = ['gender', 'device', 'location', 'discount_used']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['engagement_score'] = df['pages_visited'] * df['time_on_site'] 
df['loyalty_score'] = df['previous_purchases'] * df['avg_session_time']


le = LabelEncoder()
for col in ['gender', 'location', 'discount_used', 'device']:
    df[col] = le.fit_transform(df[col])


X = df.drop('purchase', axis=1)
y = df['purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


NN = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)


nn_pred = (model.predict(X_test) > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, nn_pred))
print(classification_report(y_test, nn_pred))


#---- Here let's visualize LOSS & ACCURACY ---------

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(NN.history['accuracy'], label='Train Accuracy')
plt.plot(NN.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Curve")

plt.subplot(1,2,2)
plt.plot(NN.history['loss'], label='Train Loss')
plt.plot(NN.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Curve")

plt.show()


model.save("customer_purchase_nn_model.h5")
joblib.dump(scaler, "customer_purchase_scaler.pkl")

print("Model and scaler saved successfully")
