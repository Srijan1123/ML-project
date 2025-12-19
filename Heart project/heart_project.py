import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\Acer\Downloads\kaggle\heart.csv")
df['Exercise angina'] = df['Exercise angina'].map({1: 'Yes', 0: 'No'})
df = pd.get_dummies(df, columns=['Chest pain type', 'Exercise angina', 'Thallium'], drop_first=True)


X = df.drop("Heart Disease", axis=1)
y = df['Heart Disease'].map({'Presence':1, 'Absence':0})


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


df['Predicted_Heart_Disease'] = rf_model.predict(X)
df['Predicted_Heart_Disease'] = df['Predicted_Heart_Disease'].map({1: 'Presence', 0: 'Absence'})
print(df[['Heart Disease', 'Predicted_Heart_Disease']].head())


sns.set(style="whitegrid")

# Heart Disease Counts - Bar Chart
plt.figure(figsize=(6,4))
sns.countplot(x='Heart Disease', data=df, palette='Set2')
plt.title("Heart Disease Counts")
plt.show()

# Sex Distribution - Pie Chart
sex_counts = df['Sex'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(sex_counts, labels=['Male','Female'], autopct='%1.1f%%', colors=['skyblue','lightgreen'])
plt.title("Sex Distribution")
plt.show()

# Chest Pain Types - Bar Chart
chest_cols = [col for col in df.columns if 'Chest pain type_' in col]
chest_counts = df[chest_cols].sum()
plt.figure(figsize=(6,4))
sns.barplot(x=chest_counts.index, y=chest_counts.values, palette='Set1')
plt.title("Chest Pain Type Counts")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# ST depression vs Max HR - Line Chart
plt.figure(figsize=(6,4))
sns.lineplot(x='Max HR', y='ST depression', data=df, marker='o')
plt.title("ST Depression vs Max Heart Rate")
plt.xlabel("Max Heart Rate")
plt.ylabel("ST Depression")
plt.show()
