import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\Acer\Desktop\AIML projects\CSV\Loan_default.csv")
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())


df.drop('LoanID', axis=1, inplace=True)


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

binary_cols = ['HasMortgage','HasDependents','HasCoSigner']
for col in binary_cols:
    df[col] = df[col].astype(str).str.strip().str.lower().map({'yes':1,'no':0})

df['Default'] = pd.to_numeric(df['Default'], errors='coerce')
df['Default'] = df['Default'].fillna(0).astype(int)
    

multi_cat_cols = ['Education','EmploymentType','MaritalStatus','LoanPurpose']
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)


x = df.drop('Default', axis=1)
y = df['Default']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print('\nPrint Logistic regression')
lo = LogisticRegression(max_iter=2000)
lo.fit(x_train, y_train)
lo_pred = lo.predict(x_test)

print('\nAccuracy of Logistic model:', accuracy_score(y_test, lo_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, lo_pred))
print('\nClassification Report:\n', classification_report(y_test, lo_pred))


print('\nRandom forest classifier')
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

print('\nAccuracy of Random Forest Classifier:', accuracy_score(y_test, rf_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, rf_pred))
print('\nClassification Report:\n', classification_report(y_test, rf_pred))


importance = rf.feature_importances_
feat_imp = pd.DataFrame({
    
    'Features': x.columns, 
    'Importance': importance,
    }).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Features', data=feat_imp, palette='viridis')
plt.title('Feature Importance - Random Forest')
plt.show()


df['Predicted_Default'] = rf.predict(x)
df['Predicted_Default'] = df['Predicted_Default'].map({1:'Default', 0:'No Default'})
print(df[['Default','Predicted_Default']].head())



#-------Loan default counts--------
plt.figure(figsize=(6,4))
sns.countplot(x='Predicted_Default', data=df, palette='Set2')
plt.title('Predicted Loan Default Counts')
plt.show()

#-------- Credit Score vs Predicted Default--------
plt.figure(figsize=(6,4))
sns.barplot(x='Predicted_Default', y='CreditScore', data=df, palette='Set1')
plt.title('Credit Score vs Predicted Default')
plt.show()

# --------Income vs Loan Amount - Scatter--------
plt.figure(figsize=(6,4))
sns.scatterplot(x='Income', y='LoanAmount', hue='Predicted_Default', data=df, palette='coolwarm')
plt.title('Income vs Loan Amount')
plt.show()
