import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
    'Name': [
        'Braund, Mr. Owen Harris',
        'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
        'Heikkinen, Miss. Laina',
        'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
        'Allen, Mr. William Henry',
        'Moran, Mr. James',
        'McCarthy, Mr. Timothy J',
        'Palsson, Master. Gosta Leonard',
        'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
        'Nasser, Mrs. Nicholas (Adele Achem)'
    ],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
    'Age': [22, 38, 26, 35, 35, 27, 54, 2, 27, 14],
    'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
    'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
    'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '330877', '17463', '349909', '347742', '237736'],
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    'Cabin': [None, 'C85', None, 'C123', None, None, 'E46', None, None, None],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C']
}
df  = pd.DataFrame(data)
#df  =pd.read_csv(r"C:\Users\Acer\OneDrive\Desktop\AI.ML\titanicc.txt")


# Basic information
'''print(df.head())
print(df.info())
print(df.isnull().sum())


#summary
total_passenger = len(df)
print("Total passenger:", total_passenger)

total_survived = df['Survived'].sum()
print("Total survived:", total_survived)

total_survival_rate = (total_survived / total_passenger) * 100
print("Total survival rate:", total_survival_rate)


#gender distribution
gender_count = df['Sex'].value_counts()
print("Total gender count:", gender_count)


gender_survival_rate = df.groupby('Sex')['Survived'].mean()  * 100
print("Gender survibal rate:")
print(gender_survival_rate)


#class distribution
class_count = df['Pclass'].value_counts()
print("Passenger by class")
print(class_count)

class_survival_rate = df.groupby('Pclass')['Survived'].mean() * 100
print("survival rate by class")
print(class_survival_rate)


#Age group analysis
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,18,50,100], labels=['Child','Adult','Senior'])

age_survival_rate = df.groupby('AgeGroup', observed=True)['Survived'].mean() * 100
print("Survival Rate by Age Group:")
print(age_survival_rate)

#family sized and survival
df['Familysize'] = df['SibSp'] + df['Parch']
family_survived_rate = df.groupby('Familysize')['Survived'].mean() * 100
print(family_survived_rate)'''




#NOW LET'S PLOT EACH AND EVERYTHING HERE


  #1.. survival rate by gender
gender_rate = df.groupby('Sex')['Survived'].mean() * 100

sns.barplot(x=gender_rate.index, y=gender_rate.values)
plt.title("survival rate by gender")
plt.ylabel("survival %")
plt.show()



    #2. survival rate by class
class_rate = df.groupby('Pclass')['Survived'].mean() * 100

sns.barplot(x = class_rate.index, y=class_rate.values)
plt.title("Survival rate by Pclass")
plt.xlabel("class")    
plt.ylabel("Survival %")
plt.show()



    #3. survival rate by Age Group
age_rate = df.groupby('Age')['Survived'].mean() * 100

sns.barplot(x=age_rate.index, y = age_rate.values)
plt.title("Survival rate by age group")   
plt.xlabel("age") 
plt.ylabel("Survival")
plt.show()



    #4. family size survival rate
family_rate = df.groupby('SibSp')['Survived'].mean() * 100

sns.barplot(x=family_rate.index, y =family_rate.values)
plt.title("Family size survival rate")
plt.xlabel("Family Size")
plt.ylabel("Survival")
plt.show()


    #5. heatmap correlation
sns.heatmap(df[['Survived','Pclass','Age','Fare','SibSp']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()







