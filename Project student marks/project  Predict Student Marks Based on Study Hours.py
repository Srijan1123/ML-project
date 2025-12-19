import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# data
hours = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
marks = np.array([20,25,35,45,50,65,70,80,90])


#train test
x_train, x_test, y_train,y_test = train_test_split(
    hours,
    marks,
    test_size=0.3,
    random_state=42

)

#model train
model = LinearRegression()
model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)
print("Actual data:", y_test)
print("predicted:", y_pred)

pred_5 = model.predict([[5]])
pred_6_5 = model.predict([[6.5]])
pred_9 = model.predict([[9]])
print("Marks for 5 hours study:", pred_5[0])
print("Marks for 6.5 hour study:", pred_6_5[0])
print("marks for 9 hours study:", pred_9[0])


#evaluation
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)


#EDA(mean median)
print("mean marks:", np.mean(marks))
print("mean hours:", np.mean(hours))


#plot
plt.scatter(hours, marks, label=("Actual data"))
plt.plot(hours, model.predict(hours), label = "Predicted line")
plt.xlabel("hours")
plt.ylabel("marks")
plt.title("hours VS marks")
plt.legend()
plt.show()

corr = np.corrcoef(hours.flatten(), marks)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()