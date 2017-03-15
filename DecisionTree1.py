print(__doc__)
import numpy as np
#Create a random dataset
rng=np.random.RandomState(1)
x=np.sort(5*rng.rand(80,1),axis=0)
y=np.sin(x).ravel()
y[::5]+=3*(0.5-rng.rand(16))

#Fit regression model
from sklearn.tree import DecisionTreeRegressor

clf_1=DecisionTreeRegressor(max_depth=2)
clf_2=DecisionTreeRegressor(max_depth=5)
clf_1.fit(x,y)
clf_2.fit(x,y)

#Predict
X_test=np.arange(0.0,5.0,0.01)[:, np.newaxis]
y_1=clf_1.predict(X_test)
y_2=clf_2.predict(X_test)

#Plot the results
import  matplotlib.pyplot as plt

plt.figure()
plt.scatter(x,y,c="k",label="data")
plt.plot(X_test,y_1,c="g",label="max_depth=2",linewidth=2)
plt.plot(X_test,y_2,c="r",label="max_depth=5",linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()