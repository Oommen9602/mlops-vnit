import pandas as pd
df=pd.DataFrame({"Height":[5.1,5.7,5.5,6,6.3,5.4,5.2,5.4,5.8,5.7,5.3],
                 "Shoe_Size":[6,8,7,10,9,5,6,7,8,8,7],
                 "Gender":[1,0,0,0,0,1,1,1,0,0,1]})
import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.scatter(df[df["Gender"]==1]["Height"],df[df["Gender"]==1]["Shoe_Size"],color="red",label="Female")
ax.scatter(df[df["Gender"]==0]["Height"],df[df["Gender"]==0]["Shoe_Size"],color="blue",label="Male")
ax.legend()
ax.set_xlabel("Height")
ax.set_ylabel("Shoe_Size")
plt.show()
X = df[["Height","Shoe_Size"]] #input features
Y = df["Gender"] #target variable
from sklearn.linear_model import LogisticRegression
#initialize the class method
model = LogisticRegression()
model.fit(X,Y)
y_pred = model.predict(X)
y_pred)