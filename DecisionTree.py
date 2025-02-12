import pandas as pd
df = pd.read_csv(r"C:\Users\ASUS\Downloads\Iris (1).csv")
print(df.sample(4))
print(df["Species"].unique())
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]],df["Species"],test_size=0.3,shuffle=True)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="entropy", max_depth=6,min_samples_leaf=4)
model.fit(X_train,y_train)
unknown_data=[[5.5,4.6,1,2.4]]
print("\n"+model.predict(unknown_data))

from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(15,12))
tree.plot_tree(model,fontsize=10,feature_names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])
plt.show()