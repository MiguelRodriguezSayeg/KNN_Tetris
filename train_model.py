import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("dataset.csv")
knn = KNeighborsClassifier(n_neighbors=4)

target = df.loc[:, df.columns != 'Label'].values.tolist()
label = df['Label']

x_train, x_test, y_train, y_test = train_test_split(target,label,train_size=0.8,test_size=0.2)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(y_pred)

print(x_test[0])

print(accuracy_score(y_test,y_pred))


