

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

df = pd.read_csv('elderlies_fall.csv',delimiter = ',')
df.head()
#mengecek data apakah terdapat NaN values atau tidak
df.isnull().values.any()
df.shape
df.dtypes
df_kol = df[['Time','SL','EEG','BP','HR','BC']]
X = np.array(df_kol.values)
X = abs(X)

#Buat label kelas dari kolom atribut 'Activity'
activity_labels = df[['Activity']]
Y = np.array(activity_labels.values)

#====================feature extraction======================
# Feature extraction
test = SelectKBest(score_func=chi2, k=6) #k=4 / k=6
myfit = test.fit(X, Y)

#Select features
features = myfit.transform(X)
print(features[0:1,:])

# Split dataset into training set and test set: 80% training and 20% test
X_train, X_test, Y_train, Y_test = train_test_split(features, activity_labels, test_size=0.2,random_state=3) 


#======================k-Nearest Neighboor=============================================

#Cari nilai k terbaik dgn memilih nilai akurasi tertinggi (antara k= 5-10)

k = []
acc = []
for i in range(2,8):
    kNN_model_wine = KNeighborsClassifier(n_neighbors= i)
    kNN_model_wine.fit(X_train, Y_train)
    Y_pred = kNN_model_wine.predict(X_test)
    print(i, metrics.accuracy_score(Y_test, Y_pred))
    k.append(i)
    acc.append(metrics.accuracy_score(Y_test, Y_pred))

plt.plot(k, acc)
plt.title('Plot nilai k dengan akurasi')
plt.ylabel('Akurasi')
plt.xlabel('nilai k')
plt.show()

start = datetime.now()     
kNN_model_activity = KNeighborsClassifier(n_neighbors= 16)

#Train the model using the training sets
kNN_model_activity.fit(X_train, Y_train)

end = datetime.now()
print("Running time to evaluate knn model: ", end-start)
#Predict the response for test dataset
Y_pred = kNN_model_activity.predict(X_test)

# Evaluasi model dengan menghitung akurasinya
# Model Accuracy, how often is the classifier correct?
print("Akurasi model klasifikasi KNN dgn k=16 :", metrics.accuracy_score(Y_test, Y_pred))



#====================klasifikasi N A I V E - B A Y E S====================

start = datetime.now()

NBC_model_activity = GaussianNB()

#Train the model using the training sets
NBC_model_activity.fit(X_train, Y_train)

end = datetime.now()

print("Running time to evaluate naive bayes model: ", end-start)

#Predict the response for test dataset
Y_pred = NBC_model_activity.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Akurasi model klasifikasi activity dgn NBC:",metrics.accuracy_score(Y_test, Y_pred))

"""
feature=4 - Akurasi model klasifikasi activity dgn NBC: 0.1455599633811413
feature=6 - Akurasi model klasifikasi activity dgn NBC: 0.1589868782422948
"""

#==========================Decision Tree==========================

start = datetime.now()
# Create/initiate the Decision Tree classifer model
DT_model_activity = tree.DecisionTreeClassifier(criterion='entropy', random_state=3)
# Train Decision Tree Classifer using the 70% of the dataset
DT_model_activity.fit(X_train,Y_train)

end = datetime.now()
print("Running time to evaluate decision tree model: ", end-start)
#Predict the response for test dataset
Y_pred = DT_model_activity.predict(X_test)

# Compute model accuray using test (30%) dataset, print the accuracy
print("Model accuracy DT:",metrics.accuracy_score(Y_test, Y_pred))

#======jika sudah memperoleh kriteria model yg akurasinya tinggi,
#buat model finalnya dengan menggunakan seluruh data input yg dimiliki======

#Visualisasi Decision Tree model 
"""
DT_model_remote_final = tree.DecisionTreeClassifier(criterion='entropy')
DT_model_remote_final.fit(X,Y)
dot_data = export_graphviz(DT_model_remote_final,feature_names=df_kol.columns, class_names=str(DT_model_remote_final.classes_), filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
#Create and save the graph of tree as image (PNG format)
graph.write_png("Dtree_remote_model.png")
"""


#====================Evaluasi dengan k-Fold====================
crossvalidation = KFold(n_splits=10, random_state=1, shuffle=True)
accuracy_kNN = cross_val_score(kNN_model_activity, X, Y, scoring="accuracy", cv=crossvalidation,  n_jobs=1)
avg_acc_kNN = np.mean(accuracy_kNN)
print("Acc kNN dengan k-Fold :"+str(avg_acc_kNN))

crossvalidation = KFold(n_splits=10, random_state=1, shuffle=True)
accuracy_NBC = cross_val_score(NBC_model_activity, X, Y, scoring="accuracy", cv=crossvalidation,  n_jobs=1)
avg_acc_NBC = np.mean(accuracy_NBC)
print("Acc NBC dengan k-Fold :"+str(avg_acc_NBC))

crossvalidation = KFold(n_splits=10, random_state=1, shuffle=True)
accuracy_DT = cross_val_score(DT_model_activity, X, Y, scoring="accuracy", cv=crossvalidation,  n_jobs=1)
avg_acc_DT = np.mean(accuracy_DT)
print("Acc DT dengan k-Fold :"+str(avg_acc_DT))













