import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import pandas as pd


df = pd.read_csv("models_pred.csv")

# 移除全错选项
# get the pred acc for this line
def get_acc(pred_csv, index):
    label = pred_csv.loc[index, 'label']
    acc = 0
    if pred_csv.loc[index, 'pred_0'] == label:
        acc += 0.2
    if pred_csv.loc[index, 'pred_1'] == label:
        acc += 0.2
    if pred_csv.loc[index, 'pred_2'] == label:
        acc += 0.2
    if pred_csv.loc[index, 'pred_3'] == label:
        acc += 0.2
    if pred_csv.loc[index, 'pred_4'] == label:
        acc += 0.2
    return round(acc,1)

pred_acc_record = {0:0, 0.2:0, 0.4:0, 0.6:0, 0.8:0, 1:0}
delete_index = []
for index in range(len(df)):
    acc = get_acc(df, index)
    pred_acc_record[acc] += 1
    # remove noise data
    if acc <= 0:
        delete_index.append(index)

df = df.drop(delete_index)
df = df.reset_index(drop=True)

X = np.array(df[["pred_0","pred_1","pred_2","pred_3","pred_4"]])
y = np.array(df[["label"]]).flatten()

#X, y = create_dataset()
# print(type(X),X.shape, type(y),y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.metrics import accuracy_score

print("start decision tree")
# decision tree
repeat_accs = []
repeat_time = []
for i in range(10):
    start = time.time()
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    repeat_accs.append(accuracy_score(decision_tree.predict(X_test), y_test))
    end = time.time()
    repeat_time.append(end - start)
mean_acc = np.mean(np.array(repeat_accs))
mean_time = np.mean(np.array(repeat_time))
print("decision tree acc:", mean_acc, "time", mean_time)


print("start random forest")
random_forest_accs = []
random_forest_times = []
# random forest
repeat_accs = []
repeat_time = []
for i in range(10):
    start = time.time()
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    repeat_accs.append(accuracy_score(random_forest.predict(X_test), y_test))
    end = time.time()
    repeat_time.append(end - start)
mean_acc = np.mean(np.array(repeat_accs))
mean_time = np.mean(np.array(repeat_time))
print("random forest acc:", mean_acc, "time", mean_time)

print("start Ada Boost")
ada_boost_accs = []
ada_boost_times = []
# Ada Boost
repeat_accs = []
repeat_time = []
for i in range(10):
    start = time.time()
    ada_boost = AdaBoostClassifier()
    ada_boost.fit(X_train, y_train)
    repeat_accs.append(accuracy_score(ada_boost.predict(X_test), y_test))
    end = time.time()
    repeat_time.append(end - start)
mean_acc = np.mean(np.array(repeat_accs))
mean_time = np.mean(np.array(repeat_time))
print("Ada Boost acc:", mean_acc, "time", mean_time)

print("start Logistic Regression")
log_re_accs = []
log_re_times = []
# Logistic Regression
repeat_accs = []
repeat_time = []
for i in range(10):
    start = time.time()
    log_re = LogisticRegression()
    log_re.fit(X_train, y_train)
    repeat_accs.append(accuracy_score(log_re.predict(X_test), y_test))
    end = time.time()
    repeat_time.append(end - start)
mean_acc = np.mean(np.array(repeat_accs))
mean_time = np.mean(np.array(repeat_time))
print("Logistic Regression acc:", mean_acc, "time", mean_time)

print("start Neural Network")
nn_accs = []
nn_times = []
# Neural Network
repeat_accs = []
repeat_time = []
for i in range(10):
    start = time.time()
    nn = MLPClassifier(max_iter=2000)
    nn.fit(X_train, y_train)
    repeat_accs.append(accuracy_score(nn.predict(X_test), y_test))
    end = time.time()
    repeat_time.append(end - start)
mean_acc = np.mean(np.array(repeat_accs))
mean_time = np.mean(np.array(repeat_time))
print("Neural Network acc:", mean_acc, "time", mean_time)

print("start SVM")
SVM_accs = []
SVM_times = []
# SVM
repeat_accs = []
repeat_time = []
for i in range(10):
    start = time.time()
    SVM = SVC()
    SVM.fit(X_train, y_train)
    repeat_accs.append(accuracy_score(SVM.predict(X_test), y_test))
    end = time.time()
    repeat_time.append(end - start)
mean_acc = np.mean(np.array(repeat_accs))
mean_time = np.mean(np.array(repeat_time))
print("SVM acc:", mean_acc, "time", mean_time)


'''
# acc plot
l1, = plt.plot(train_sizes, decision_accs, color="blue")
l2, = plt.plot(train_sizes, random_forest_accs, color="yellow")
l3, = plt.plot(train_sizes, ada_boost_accs, color="green")
l4, = plt.plot(train_sizes, log_re_accs, color="red")
l5, = plt.plot(train_sizes, nn_accs, color="fuchsia")
l6, = plt.plot(train_sizes, SVM_accs, color="darkgoldenrod")
plt.legend(handles=[l1, l2, l3, l4, l5, l6],
           labels=["Decision Tree", "Random Forest", "Ada Boost", "Logistic Regression", "Neural Network", "SVM"])
plt.title("accuracy")
plt.savefig("accuracy.png")
plt.show()

# time plot
l1, = plt.plot(train_sizes, decision_times, color="blue")
l2, = plt.plot(train_sizes, random_forest_times, color="yellow")
l3, = plt.plot(train_sizes, ada_boost_times, color="green")
l4, = plt.plot(train_sizes, log_re_times, color="red")
l5, = plt.plot(train_sizes, nn_times, color="fuchsia")
l6, = plt.plot(train_sizes, SVM_times, color="darkgoldenrod")
plt.legend(handles=[l1, l2, l3, l4, l5, l6],
           labels=["Decision Tree", "Random Forest", "Ada Boost", "Logistic Regression", "Neural Network", "SVM"])
plt.title("time")
plt.savefig("time.png")
plt.show()
'''