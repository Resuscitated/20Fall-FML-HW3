import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from logitboost import LogitBoost
import os

arrays = []

f = open("./data/abalone.data")
line = f.readline()
while line:
    data = line.split(',')
    if data[0] == 'M':
        data[0] = '1'
        data.insert(1, '0')
        data.insert(2, '0')
    if data[0] == 'F':
        data[0] = '0'
        data.insert(1, '1')
        data.insert(2, '0')
    if data[0] == 'I':
        data[0] = '0'
        data.insert(1, '0')
        data.insert(2, '1')
    shape = len(data)
    data[shape-1]=data[shape-1][:-1]
    if int(data[shape - 1]) <= 9:
        data[shape - 1] = '-1'
    else:
        data[shape - 1] = '1'
    data.insert(0,data[shape-1])
    data=data[:-1]
    arrays.append(data)
    line = f.readline()
f.close()

train_data = arrays[:3133]
test_data = arrays[3133:]
train_data = np.array(train_data)
test_data = np.array(test_data)
train_data = train_data.astype(np.float)
test_data = test_data.astype(np.float)
train_data_label = train_data[:, 0]
train_data_attr = train_data[:, 1:]
test_data_label = test_data[:, 0]
test_data_attr = test_data[:, 1:]


start_T = 1
end_T = 10
fold = 10
size = len(train_data)
test_size = len(test_data)
one_size = int(size/fold)

stat_mean_ada=np.ndarray([end_T - start_T + 1])
stat_std_ada=np.ndarray([end_T - start_T + 1])
stat_mean_logit=np.ndarray([end_T - start_T + 1])
stat_std_logit=np.ndarray([end_T - start_T + 1])
test_mean_ada = np.ndarray([end_T - start_T + 1])
test_mean_logit = np.ndarray([end_T - start_T + 1])
for T in range(start_T, end_T + 1):
    err_ada = np.ndarray([fold])
    err_logit = np.ndarray([fold])
    err_ada_test = np.ndarray([fold])
    err_logit_test = np.ndarray([fold])
    for i in range(fold):
        label_train_CV = np.append(train_data_label[:i * one_size], train_data_label[(i + 1) * one_size:])
        attr_train_CV = np.append(train_data_attr[:i * one_size], train_data_attr[(i + 1) * one_size:], axis=0)
        label_test_CV = train_data_label[i * one_size:(i + 1) * one_size]
        attr_test_CV = train_data_attr[i * one_size:(i + 1) * one_size]

        ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=(T * 100))
        ada.fit(attr_train_CV, label_train_CV)

        logit = LogitBoost(n_estimators=(T * 100))
        logit.fit(attr_train_CV, label_train_CV)

        label_predict_ada_CV = ada.predict(attr_test_CV)
        label_predict_logit_CV = logit.predict(attr_test_CV)

        test_data_label_predict_ada = ada.predict(test_data_attr)
        test_data_label_predict_logit = logit.predict(test_data_attr)

        err_ada[i] = 1 - sum(label_predict_ada_CV == label_test_CV) / one_size
        err_logit[i] = 1 - sum(label_predict_logit_CV == label_test_CV) / one_size
        err_ada_test[i] = 1 - sum(test_data_label_predict_ada == test_data_label) / test_size
        err_logit_test[i] = 1 - sum(test_data_label_predict_logit == test_data_label) / test_size

    stat_mean_ada[T - start_T] = err_ada.mean()
    stat_std_ada[T - start_T] = np.std(err_ada)
    stat_mean_logit[T - start_T] = err_logit.mean()
    stat_std_logit[T - start_T] = np.std(err_logit)
    test_mean_ada[T - start_T] = err_ada_test.mean()
    test_mean_logit[T - start_T] = err_logit_test.mean()

x_axis = range(start_T*100, end_T*100 + 1, 100)
plt.plot(x_axis, stat_mean_ada)
plt.plot(x_axis, stat_mean_ada + stat_std_ada)
plt.plot(x_axis, stat_mean_ada - stat_std_ada)
plt.xticks(range(start_T*100, end_T*100 + 1, 100))
plt.legend(['mean','mean + std','mean - std'])
plt.xlabel('T')
plt.ylabel('Cross-validation Error')
plt.title('AdaBoost Train Error')
plt.show()


x_axis = range(start_T*100, end_T*100 + 1, 100)
plt.plot(x_axis, stat_mean_logit)
plt.plot(x_axis, stat_mean_logit + stat_std_logit)
plt.plot(x_axis, stat_mean_logit - stat_std_logit)
plt.xticks(range(start_T*100, end_T*100 + 1, 100))
plt.legend(['mean','mean + std','mean - std'])
plt.xlabel('T')
plt.ylabel('Cross-validation Error')
plt.title('LogisticBoost Train Error')
plt.show()

x_axis = range(start_T*100, end_T*100 + 1, 100)
plt.plot(x_axis, stat_mean_ada)
plt.plot(x_axis, stat_mean_logit)
plt.plot(x_axis, test_mean_ada)
plt.plot(x_axis, test_mean_logit)
plt.xticks(range(start_T*100, end_T*100 + 1, 100))
plt.legend(['AdaBoost Train Error', 'LogisticBoost Train Error','AdaBoost Test Error', 'LogisticBoost Test Error'])
plt.xlabel('T')
plt.ylabel('Error')
plt.title('Test and Train Error')
plt.show()