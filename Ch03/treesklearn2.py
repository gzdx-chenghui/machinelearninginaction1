# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

''' 数据读入 '''
raw_train_data = pd.read_csv("datatree2.csv")
# print('raw_train_data>>>>', raw_train_data)
raw_train_data.to_csv('datatree22.csv', sep='\t', index=False, header=False)
data = []
labels = []
with open("datatree22.csv") as ifile:
    for line in ifile:
        tokens = line.strip().split('\t')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
x = np.array(data)
# print('x>>>>>>>>>>',x)
labels = np.array(labels)
print('labels>>>>>', labels)
y = np.zeros(labels.shape)
print('y>>>>>>>', y)

''' 标签转换为0/1 '''
y[labels == '>=160'] = 3
y[labels == '>=143.1AND<160'] = 2
y[labels == '<143.1'] = 1
print('y>>>>>>', y)

''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)

''' 把决策树结构写入文件 '''
with open("tree20160707.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

'''测试结果的打印'''
answer = clf.predict(x_test)
answer1 = clf.score(x_test, y_test)
print(x_train)
print(answer)
print(answer1)
print(y_train)
print(np.mean(answer == y_test))

'''准确率与召回率'''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
answer = clf.predict_proba(x)[:, 1]
print(classification_report(y, answer, target_names=['thin', 'fat']))
