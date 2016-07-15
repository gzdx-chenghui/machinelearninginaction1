

from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import numpy as np
#''''' 数据读入 '''
fr1 = open("trainData1.csv")
fr2 = open("trainData2.csv")
lenses1 = [inst.strip().split('\t') for inst in fr1.readlines()]
lenses2 = [inst.strip().split('\t') for inst in fr2.readlines()]
lensesLabels = ['Demand', 'OrderQu', 'FillRate', 'Sale', 'Salepre', 'Averagestock', 'Stock_salesRatio']
#
#''''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(lenses1, lenses2, test_size = 0.3)

#''''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)
#''''' 把决策树结构写入文件 '''
with open('treeout2.dot', 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
## ''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)
##'''''测试结果的打印'''
answer = clf.predict(x_train)
print(x_train)
print(answer)
print(y_train)
print(np.mean( answer == y_train))
##
##'''''准确率与召回率'''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
print(precision)
answer = clf.predict_proba(lenses1)[:,1]
print(classification_report(lenses2, answer, target_names = [1,2,3]))
