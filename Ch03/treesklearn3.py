import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import ibm_db_dbi

# 接数DB2数据库


# con = ibm_db_dbi.connect(
#   "DATABASE=artdb;HOSTNAME=10.110.1.178;PORT=50000;PROTOCOL=TCPIP;UID=db2inst1;PWD=db2inst1;", "", "")
# sql = "select (case when t.NEED>=0 then t.NEED else 0 end)as  NEED,(case when  t.PURCH>=0 then t.PURCH else 0 end ) as  PURCH,(case when t.MZL>=0 then t.MZL else 0 end ) as MZL,(case when t.SOLD>=0 then t.SOLD else 0 end )as SOLD,(case when t.AMT>=0 then t.AMT else 0 end )as AMT,(case when t.CXB>=0 then t.CXB else 0 end ) as CXB,(case when t.PRI<201.4 then 'PRI<PRI3'  when t.PRI>=201.4 AND t.PRI<211.4 then '201.4<=PRI<211.4'  when t.PRI>=211.4 AND t.PRI<221.4 then '211.4<=PRI<221.4' when t.PRI>=221.4 AND t.PRI<230 then '221.4<=PRI<230' when t.PRI>=230 then 'PRI>PRI4' else '0' end ) as targetcalss from RIM_CUST_ITEM_DL_WEEK t where t.ITEM_ID='10001128'"
# results1 = pd.read_sql(sql, con)
# print(results1)
# results1.to_csv('datatree33.csv', sep='\t', index=False, header=False)
data = []
labels = []
with open("datatree33.csv") as ifile:
    for line in ifile:
        tokens = line.strip().split('\t')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
x = np.array(data)
print('x>>>>>>>>>>', x)
labels = np.array(labels)
print('labels>>>>>', labels)
y = np.zeros(labels.shape)
print('y>>>>>>>', y)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

''' 标签转换为0/1 '''
y[labels == 'PRI<PRI3'] = 1
y[labels == '201.4<=PRI<211.4'] = 2
y[labels == '211.4<=PRI<221.4'] = 3
y[labels == '221.4<=PRI<230'] = 4
y[labels == 'PRI>PRI4'] = 5
print('y>>>>>>', y)

'''使用SelectFromModel算法进行降维处理'''
clff = ExtraTreesClassifier()
clff = clff.fit(x, y)
model = SelectFromModel(clff, prefit=True)

x_new = model.transform(x)
print('x_new>>>>>>>>>', x_new)

''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = DecisionTreeClassifier(criterion='entropy')
'''使用随机森林算法构建模型'''
clfRandom = RandomForestClassifier(criterion='entropy')
# print(clf)
print('>>>>>>>>>>>>>>>>>>>>>>>>')
print('clfR', clfRandom)
# clf.fit(x_train, y_train)
clfRandom.fit(x_train, y_train)

''' 把决策树结构写入文件 '''
# with open("tree20160714.dot", 'w') as f:
#    f = export_graphviz(clf, out_file=f)
with open("tree20160714R.dot", 'w') as f:
    f = export_graphviz(clfRandom, out_file=f)

''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
# print(clf.feature_importances_)
print(clfRandom.feature_importances_)

'''测试结果的打印'''
# answer = clf.predict(x_test)
# answer1 = clf.score(x_test, y_test)
answerRandomPrerdict = clfRandom.predict(x_test)
answerRandom = clfRandom.score(x_test, y_test)
# print(x_train)
# print('answer>>>>>>', answer)
# print('answer1>>>>>', answer1)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print('answerRandom', answerRandom)
# print(y_train)
# print(np.mean(answer == y_test))

'''准确率与召回率'''
# precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
# answer = clf.predict_proba(x)[:, 1]
# print(classification_report(y, answer, target_names=['thin', 'fat']))
