import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


alldata = pd.read_csv('../5/titanic.tsv', header=0, sep='\t')


alldata["Sex"] = alldata["Sex"].apply(lambda x: 1 if x == 'male' else 0)

mean = alldata["Age"].mean()
# print(mean)
alldata["Age"] = alldata["Age"].fillna(29)  # bo mean równa się 29


X = alldata[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
Y = alldata["Survived"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)
Y_predicted = logisticRegr.predict(X_test)


matrix = metrics.confusion_matrix(Y_test, Y_predicted)

print(f'Matrix of prediction {matrix}')


print(f'Accuracy: {metrics.accuracy_score(Y_test, Y_predicted)}')
print(f'Precision: {metrics.precision_score(Y_test, Y_predicted)}')
print(f'Recall: {metrics.recall_score(Y_test, Y_predicted)}')
square_error = metrics.mean_squared_error(Y_test, Y_predicted)
print(f'Mean square error of metric: {square_error}')

precision, recall, fscore, support = metrics.precision_recall_fscore_support(Y_test, Y_predicted, average='micro')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F-score: {fscore}')

score = logisticRegr.score(X_test, Y_test)
print(f'Model score: {score}')

print(metrics.classification_report(Y_test, Y_predicted))

# print(alldata["Survived"])
# print()
#
# print(alldata["PassengerId"])
# print()
#
# print(alldata["Pclass"])
# print()
#
# # for machine learning we don't use column "Name"
#
# # male 1
# # female 0
# alldata["Sex"] = alldata["Sex"].apply(lambda x: 1 if x == 'male' else 0)
# print(alldata["Sex"])
# print()
#
# mean = alldata["Age"].mean()
# print(mean)
# alldata["Age"] = alldata["Age"].fillna(29)  # bo mean równa się 29
# print(alldata["Age"])
# print()
#
# print(alldata["SibSp"])
# print()
#
# print(alldata["Parch"])
# print()
#
# # for machine learning we don't use column "Ticket"
#
# print(alldata["Fare"])
# print()
#
# # for machine learning we don't use column "Cabin"
#
# data_embarked = pd.get_dummies(alldata["Embarked"], columns=["Czy S", "Czy C", "Czy Q"])
# print(data_embarked)
# print()
#
#
#
#
#
#
