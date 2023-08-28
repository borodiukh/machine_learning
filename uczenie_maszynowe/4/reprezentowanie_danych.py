import pandas as pd

alldata = pd.read_csv('../5/titanic.tsv', header=0, sep='\t')

print(alldata["Survived"])
print()

print(alldata["PassengerId"])
print()

print(alldata["Pclass"])
print()

# for machine learning we don't use column "Name"

# male 1
# female 0
alldata["Sex"] = alldata["Sex"].apply(lambda x: 1 if x == 'male' else 0)
print(alldata["Sex"])
print()

mean = alldata["Age"].mean()
print(mean)
alldata["Age"] = alldata["Age"].fillna(29)  # bo mean równa się 29
print(alldata["Age"])
print()

print(alldata["SibSp"])
print()

print(alldata["Parch"])
print()

# for machine learning we don't use column "Ticket"

print(alldata["Fare"])
print()

# for machine learning we don't use column "Cabin"

data_embarked = pd.get_dummies(alldata["Embarked"], columns=["Czy S", "Czy C", "Czy Q"])
print(data_embarked)
print()






