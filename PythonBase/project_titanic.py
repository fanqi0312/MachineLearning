import pandas #ipython notebook

titanic = pandas.read_csv("data/titanic_train.csv")


# Survived：获救 1是
# Pclass：仓位1，2，3
# Sex：
# Age：
# SibSp：同行同龄人亲人，如兄弟姐妹，夫妻
# Parch：带领的老人或小孩
# Ticket：票编号
# Fare：票价
# Cabin：住在哪里？还有很多缺失值。
# Embarked：上船位置

# 1.均值填充,Age缺失，714（共891）
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# 2.将字符映射为数值，如性别

# 显示几种可能性
print(titanic["Sex"].unique())
# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# 3. 映射上船地点
print(titanic["Embarked"].unique())

# 缺省值，用众数填充
titanic["Embarked"] = titanic["Embarked"].fillna('S')

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2




# 分析数据
print(titanic.describe())
print(titanic.head(10))


# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)