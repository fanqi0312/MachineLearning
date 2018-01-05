import pandas as pd
import numpy as np
titanic_survival = pd.read_csv("data/titanic_train.csv")
titanic_survival.head()

#The Pandas library uses NaN, which stands for "not a number", to indicate a missing value.
#we can use the pandas.isnull() function which takes a pandas series and returns a series of True and False values
age = titanic_survival["Age"]
#print(age.loc[0:10])

# 是否为缺失值
age_is_null = pd.isnull(age)
#print age_is_null
age_null_true = age[age_is_null]
#print age_null_true
# 177个缺失值
age_null_count = len(age_null_true)
print(age_null_count)

# 有缺失值，则无法求平均年龄
#The result of this is that mean_age would be nan. This is because any calculations we do with a null value also result in a null value
mean_age = sum(titanic_survival["Age"]) / len(titanic_survival["Age"])
print(mean_age) #nan

# 排除缺失值后再计算
#we have to filter out the missing values before we calculate the mean.
good_ages = titanic_survival["Age"][age_is_null == False]
#print good_ages
correct_mean_age = sum(good_ages) / len(good_ages)
print(correct_mean_age) # 29.6991176471

# 自动过滤缺失值
# missing data is so common that many pandas methods automatically filter for it
correct_mean_age = titanic_survival["Age"].mean()
print(correct_mean_age) #29.6991176471

# 每个仓位的平均价格
#mean fare for each class
passenger_classes = [1, 2, 3]
fares_by_class = {}
for this_class in passenger_classes:
    pclass_rows = titanic_survival[titanic_survival["Pclass"] == this_class]
    pclass_fares = pclass_rows["Fare"]
    fare_for_class = pclass_fares.mean()
    fares_by_class[this_class] = fare_for_class
print(fares_by_class) # {1: 84.15468749999992, 2: 20.66218315217391, 3: 13.675550101832997}