"""
泰坦尼克数据集
"""

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
#print(age_is_null
age_null_true = age[age_is_null]
#print(age_null_true
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
#print(good_ages
correct_mean_age = sum(good_ages) / len(good_ages)
print(correct_mean_age) # 29.6991176471

# 自动过滤缺失值
# missing data is so common that many pandas methods automatically filter for it
correct_mean_age = titanic_survival["Age"].mean()
print(correct_mean_age) #29.6991176471



print("================================ 关系")
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


# 仓位和生存概率的关系，np.mean均值
#index tells the method which column to group by
#values is the column that we want to apply the calculation to
#aggfunc specifies the calculation we want to perform
passenger_survival = titanic_survival.pivot_table(index="Pclass", values="Survived", aggfunc=np.mean)
print(passenger_survival)
# 1    0.629630
# 2    0.472826
# 3    0.242363

passenger_age = titanic_survival.pivot_table(index="Pclass", values="Age")
print(passenger_age)


#两个参数的关系 np.sum求和
port_stats = titanic_survival.pivot_table(index="Embarked", values=["Fare","Survived"], aggfunc=np.sum)
print(port_stats)
# Fare        Survived      Embarked
# C         10072.2962        93
# Q          1022.2543        30
# S         17439.3988       217


#specifying axis=1 or axis='columns' will drop any columns that have null values
drop_na_columns = titanic_survival.dropna(axis=1)
new_titanic_survival = titanic_survival.dropna(axis=0,subset=["Age", "Sex"])
#print(new_titanic_survival

# 指定数据的位置，获取数据
row_index_83_age = titanic_survival.loc[83,"Age"]
row_index_1000_pclass = titanic_survival.loc[766,"Pclass"]
print(row_index_83_age)
print(row_index_1000_pclass)


new_titanic_survival = titanic_survival.sort_values("Age",ascending=False)
print(new_titanic_survival[0:10])
itanic_reindexed = new_titanic_survival.reset_index(drop=True)
print(itanic_reindexed.loc[0:10])


# This function returns the hundredth item from a series
def hundredth_row(column):
    # Extract the hundredth item
    hundredth_item = column.iloc[99]
    return hundredth_item

# Return the hundredth item from each column
hundredth_row = titanic_survival.apply(hundredth_row)
print(hundredth_row)


def not_null_count(column):
    column_null = pd.isnull(column)
    null = column[column_null]
    return len(null)


column_null_count = titanic_survival.apply(not_null_count)
print(column_null_count)



#By passing in the axis=1 argument, we can use the DataFrame.apply() method to iterate over rows instead of columns.
def which_class(row):
    pclass = row['Pclass']
    if pd.isnull(pclass):
        return "Unknown"
    elif pclass == 1:
        return "First Class"
    elif pclass == 2:
        return "Second Class"
    elif pclass == 3:
        return "Third Class"

classes = titanic_survival.apply(which_class, axis=1)
print(classes)


# 是否为未成年
def is_minor(row):
    if row["Age"] < 18:
        return True
    else:
        return False

minors = titanic_survival.apply(is_minor, axis=1)
#print(minors

def generate_age_label(row):
    age = row["Age"]
    if pd.isnull(age):
        return "unknown"
    elif age < 18:
        return "minor"
    else:
        return "adult"

age_labels = titanic_survival.apply(generate_age_label, axis=1)
print(age_labels)


titanic_survival['age_labels'] = age_labels
age_group_survival = titanic_survival.pivot_table(index="age_labels", values="Survived")
print(age_group_survival)
# age_labels
# adult      0.381032
# minor      0.539823
# unknown    0.293785
# Name: Survived, dtype: float64