import pandas

food_info = pandas.read_csv("data/food_info.csv")
print(type(food_info)) #<class 'pandas.core.frame.DataFrame'>
print(food_info.dtypes) # String 为object
# NDB_No               int64
# Shrt_Desc           object
# Water_(g)          float64

print("================================ 显示数据")
first_rows = food_info.head()
# 默认显示前5条
print(food_info.head())
# 显示前3条
print(food_info.head(3))
# 显示后2条
print(food_info.tail(2))
# 列名
print(food_info.columns)
# 显示维度
print(food_info.shape)


print("================================ 取值")
#pandas uses zero-indexing
#Series object representing the row at index 0.
# 取数据
print(food_info.loc[0])

# Series object representing the seventh row.
#food_info.loc[6]

# Will throw an error: "KeyError: 'the label [8620] is not in the [index]'"
#food_info.loc[8620]
#The object dtype is equivalent to a string in Python

# 常见类型
#object - For string values
#int - For integer values
#float - For float values
#datetime - For time values
#bool - For Boolean values
#print(food_info.dtypes)


print("================================ 切片")
# Returns a DataFrame containing the rows at indexes 3, 4, 5, and 6.
food_info.loc[3:6]

# Returns a DataFrame containing the rows at indexes 2, 5, and 10. Either of the following approaches will work.
# Method 1
two_five_ten = [2,5,10]
food_info.loc[two_five_ten]

# Method 2
food_info.loc[[2,5,10]]

# Series object representing the "NDB_No" column.
# 按照列名取值
ndb_col = food_info["NDB_No"]
#print ndb_col
# Alternatively, you can access a column by passing in a string variable.
#col_name = "NDB_No"
#ndb_col = food_info[col_name]

# 取多个列名
columns = ["Zinc_(mg)", "Copper_(mg)"]
zinc_copper = food_info[columns]
#print zinc_copper
#print zinc_copper
# Skipping the assignment.

#
zinc_copper = food_info[["Zinc_(mg)", "Copper_(mg)"]]

#print(food_info.columns)
#print(food_info.head(2))

# 列名转为为List
col_names = food_info.columns.tolist()
#print col_names
gram_columns = []

for c in col_names:
    if c.endswith("(g)"):
        gram_columns.append(c)
gram_df = food_info[gram_columns]
print(gram_df.head(3))


print("================================ 切片")

#print food_info["Iron_(mg)"]
#div_1000 = food_info["Iron_(mg)"] / 1000
#print div_1000
# Adds 100 to each value in the column and returns a Series object.
#add_100 = food_info["Iron_(mg)"] + 100

# Subtracts 100 from each value in the column and returns a Series object.
#sub_100 = food_info["Iron_(mg)"] - 100

# Multiplies each value in the column by 2 and returns a Series object.
#mult_2 = food_info["Iron_(mg)"]*2


print("================================ 计算")
# 单位转换
#It applies the arithmetic operator to the first value in both columns, the second value in both columns, and so on
water_energy = food_info["Water_(g)"] * food_info["Energ_Kcal"]
water_energy = food_info["Water_(g)"] * food_info["Energ_Kcal"]
iron_grams = food_info["Iron_(mg)"] / 1000
food_info["Iron_(g)"] = iron_grams


#Score=2×(Protein_(g))−0.75×(Lipid_Tot_(g))
weighted_protein = food_info["Protein_(g)"] * 2
weighted_fat = -0.75 * food_info["Lipid_Tot_(g)"]
initial_rating = weighted_protein + weighted_fat



# the "Vit_A_IU" column ranges from 0 to 100000, while the "Fiber_TD_(g)" column ranges from 0 to 79
#For certain calculations, columns like "Vit_A_IU" can have a greater effect on the result,
#due to the scale of the values
# The largest value in the "Energ_Kcal" column.
# 最大值
max_calories = food_info["Energ_Kcal"].max()
# Divide the values in "Energ_Kcal" by the largest value.

# 归一化
normalized_calories = food_info["Energ_Kcal"] / max_calories
normalized_protein = food_info["Protein_(g)"] / food_info["Protein_(g)"].max()
normalized_fat = food_info["Lipid_Tot_(g)"] / food_info["Lipid_Tot_(g)"].max()
food_info["Normalized_Protein"] = normalized_protein
food_info["Normalized_Fat"] = normalized_fat




#By default, pandas will sort the data by the column we specify in ascending order and return a new DataFrame
# Sorts the DataFrame in-place, rather than returning a new DataFrame.
#print food_info["Sodium_(mg)"]
# 排序
food_info.sort_values("Sodium_(mg)", inplace=True)
print(food_info["Sodium_(mg)"])
#Sorts by descending order, rather than ascending.

# 降序
food_info.sort_values("Sodium_(mg)", inplace=True, ascending=False)
print(food_info["Sodium_(mg)"])