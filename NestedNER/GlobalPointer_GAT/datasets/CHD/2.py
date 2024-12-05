import pandas as pd

# 创建数据字典
data = {
    '实体类别': ['DIS', 'SYM', 'SIG', 'DUR', 'PHE', 'TER', 'MED', 'DRU', 'ANA', 'AEX', 'DEF', 'LAE', 'NUR', 'TRE', 'PHI'],
    '外层实体': [13666, 25110, 26320, 11634, 8371, 23306, 15533, 10201, 29051, 9960, 7626, 15604, 3495, 3405, 4157],
    '嵌套实体': [8225, 281, 6, 42, 1178, 10, 12, 20, 27152, 289, 252, 1397, 2, 670, 3]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 计算外层实体和嵌套实体的总数
total_outer = df['外层实体'].sum()
total_nested = df['嵌套实体'].sum()

# 计算百分比，保留两位小数
df['外层实体占比'] = (df['外层实体'] / total_outer * 100).round(2)
df['嵌套实体占比'] = (df['嵌套实体'] / total_nested * 100).round(2)

# 调整百分比，使其总和为100%
def adjust_percentages(percentages):
    rounded_percentages = percentages.round(2)
    total = rounded_percentages.sum()
    difference = 100 - total
    
    if difference != 0:
        adjustment_index = rounded_percentages.idxmax()  # 找到最大的百分比项进行调整
        rounded_percentages.at[adjustment_index] += difference
    
    return rounded_percentages

# 调整外层实体和嵌套实体的占比
df['外层实体占比'] = adjust_percentages(df['外层实体占比'])
df['嵌套实体占比'] = adjust_percentages(df['嵌套实体占比'])

# 转换为百分数形式
df['外层实体占比'] = df['外层实体占比'].astype(str) + '%'
df['嵌套实体占比'] = df['嵌套实体占比'].astype(str) + '%'

# 打印结果
print(df[['实体类别', '外层实体占比', '嵌套实体占比']])
