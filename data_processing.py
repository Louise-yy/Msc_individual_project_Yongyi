# 处理csv中的数据
import pandas as pd
import ast

pd.set_option('display.max_columns', None)

# # 读取0, 2, 11, 12, 13, 14列
# df = pd.read_csv('EPIC_100_train.csv')
# df['all_nouns'] = df['all_nouns'].apply(ast.literal_eval)
# df['all_noun_classes'] = df['all_noun_classes'].apply(ast.literal_eval)
# selected_columns = df.iloc[:, [0, 2, 11, 12, 13, 14]]
# # print(selected_columns.head(10))
# selected_columns.to_csv('new_EPIC_100_train.csv', index=False)

# # 展开
# df_exp = selected_columns.explode(['all_noun_classes', 'all_nouns'])
# df_exp = df_exp.reset_index(drop=True)
# print(df_exp.to_string())

# #排序
# value_counts = df_exp['all_nouns'].value_counts().sort_values(ascending=False)
# # 获取排序前 20 位的值
# top_20_values = value_counts.head(20).index.tolist()
# print(top_20_values)
#
# # 保留属于排序前 20 的行，删除剩下的行
# df_filtered = df_exp[df_exp['all_nouns'].isin(top_20_values)]
# print(df_filtered)
#
#
# # 将结果输出到新的 CSV 文件
# df_filtered.to_csv('front20.csv', index=False)


# P01_109, P02_108, P02_112, P02_129, P04_107中所有位于前二十的帧数总和 1840




# 筛选以指定数字开头的行
df = pd.read_csv("file/front20.csv")
# selected_rows = df[df['narration_id'].str.startswith(('P01_109', 'P02_108', 'P02_112', 'P02_129', 'P04_107'))]
selected_rows = df[df['narration_id'].str.startswith(('P01_109', 'P02_108'))]
selected_rows['narration_id'] = selected_rows['narration_id'].astype(str) + ".jpg"
print(selected_rows.to_string())
print(selected_rows.shape[0])

# 筛选列
selected_columns = selected_rows.iloc[:, [0, 2, 3]]


selected_columns.to_csv('label.csv', index=False)




# # 把csv中的名字前缀全去掉，只留下最后一个代表本视频序号的数字
# file_path = "P01_109.csv"  # 文件路径
#
# # 读取CSV文件
# df = pd.read_csv(file_path)
#
# # 提取数字部分
# df['narration_id'] = df['narration_id'].str.replace('P01_109_', '').str.replace('.jpg', '')
#
# # 保存修改后的数据到CSV文件
# df.to_csv('P01_109(2).csv', index=False)



