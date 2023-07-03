# 处理数据
import os

import pandas as pd
import ast
import tarfile
import cv2
import glob


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# # 1.用不到！！从EPIC_100_train.csv中读取指定列，放到new_EPIC_100_train.csv中，并从中选出all_nouns的值属于前20的行放到front20.csv中
# df = pd.read_csv('file/EPIC_100_train.csv')
# df['all_nouns'] = df['all_nouns'].apply(ast.literal_eval)
# df['all_noun_classes'] = df['all_noun_classes'].apply(ast.literal_eval)
# selected_columns = df.iloc[:, [0, 6, 7, 8, 11, 12, 13, 14]]
# # print(selected_columns.head(10))
# # selected_columns.to_csv('file/new_EPIC_100_train.csv', index=False)
#
# # 展开
# df_exp = selected_columns.explode(['all_noun_classes', 'all_nouns'])
# df_exp = df_exp.reset_index(drop=True)
# # print(df_exp.to_string())
#
# # 排序
# value_counts = df_exp['all_nouns'].value_counts().sort_values(ascending=False)
# # 获取排序前 20 位的值
# top_20_values = value_counts.head(20).index.tolist()
# print(top_20_values)
# # ['tap', 'plate', 'knife', 'pan', 'cupboard',
# # 'spoon', 'bowl', 'drawer', 'glass', 'sponge',
# # 'hand', 'lid', 'fridge', 'fork', 'spatula',
# # 'onion', 'dough', 'meat', 'cloth', 'container']
#
# # 保留属于排序前 20 的行，删除剩下的行
# df_filtered = df_exp[df_exp['all_nouns'].isin(top_20_values)]
#
# # 将结果输出到新的 CSV 文件
# df_filtered.to_csv('file/front20.csv', index=False)
#
# # P01_109, P02_108, P02_112, P02_129, P04_107中所有位于前二十的帧数总和 1840




# # 2.筛选属于特定文件夹中的行
# df = pd.read_csv("file/front20.csv")
# # selected_rows = df[df['narration_id'].str.startswith(('P01_109', 'P02_108', 'P02_112', 'P02_129', 'P04_107'))]
# selected_rows = df[df['narration_id'].str.startswith('P02_112')]  # ################################3
# # selected_rows = df[df['narration_id'].str.startswith(('P01_109', 'P02_108'))]
# # 给第一列后面增加.jpg字符串
# # selected_rows['narration_id'] = selected_rows['narration_id'].astype(str) + ".jpg"
# # print(selected_rows.to_string())
# print(selected_rows.shape[0])
# # 筛选需要的列
# selected_columns = selected_rows.iloc[:, [0, 1, 2, 3, 6, 7]]
# # 输出
# selected_columns.to_csv('file/P02_112.csv', index=False)  # ##############################



# # 3.把csv中stop_frame这一列从11074改成./frame_0000011074.jpg
# file_path = "file/P02_112.csv"  # ####################################
# # 读取CSV文件
# df = pd.read_csv(file_path)
# # 提取数字部分
# df['stop_frame'] = './frame_' + df['stop_frame'].astype(str).str.zfill(10) + '.jpg'
# # print(df['start_frame'])
# # 保存修改后的数据到CSV文件
# df.to_csv('file/P02_112(2).csv', index=False)  # #################################



# # 4.从tar压缩包中提取与CSV文件中stop_frame列的数据对应的照片
# tar_file_path = '/media/louise/UBUNTU 20_0/yy/P02_112.tar'  # ############################
# csv_file_path = 'file/P02_112(2).csv'  # ########################
# df = pd.read_csv(csv_file_path)
# output_folder = 'P02_112'  # ##########################
# os.makedirs(output_folder, exist_ok=True)
# # 获取start_frame列中的所有数据作为目标图片名列表
# target_image_names = df['stop_frame'].astype(str).tolist()
# number = 0
# # 提取匹配的照片
# with tarfile.open(tar_file_path, 'r') as tar_ref:
#     for member in tar_ref.getmembers():
#         if member.name in target_image_names:
#             number += 1
#             print(member.name)
#             # 提取照片到指定路径
#             tar_ref.extract(member, output_folder)
# print("提取完成" + " number= " + str(number))
# # 统计需要提取的图片个数
# # 去除重复数据，并返回剩下的数据个数
# unique_count = df['stop_frame'].drop_duplicates().count()
# print("需要提取的图片个数：", unique_count)



# # 5.把P02_108.csv的stop_frame列中的数按大小排列并输出 主要是为了检查提取出来的图片对不对
# csv_file_path = 'file/P02_112.csv'  # ###########################
# df = pd.read_csv(csv_file_path)
# # 按照 start_frame 列进行升序排序
# df_sorted = df.sort_values(by='stop_frame')
# print(df_sorted['stop_frame'])



# # 6.把P02_108.csv中stop_frame这一列从11074改成P02_108_11074.jpg并保存到P02_108(2).csv中
# file_path = "file/P02_112.csv"  # ###################################
# # 读取CSV文件
# df = pd.read_csv(file_path)
# # 提取数字部分
# df['stop_frame'] = 'P02_112_' + df['stop_frame'].astype(str) + '.jpg'  # ##############################
# # print(df['stop_frame'])
# # 保存修改后的数据到CSV文件
# df.to_csv('file/P02_112(2).csv', index=False)  # ###############################



# # 7.更改文件夹中图片的名称，全都改为P02_108_1.jpg的格式
# folder_path = 'P02_112'  # ########################33
# prefix = "P02_112_"  # ##########################3
# # 获取文件夹中的所有文件名
# files = os.listdir(folder_path)
# number = 0
# # 遍历文件名
# for filename in files:
#     number += 1
#     # 提取数字部分
#     digits = filename.split('_')[1]
#     digits = digits.lstrip('0')
#     print(digits)
#     # 构建新的文件名
#     new_filename = f'{prefix}{digits}'
#
#     # 构建原始文件路径和目标文件路径
#     old_path = os.path.join(folder_path, filename)
#     print("old_path:"+old_path)
#     new_path = os.path.join(folder_path, new_filename)
#     print("new_path:" + new_path)
#     # 重命名文件
#     os.rename(old_path, new_path)
#
# print("文件名修改完成 " + str(number))


# 将所有的（2）文件合成为一个
# 要合并的CSV文件路径
csv_files = glob.glob('/media/louise/UBUNTU 20_0/yy/DL/file/data_file/*.csv')

# 读取并合并CSV文件
merged_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
selected_columns = merged_df.iloc[:, [2, 4, 5]]

# 输出合并后的数据框
print(selected_columns)
print(len(selected_columns))

# 保存合并后的数据框为新的CSV文件
selected_columns.to_csv('file/label.csv', index=False)
