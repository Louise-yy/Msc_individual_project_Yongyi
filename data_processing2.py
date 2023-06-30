# 处理图片数据
import os
import pandas as pd

# 更改文件夹中图片的名称，全都改为P01_109_1的格式
folder_path = "P02_108"  # 文件夹路径
prefix = "P02_108_"  # 新的文件名前缀

file_list = os.listdir(folder_path)  # 获取文件夹中的文件列表

for i, file_name in enumerate(file_list):
    _, ext = os.path.splitext(file_name)  # 获取文件名和扩展名
    new_file_name = f"{prefix}{i+1}{ext}"  # 构建新的文件名
    old_file_path = os.path.join(folder_path, file_name)  # 原始文件路径
    new_file_path = os.path.join(folder_path, new_file_name)  # 新文件路径
    os.rename(old_file_path, new_file_path)  # 重命名文件

ext = '.jpg'
for i in range(500):
    # _, ext = os.path.splitext(file_name)  # 获取文件名和扩展名
    frame_number_string = str(i).zfill(10)
    orig_file_name = f"frame_{frame_number_string}{ext}"
    print(orig_file_name)
    new_file_name = f"{prefix}{i+1}{ext}"  # 构建新的文件名
    print(new_file_name)
    old_file_path = os.path.join(folder_path, orig_file_name)  # 原始文件路径
    new_file_path = os.path.join(folder_path, new_file_name)  # 新文件路径
    os.rename(old_file_path, new_file_path)  # 重命名文件



# # 按照csv中的顺序在文件夹中排列图片
# folder_path = "P02_108"  # 文件夹路径
# csv_file = "P02_108.csv"  # CSV文件路径
#
# # 读取CSV文件
# df = pd.read_csv(csv_file)
#
# # 获取CSV文件第一列的图片名称列表
# csv_image_names = df['narration_id'].tolist()
#
# print(csv_image_names)
#
# # 遍历文件夹中的文件
# for file_name in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, file_name)
#     if file_name not in csv_image_names:
#         # 删除不存在于CSV文件中的图片
#         os.remove(file_path)
#         print(f"删除文件: {file_name}")




# # 输出CSV文件中存在但文件夹中不存在的图片的narration_id：
#
# folder_path = "P01_109"  # 文件夹路径
# csv_file = "P01_109.csv"  # CSV文件路径
#
# # 读取CSV文件
# df = pd.read_csv(csv_file)
#
# # 获取CSV文件中的narration_id列数据
# csv_narration_ids = df['narration_id'].tolist()
#
# # 遍历CSV文件中的narration_id
# for narration_id in csv_narration_ids:
#     file_path = os.path.join(folder_path, f"{narration_id}")
#     if not os.path.exists(file_path):
#         print(f"文件夹中不存在的图片的narration_id: {narration_id}")
