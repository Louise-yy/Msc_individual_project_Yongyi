# 处理图片数据
import os

import cv2
import pandas as pd
import tarfile


# 更改文件夹中图片的名称，全都改为P01_109_1的格式
# folder_path = "P02_108"  # 文件夹路径
# prefix = "P02_108_"  # 新的文件名前缀
#
# file_list = os.listdir(folder_path)  # 获取文件夹中的文件列表
#
# ext = '.jpg'
# for i in range(1, 501):
#     frame_number_string = str(i).zfill(10)
#     orig_file_name = f"frame_{frame_number_string}{ext}"
#     print("orig_file_name:"+orig_file_name)
#     new_file_name = f"{prefix}{i}{ext}"  # 构建新的文件名
#     print("new_file_name:"+new_file_name)
#     # old_file_path = os.path.join(folder_path, orig_file_name)  # 原始文件路径
#     # new_file_path = os.path.join(folder_path, new_file_name)  # 新文件路径
#     # os.rename(old_file_path, new_file_path)  # 重命名文件


# 展示某一个图片
# image = cv2.imread("P02_108/P02_108_447.jpg")
# cv2.imshow('Image', image)
# cv2.waitKey(0)  # 等待按下任意按键后关闭窗口
# cv2.destroyAllWindows()


# # 删除不存在于CSV文件中的图片
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
