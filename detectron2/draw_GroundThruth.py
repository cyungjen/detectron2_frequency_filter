import json
import cv2
import os
from matplotlib import pyplot as plt

# 读取 COCO 标注文件
filename = 'Traffic_2560x1600_30'
with open(f"SFU_ground_truth/{filename}_val.json") as f:
    annotations = json.load(f)

# 获取类别信息
categories = {category['id']: category['name'] for category in annotations['categories']}

# 输出视频的参数
video_path = f'{filename}.mp4'
frame_size = (2560, 1600)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, 30, frame_size)

# 处理多张图片
for image_info in annotations['images']:
    image_id = image_info['id']
    image_path = f"datasets/{filename}/imgs/" + image_info['file_name']
    # 读取图片
    image = cv2.imread(image_path)
    image = cv2.resize(image, frame_size)  # 调整尺寸以匹配视频尺寸

    # 获取图片的所有标注
    image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

    # 绘制每个标注的边界框和类别
    for ann in image_annotations:
        bbox = ann['bbox']
        x, y, w, h = map(int, bbox)
        category_id = ann['category_id']
        category_name = categories[category_id]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    # 将处理后的图片添加到视频
    video.write(image)

# 释放视频写入对象
video.release()

print("视频已保存到", video_path)


# #//////////////////////////////////////////////////////////////////////
# import json
# import cv2
# import matplotlib.pyplot as plt

# # 读取 COCO 标注文件
# with open("SFU_ground_truth/RaceHorses_832x480_30_val.json") as f:
#     annotations = json.load(f)

# # 获取类别信息
# categories = {category['id']: category['name'] for category in annotations['categories']}

# # 选择一张图片（示例中使用第一张图片）
# image_info = annotations['images'][0]
# image_id = image_info['id']
# image_path = "datasets/RaceHorses_832x480_30/imgs/" + image_info['file_name']


# # 读取图片
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # 获取图片的所有标注
# image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

# # 绘制每个标注的边界框和类别
# for ann in image_annotations:
#     bbox = ann['bbox']  # 边界框 [x, y, width, height]
#     x, y, w, h = map(int, bbox)
#     category_id = ann['category_id']
#     category_name = categories[category_id]
    
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# # 显示图片
# plt.imshow(image)
# plt.axis('off')
# plt.show()
