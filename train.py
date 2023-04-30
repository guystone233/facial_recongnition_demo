import os
import cv2
import numpy as np
from PIL import Image
 
IMAGE_PATH = 'images/' # 训练图片保存路径
recognizer = cv2.face.LBPHFaceRecognizer_create() # 创建LBPH识别器
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 可以在opencv安装目录下找到

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)] #遍历该目录下的图片，存放在列表中
    face_samples = []
    ids = []
 
    for image_path in image_paths:
        img = Image.open(image_path).convert('L') # "L"表示灰度模式，将彩色图像转换为灰度图像
        img_np = np.array(img, 'uint8') # 将img数据转换为数组形式，方便后续的处理
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg': # 如果图片的后缀不是jpg的话，跳过该图片
            continue
 
        id = int(os.path.split(image_path)[-1].split(".")[0]) # 获取图片的人脸对应的编号
        faces = detector.detectMultiScale(img_np) # 检测人脸
 
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y + h, x:x + w])
            ids.append(id)
    return face_samples, ids # 返回人脸数据和编号的列表，这两个列表的下标是一一对应的
 
 
faces, ids = get_images_and_labels(IMAGE_PATH) # 获取人脸数据和编号数据
recognizer.train(faces, np.array(ids)) # 训练，参数为人脸数据和编号数据
recognizer.save('trainer.yml') # 保存训练结果到trainer.yml文件
