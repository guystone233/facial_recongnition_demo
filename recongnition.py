import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create() # cv2.face需要安装opencv-contrib-python
recognizer.read('trainer.yml') # 读取训练结果
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # 可以在opencv安装目录下找到
font = cv2.FONT_HERSHEY_SIMPLEX # 字体，可根据需要自行更改
idnum = 0

cam = cv2.VideoCapture(0)
cam.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')) # 设置视频格式为MJPG，6表示CV_CAP_PROP_FOURCC，参考：https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get

# 用于人脸检测的最小宽度和高度，cam.get(3)和cam.get(4)分别表示视频的宽度和高度，这里设置为视频宽度和高度的10%
minW = 0.1 * cam.get(3) 
minH = 0.1 * cam.get(4) 

user_info = {}
#用户id和楼层号存储在user_info.txt中按行存储，格式为：id floor，这里将其读取到user_info字典中
with open('user_info.txt', 'r') as f:
    for line in f.readlines():
        user_info[line.split()[0]] = int(line.split()[1])

while True:
    ret, img = cam.read() # 读取视频帧
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换为灰度图像
    #face_detector.detectMultiScale参数说明
    #第一个参数是要检测的图片，一般是灰度图像加快检测速度；
    #第二个是scaleFactor，表示每次图像尺寸减小的比例,可以理解为相机的X倍镜。它是用来逐渐调整检测器的精度的，这个参数的值越大，精度越低，检测速度越快，越容易漏检；反之，这个值越小，精度越高，检测速度越慢，漏检的几率越小。
    #第三个是minNeighbors，表示每一个目标至少要被检测到3次才算是真的目标，可以理解为目标检测的精确度；
    #第四个是minSize，表示检测到的目标矩形的最小尺寸，小于这个尺寸的矩形会被忽略，这个参数用来排除一些明显不是人脸的区域。
    #返回的faces是人脸的坐标集合，x,y,w,h分别表示人脸的左上角的坐标和人脸的宽和高
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    if faces == (): # 如果没有检测到人脸，直接显示视频帧，以保持视频流的流畅性
        cv2.imshow('camera', img)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # 画出人脸框，第一个参数是图像，第二个参数是人脸左上角坐标，第三个参数是人脸右下角坐标，第四个参数是画线对应的rgb颜色，第五个参数是线的宽度
        idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w]) # 识别人脸，返回识别结果和置信度
        #这里的置信度评分用来衡量所识别人脸与原模型的差距，0表示完全匹配，一般置信度小于80时可以认为识别成功，可以根据实际情况调整
        if confidence < 80:
                floor = user_info[str(idnum)]
        else:
            floor = "unknown"
        cv2.putText(img, "floor:" + str(floor), (x + 5, y - 5), font, 1, (0, 0, 255), 1) # 显示楼层号
        cv2.putText(img, str('{:.0f}'.format(confidence)), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1) # 显示置信度
        #confidence调整为整数显示
        cv2.imshow('camera', img)
    k = cv2.waitKey(10)
    if k == 27:
        break # 按ESC退出
cam.release() # 释放摄像头
cv2.destroyAllWindows() # 关闭所有窗口