import cv2
 
IMAGE_PATH = 'images/' # 人脸图片保存路径
MAX_FACES = 1000 # 采集1000张人脸图像，可以根据需要自行修改

cap = cv2.VideoCapture(0) # 开启摄像头
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 人脸检测器，可以在opencv安装目录下找到，也可以使用其他的人脸检测器，这里使用的是opencv自带的
face_id = input('input user id')  # 输入用户编号
user_floor = input('input user floor')  # 输入用户楼层
#保存用户编号和楼层
with open('user_info.txt','a') as f:
    f.write(face_id + ' ' + user_floor + '\n')
print('Start collecting face images, look at the camera with rotating your face gently and wait ...')
count = 0 # 存储采集人脸张数
 
while cap.isOpened():
    ret, frame = cap.read() # 读取一帧的图像
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转换为灰度图像
    else:
        break
    #face_detector.detectMultiScale参数说明
    #第一个参数是要检测的图片，一般是灰度图像加快检测速度；
    #第二个是scaleFactor，表示每次图像尺寸减小的比例,可以理解为相机的X倍镜。它是用来逐渐调整检测器的精度的，这个参数的值越大，精度越低，检测速度越快，越容易漏检；反之，这个值越小，精度越高，检测速度越慢，漏检的几率越小。
    #第三个是minNeighbors，表示每一个目标至少要被检测到3次才算是真的目标，可以理解为目标检测的精确度；
    #返回的faces是人脸的坐标集合，x,y,w,h分别表示人脸的左上角的坐标和人脸的宽和高
    faces = face_detector.detectMultiScale(gray, 1.3, 5) # 检测人脸
    for (x, y, w, h) in faces: # 遍历每一个人脸
        cv2.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0), 2) # 画出人脸框，第一个参数是图像，第二个参数是人脸左上角坐标，第三个参数是人脸右下角坐标，第四个参数是画线对应的rgb颜色，第五个参数是线的宽度
        count += 1 
        cv2.imwrite(IMAGE_PATH + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w]) # 保存人脸图像为编号.第几张.jpg的格式在images文件夹下，使用灰度图像
        cv2.imshow('image', frame) # 显示图像
        print('image' + str(count) + 'saved')
    k = cv2.waitKey(1) 
    if k == 27: # 按下ESC键退出
        break
    elif count >= MAX_FACES: # 达到最大采集数量退出
        break
cap.release() # 释放摄像头
cv2.destroyAllWindows() # 关闭所有窗口