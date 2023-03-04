import xml.dom.minidom
import cv2
import os
import xml.etree.ElementTree as ET
import pickle
img_name = '/data/lijx/myWSOD-second/data/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
proposals = '/data/lijx/myWSOD-second/data/selective_search_data/voc_2007_trainval.pkl'
with open(proposals,"rb") as f:
    data = pickle.load(f)['boxes'][0][:1000]
img = cv2.imread(img_name)


for obj in data:
    x1 = int(float(obj[0])) + 1
    y1 = int(float(obj[1])) + 1
    x2 = int(float(obj[2]))
    y2 = int(float(obj[3]))

    # xmin, ymin, xmax, ymax分别为xml读取的坐标信息
    left_top = (int(x1), int(y1))
    right_down = (int(x2), int(y2))
    cv2.rectangle(img, left_top, right_down, (255, 0, 0), 1)
    # cv2.rectangle(img, (int(x1 + 3), int(y1 - 15)), (max(int(x2 - 10), 100), int(y1)), (255, 255, 255), -1)
    # cv2.putText(img, cls, (int(x1 + 4), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


cv2.imwrite("test.jpg", img)
cv2.waitKey(0)            # 等待用户按下按键
cv2.destroyAllWindows()   # 关闭所有窗口