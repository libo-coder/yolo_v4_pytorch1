# -*- coding: utf-8 -*-
"""
对单张图片进行预测
@author: libo
"""
from yolo import YOLO
from PIL import Image

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.save('./debug/r_image.jpg')
        r_image.show()
