import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image

#img=cv2.imread(r'C:\Users\vennela.vulluri\Desktop\NFI-00101001.PNG')
img=cv2.imread(r'C:\Users\vennela.vulluri\Desktop\ABC.jpg')
print(img.size,type(img.size))

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('image', gray)

dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
#cv2.imshow('image', dst)
#cv2.waitKey(0)
#plt.subplot(121), plt.imshow(img)
#plt.subplot(122), plt.imshow(dst)
#plt.show()

    # 40x10 image as a flatten array
flatten_img = cv2.resize(img, (40, 10), interpolation=cv2.INTER_AREA).flatten()

    # resize to 400x100
resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
columns = np.sum(resized, axis=0)  # sum of all columns
lines = np.sum(resized, axis=1)  # sum of all lines
h, w = img.shape
aspect = w / h

cv2.imshow('image', resized)
cv2.waitKey(0)
print(columns,lines)
print(aspect)


ary = np.array(dst)
r,g,b = np.split(ary,3,axis=2)
r=r.reshape(-1)
g=r.reshape(-1)
b=r.reshape(-1)
bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2],
zip(r,g,b)))
bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
bitmap = np.dot((bitmap > 128).astype(float),255)
im = Image.fromarray(bitmap.astype(np.uint8))
im.show()

