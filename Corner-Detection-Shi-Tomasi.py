import cv2
import numpy as np
import matplotlib.pyplot as plt

me= cv2.imread('/home/nlab/OpenCv_Tutorials_Shobhit/Object Detection/group.jpg')
me = cv2.cvtColor(me, cv2.COLOR_BGR2RGB)
plt.imshow(me)
plt.show()

gray_me = cv2.cvtColor(me, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_me, cmap='gray')

corners = cv2.goodFeaturesToTrack(gray_me, 5, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(me, (x, y), 7, (255, 0, 0), -1)

plt.imshow(me)
plt.show()