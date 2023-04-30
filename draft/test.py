import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
# from src.io.psee_loader import PSEELoader
import cv2
import os
   
# plot test
'''
its already binary after canny
'''


def plot():
   plt.figure(num=1,figsize=(8,4)) 
   # x=np.linspace(-1,1,50)
   # x = np.arange(1,8)
   y = 0.5     
   plt.ylim([-1.2, 1.2])  # 刻度范围
   plt.xticks([])  # 坐标刻度不可见
   plt.yticks([-1,-0.5,0,0.5,1])
   # y_major_locator=MultipleLocator(0.5)
   ax = plt.gca()
   # ax.yaxis.set_major_locator(y_major_locator)

   ax.spines["top"].set_color('none')#上轴不显示
   ax.spines["right"].set_color('none')#右
   ax.spines["bottom"].set_color('blue')
   # ax.xaxis.set_ticks_position['bottom']
   ax.spines["bottom"].set_position(('data', 0))
   plt.plot( y, color='blue', linewidth=3, linestyle='-')
   plt.show()


# def e_file_load(e_filedir, e_filename):
#     file = e_filedir + '/' + e_filename
#     video = PSEELoader(file)
#     # print(video)  # PSEELoader: -- config ---
#     # print(video.event_count())  # number of events in the file
#     # print(video.total_time())  # duration of the file in mus
#     events = video.load_n_events(video.event_count())
#     return events
   

def dilation(img):
   kernel = np.ones((3,3),np.uint8)
   # erosion_img = cv2.erode(img,kernel,iterations = 1)
   dila_img = cv2.dilate(img,kernel,iterations = 1)

   return dila_img

def er_di(img):
   kernel = np.ones((2,2),np.uint8)
   erosion_img = cv2.erode(img,kernel,iterations = 1)
   dilation = cv2.dilate(erosion_img,kernel,iterations = 1)

   # cv2.imshow('erosion',erosion)
   return dilation

def sobel(img):
   grad_x = cv2.Sobel(img, cv2.CV_64F, 1,0)
   grad_y = cv2.Sobel(img, cv2.CV_64F, 0,1)
   
   absX = cv2.convertScaleAbs(grad_x)  # 转回uint8
   absY = cv2.convertScaleAbs(grad_y)
   # nms = NMS(absX, absY)
   # nms = NMS(grad_x, grad_y)
   dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
   return dst

def ev_process(img):
   # print(filename)
   # ev_img = cv2.imread(path + '/' + filename,0)
   ev_img_canny = cv2.Canny(ev_img,100,200)
   ev_img_sobel = sobel(ev_img)
   
   img_list = []

   canny_di = dilation(ev_img_canny)
   canny_er_di = er_di(ev_img_canny)
   kernel = np.ones((3, 3), np.uint8)
   canny_open = cv2.morphologyEx(ev_img_canny, cv2.MORPH_OPEN, kernel)

   sobel_di = dilation(ev_img_sobel)
   sobel_er_di = er_di(ev_img_sobel)
   sobel_open = cv2.morphologyEx(ev_img_sobel, cv2.MORPH_OPEN, kernel)

   img_list = [ev_img_canny,ev_img_sobel,canny_di,sobel_di,canny_er_di,sobel_er_di,canny_open,sobel_open]
   img_combine = []
   for i,img in enumerate(img_list):
      if i % 2 == 1:
         print(i)
         vconcat_img = cv2.vconcat([img_list[i-1],img_list[i]]) 
         img_combine.append(vconcat_img)
         # cv2.imshow('vconcat_img',vconcat_img)
         # k = cv2.waitKey(0)
         # if k == 27:         # ESC
         #    cv2.destroyAllWindows()
      # else:
   img_combine = cv2.hconcat(img_combine)
   cv2.imshow("img_combine",img_combine)
   sobel_er_di_canny = cv2.Canny(sobel_er_di,100,200)
   cv2.imshow('sobel_er_di_canny',sobel_er_di_canny)
   k = cv2.waitKey(0)
   if k == 27:         # ESC
      cv2.destroyAllWindows()

   # save_path = '/Users/cainan/Desktop/Project/data/processed/canny'
         # cv2.imwrite(save_path + '/' + 'event_canny_open' + str(num) + '.png',MORPH_OPEN_img_canny)


if __name__ == '__main__':
   path = '/Users/cainan/Desktop/Project/data/processed/cutoff_10'
   path = '/Users/cainan/Desktop/Project/data/processed/origin_rectify'
   dirs = os.listdir(path)
   num = 105
   for filename in dirs:
      if 'ev' in filename and str(num) in filename:
         # print(filename)
         ev_img = cv2.imread(path + '/' + filename,0)
         ev_process(ev_img)


         # ev_img_canny = cv2.Canny(ev_img,100,200)
         # ev_img_sobel = sobel(ev_img)
         
         # img_list = []

         # canny_di = dilation(ev_img_canny)
         # canny_er_di = er_di(ev_img_canny)
         # kernel = np.ones((3, 3), np.uint8)
         # canny_open = cv2.morphologyEx(ev_img_canny, cv2.MORPH_OPEN, kernel)

         # sobel_di = dilation(ev_img_sobel)
         # sobel_er_di = er_di(ev_img_sobel)
         # sobel_open = cv2.morphologyEx(ev_img_sobel, cv2.MORPH_OPEN, kernel)

         # img_list = [ev_img_canny,ev_img_sobel,canny_di,sobel_di,canny_er_di,sobel_er_di,canny_open,sobel_open]
         # img_combine = []
         # for i,img in enumerate(img_list):
         #    if i % 2 == 1:
         #       print(i)
         #       vconcat_img = cv2.vconcat([img_list[i-1],img_list[i]]) 
         #       img_combine.append(vconcat_img)
         #       # cv2.imshow('vconcat_img',vconcat_img)
         #       # k = cv2.waitKey(0)
         #       # if k == 27:         # ESC
         #       #    cv2.destroyAllWindows()
         #    # else:
         # img_combine = cv2.hconcat(img_combine)
         # cv2.imshow("img_combine",img_combine)
         # sobel_er_di_canny = cv2.Canny(sobel_er_di,100,200)
         # cv2.imshow('sobel_er_di_canny',sobel_er_di_canny)
         # k = cv2.waitKey(0)
         # if k == 27:         # ESC
         #    cv2.destroyAllWindows()

         # save_path = '/Users/cainan/Desktop/Project/data/processed/canny'
         # cv2.imwrite(save_path + '/' + 'event_canny_open' + str(num) + '.png',MORPH_OPEN_img_canny)

      elif 'fr' in filename and str(num) in filename:
         # TODO 形态和canny看看哪个优先
         fr_img = cv2.imread(path + '/' + filename,0)
         delation_img = dilation(fr_img)
         canny_deli = cv2.Canny(delation_img,100,200)
         fr_img_canny = cv2.Canny(fr_img, 100, 200) 
         img_combine = cv2.hconcat([fr_img,fr_img_canny])
         cv2.imshow('frimg_combine',img_combine)
         k = cv2.waitKey(0)
         if k == 27:         # ESC
            cv2.destroyAllWindows()

      else:
         print('1')
         print(filename)







