import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt
'''
with old parameters f and b
'''

def img_normalization(img,percentile_low = 0.1,percentile_high = 99.9):
    rmin,rmax = np.percentile(img,(percentile_low,percentile_high))
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
    return img  

def plot(xpoints, ypoints):
    plt.figure(num = 1,figsize = (8,4)) 
    if np.max(np.abs(ypoints)) > 1 :
        print('ypoints value is wrong!!!!')
    index_ymax = np.argmax(ypoints)
    # print(index_ymax)   # 33
    plt.ylim([-1.2, 1.2])  # 刻度范围
    plt.xticks([])  # 坐标刻度不可见
    plt.yticks([-1,-0.5,0,0.5,1])

    ax = plt.gca()
    ax.spines["top"].set_color('none')#上轴不显示
    ax.spines["right"].set_color('none')#右
    ax.spines["bottom"].set_color('blue')
    ax.spines["bottom"].set_position(('data', 0))
    plt.plot(xpoints, ypoints, color='blue', linewidth=2, linestyle='-')
    plt.plot(xpoints[index_ymax], ypoints[index_ymax], color='red', marker='o')
    print('xpoints[index_ymax]',xpoints[index_ymax],'ypoints[index_ymax]',ypoints[index_ymax])
    plt.show()

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - 我们在img2相应位置绘制极点生成的图像 line zai img1 shang
        lines - 对应的极点 '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

if __name__ == '__main__':

    save_path = '/Users/cainan/Desktop/Project/data/processed'

    '''
    LOAD DATA
    '''
    # load event preprocessed img
    # ev_img_path = '/Users/cainan/Desktop/Project/data/processed'
    ev_img_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
    # ev_img_filename = 'event_binary0.png'
    ev_img_filename = 'cam1-0noline107ev_rectified_alpha_5.png'
    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640)
    # The baseline b = 65.44 mm, focal lengths f = 555 pixels
    # ev_b = 65.44
    # ev_f = 555.0

    # load frame preprocessed img
    prepro_frame_img_path = ev_img_path
    # prepro_frame_img_filename = 'frame_binary0.png'
    prepro_frame_img_filename = 'cam1-0noline107fr_rectified_alpha_5.png'

    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # # shape (1536,2048)  3.2times of 480

    '''
    processing (resize)frame and pick the patch from frame
    '''
    # frame_f = 1301.0  # f = 1301 pixels for FLIR camera. 
    # ratio = ev_f / frame_f 
    # resize_size = (int(ratio * pre_frame_img.shape[1]),int(ratio * pre_frame_img.shape[0]))
    # resize_frame_img = cv2.resize(pre_frame_img, resize_size)
    resize_frame_img = pre_frame_img
    patch_c = [259, 197]   # [x,y] 
    patch_height = 20 
    patch_width = 20
    # print('ph pw from variable', patch_height,patch_width)
    # patch = resize_frame_img[240:280, 110:190]   # height width
    top =patch_c[1] -int( patch_height * 0.5)
    left = patch_c[0] - int(patch_width * 0.5)
    # print('left' , left) 
    patch = resize_frame_img[top:top + patch_height, left:left + patch_width]   # top down left right
    # print('patch_height, patch_width from patch', patch.shape[:2])
    

    color_frame_patch = cv2.cvtColor(resize_frame_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color_frame_patch,  (left, top), (left + patch_width, top + patch_height),(0,0,255), 1)  # red
    cv2.imshow('color_frame_patch',color_frame_patch)
    # print('show')
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     # cv2.imwrite(save_path + '/' + 'ev_img_find' + str(i) + '.png',ev_img)
    #     # cv2.imwrite(save_path + '/' + 'NCC_result' + str(i) + '.png',norm_result)
    #     cv2.destroyAllWindows()
    # quit()

    '''
    NCC cross correlation  do it only on the epipolar line 
    '''
    result = cv2.matchTemplate(ev_img, patch, cv2.TM_CCOEFF_NORMED) # (origin(big_img), patch(Template), mat_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print('max_val',max_val)

    top_left = max_loc
    bottom_right = (top_left[0] + patch_width, top_left[1] + patch_height)
    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
    print('top_left',top_left,' bottom_right',bottom_right)  # top_left (223, 185)  bottom_right (243, 205) right!

    # epipolar_line = patch_c[1]   # right after the rectify
    # epipolar_line = top_left[1] + int(patch_width * 0.5)
    # epipolar_line = top_left[1]
    if patch_c[1] - (top_left[1] + int(patch_width * 0.5)) != 0:
        print('patch_c[1] - epipolar_line = ',patch_c[1] - (top_left[1] + int(patch_width * 0.5)))
    print('epipolar_line',top_left[1] + int(patch_width * 0.5))
    epi_result = result[top_left[1]]
    # epi_result = result[epipolar_line]  # size 561 array   # the wrong position

    NCC_norm_result = img_normalization(result)  # shape 441 561 and np
    cv2.line(ev_img,(0, patch_c[1]),(640, patch_c[1]),(255,0,0),1)  # blue draw the epipolar line on the event image
    cv2.rectangle(ev_img, top_left, bottom_right, (0,255,255), 2) # 在原图上画矩形 yellow
    cv2.imshow('ev_img_find',ev_img)
    cv2.imshow('NCC_norm_result',NCC_norm_result)

    x = np.array([i + int(patch_width * 0.5) for i in range(result.shape[1])]) # len 561
    # TODO output the figures max
    plot(x, epi_result)  # plot NCC result on the epipolar line

    k = cv2.waitKey(0)
    i = 1
    if k == 27:         # ESC
        # cv2.imwrite(save_path + '/' + 'ev_img_find' + str(i) + '.png',ev_img)
        # cv2.imwrite(save_path + '/' + 'NCC_result' + str(i) + '.png',norm_result)
        cv2.destroyAllWindows()

