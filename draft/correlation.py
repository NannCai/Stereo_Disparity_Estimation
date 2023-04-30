import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt

def img_normalization(img,percentile_low = 0.1,percentile_high = 99.9):
    rmin,rmax = np.percentile(img,(percentile_low,percentile_high))
    # print('min' ,rmin)
    # print('max',rmax)
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
    return img  

def plot(xpoints, ypoints):

    plt.figure(num = 1,figsize = (8,4)) 
    if np.max(np.abs(ypoints)) > 1 :
        print('ypoints value is wrong!!!!')
    index_ymax = np.argmax(ypoints)
    # max_point = ypoints[index_ymax] 
    print(index_ymax)   # 33
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
    # plt.plot( y, color='blue', linewidth=3, linestyle='-')
    plt.plot(xpoints, ypoints, color='blue', linewidth=2, linestyle='-')
    plt.plot(xpoints[index_ymax], ypoints[index_ymax], color='red', marker='o')
    plt.show()
    NCC_point = [xpoints[index_ymax], ypoints[index_ymax] + 20]
    return NCC_point


if __name__ == '__main__':

    save_path = '/Users/cainan/Desktop/Project/data/processed'
    i = 1

    '''
    LOAD DATA
    '''
    # load event preprocessed img
    ev_img_path = '/Users/cainan/Desktop/Project/data/processed'
    ev_img_filename = 'event_binary0.png'
    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640)
    # The baseline b = 65.44 mm, focal lengths f = 555 pixels
    ev_b = 65.44
    ev_f = 555.0
    cv2.imshow('ev_img',ev_img)

    # load frame preprocessed img
    prepro_frame_img_path = ev_img_path
    prepro_frame_img_filename = 'frame_binary0.png'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # # shape (1536,2048)  3.2times of 480
    # f = 1301 pixels for FLIR camera.
    frame_f = 1301.0
    ratio = ev_f / frame_f 
    resize_size = (int(ratio * pre_frame_img.shape[1]),int(ratio * pre_frame_img.shape[0]))
    resize_frame_img = cv2.resize(pre_frame_img, resize_size)
    cv2.imshow('not cut frame_img',resize_frame_img)
    left =int((resize_frame_img.shape[1] - ev_img.shape[1]) * 0.5)
    right = left + ev_img.shape[1]
    up = int((resize_frame_img.shape[0] - ev_img.shape[0]) * 0.5)
    down = up + ev_img.shape[0]
    # cut_size = (left:right, up:down)
    crop_frame = resize_frame_img[up:down, left:right]
    # print(pre_frame_img.shape)  # (480, 640) the img is wrong
    cv2.imshow('crop_frame',crop_frame)
    # patch = calibrated_frame[110:250, 115:180]
    # quit()
    patch = crop_frame[240:280, 110:190]   # height width
    patch_c = [260,150]
    crop_frame_patch = cv2.cvtColor(crop_frame, cv2.COLOR_GRAY2BGR)
    # crop_frame_patch = 
    cv2.line(crop_frame_patch,(0, patch_c[0]),(640, patch_c[0]),(255,0,0),5)
    cv2.rectangle(crop_frame_patch, (190, 280), (110, 240), (0,0,255), 2)
    cv2.imshow('crop_frame_patch',crop_frame_patch)
    ph, pw = patch.shape[:2]


    k = cv2.waitKey(0)
    if k == 27:         # ESC
        cv2.imwrite(save_path + '/' + 'ev_img' + str(i) + '.png',ev_img)
        cv2.imwrite(save_path + '/' + 'resize_frame_img' + str(i) + '.png',resize_frame_img)
        cv2.imwrite(save_path + '/' + 'crop_frame' + str(i) + '.png',crop_frame)
        cv2.imwrite(save_path + '/' + 'crop_frame_patch' + str(i) + '.png',crop_frame)
        cv2.destroyAllWindows()

    # quit()
    # patch:from frame ori:from event
    result = cv2.matchTemplate(ev_img, patch, cv2.TM_CCOEFF_NORMED) # (origin(big_img), patch(Template), mat_method)
    # ph, pw = patch.shape[:2]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    norm_result = img_normalization(result)  # shape 441 561 and np
    epipolar_line = patch_c[0]
    epi_result = result[epipolar_line]  # size 561 array
    x = np.array([i + 40 for i in range(norm_result.shape[1])]) # len 561
    oneline_NCC_point = plot(x, epi_result)
    # print(x[-1])  # 600
    top_left = max_loc
    bottom_right = (top_left[0] + pw, top_left[1] + ph)
    # top_left_line = int(oneline_NCC_point[0] - pw * 0.5), int()
    # bottom_right_line = 

    # flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    # print( flags )  # COLOR_GRAY2BGR    
    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
    print('top_left',top_left,' bottom_right',bottom_right)
    # cv2.line(ev_img,(0, oneline_NCC_point[1]),(640, oneline_NCC_point[1]),(255,0,0),5)
    cv2.line(ev_img,(0, patch_c[0]),(640, patch_c[0]),(255,0,0),5)
    # cv.line(img,(0,0),(511,511),(255,0,0),5)
    cv2.rectangle(ev_img, top_left, bottom_right, (0,255,255), 2) # 在原图上画矩形
    cv2.imshow('ev_img_find',ev_img)
    cv2.imshow('NCC_norm_result',norm_result)
    cv2.imshow('crop_frame_patch',crop_frame_patch)


 


    k = cv2.waitKey(0)
    if k == 27:         # ESC
        # cv2.imshow('ev_img',ev_img)
        # cv2.imshow('not cut frame_img',resize_frame_img)
        # cv2.imshow('crop_frame',crop_frame)
        # cv2.imshow('crop_frame_patch',crop_frame)
        # cv2.imshow('ev_img_find',ev_img)
        # cv2.imshow('NCC_result',result)

        # cv2.imwrite(save_path + '/' + 'ev_img_find' + str(i) + '.png',ev_img)
        # cv2.imwrite(save_path + '/' + 'NCC_result' + str(i) + '.png',norm_result)



        cv2.destroyAllWindows()





