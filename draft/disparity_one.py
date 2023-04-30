import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt
'''
until 11.12.2022 google document  TODO
do the NCC only on the epiline and plot
result是一个结果矩阵,假设待匹配图像为 I,宽高为(W,H),模板图像为 T,宽高为(w,h)。那么result的大小就为(W-w+1,H-h+1) 。
原因是因为,在匹配时,以模板大小的搜索框依次遍历整张图片时,每行需要遍历(W-w+1)次,每列需要遍历(H-h+1)。
弄出ROI 只在ROI上做cv2.matchTemplate 计算就只在 w-w+1 = 1 行上进行 不会进行多余的计算
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
    plt.ylim([-1.2, 1.2]) 
    plt.xticks([])  
    plt.yticks([-1,-0.5,0,0.5,1])

    ax = plt.gca()
    ax.spines["top"].set_color('none')
    ax.spines["right"].set_color('none')
    ax.spines["bottom"].set_color('blue')
    ax.spines["bottom"].set_position(('data', 0))
    plt.plot(xpoints, ypoints, color='blue', linewidth=2, linestyle='-')
    plt.plot(xpoints[index_ymax], ypoints[index_ymax], color='red', marker='o')
    print('xpoints[index_ymax]',xpoints[index_ymax],'ypoints[index_ymax]',ypoints[index_ymax])
    plt.show()

def show_debug(img_name,img):  # img_name,img
    cv2.imshow(img_name,img)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        cv2.destroyAllWindows()    
    quit()

if __name__ == '__main__':
    save_path = '/Users/cainan/Desktop/Project/data/processed'
    '''
    LOAD DATA
    '''
    # load event preprocessed img
    ev_img_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
    ev_img_filename = 'cam1-0noline107ev_rectified_alpha_5.png'
    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640
    # load frame preprocessed img
    prepro_frame_img_path = ev_img_path
    prepro_frame_img_filename = 'cam1-0noline107fr_rectified_alpha_5.png'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # # shape (1536,2048)  3.2times of 480
    '''
    processing (resize)frame and pick the patch from frame
    '''
    resize_frame_img = pre_frame_img
    patch_c = [259, 197]   # [x,y] 
    patch_height = 20 
    patch_width = 20
    top =patch_c[1] -int( patch_height * 0.5)
    left = patch_c[0] - int(patch_width * 0.5)
    patch = resize_frame_img[top:top + patch_height, left:left + patch_width]   # top down left right    
    color_frame_patch = cv2.cvtColor(resize_frame_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color_frame_patch,  (left, top), (left + patch_width, top + patch_height),(0,0,255), 1)  # red
    cv2.imshow('color_frame_patch',color_frame_patch)
    '''
    NCC cross correlation  do it only on the epipolar line 
    '''
    ev_img_epi = ev_img[top:top + patch_height,:] # top down left right; shape (20,640)
    # show_debug('ev_img_epi',ev_img_epi)  # img_name,img

    result = cv2.matchTemplate(ev_img, patch, cv2.TM_CCOEFF_NORMED) # (origin(big_img), patch(Template), mat_method)
    minMaxLoc_result = cv2.minMaxLoc(result) # min_val, max_val, min_loc, max_loc
    epi_result = cv2.matchTemplate(ev_img_epi, patch, cv2.TM_CCOEFF_NORMED) # shape (1,621)
    minMaxLoc_epi_result = cv2.minMaxLoc(epi_result)
    _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
    # epi_result = epi_result[0]  # shape from (1,621) to (621,)
    print('min_val, max_val, min_loc, max_loc',minMaxLoc_result)   # min_val, max_val, min_loc, max_loc (-0.4046742618083954, 0.5434345602989197, (310, 394), (224, 186))
    print('epi_val min_val, max_val, min_loc, max_loc',minMaxLoc_epi_result)  # (-0.18111075460910797, 0.49978014826774597, (384, 0), (222, 0))
    x = np.array([i + int(patch_width * 0.5) for i in range(result.shape[1])]) # len 561
    plot(x, epi_result[0]) 
    e_max_x_left = minMaxLoc_epi_result[3][0]
    e_max_x = e_max_x_left + int(patch_width * 0.5) 
    disp = e_max_x - patch_c[0]
    ev_img_color = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(ev_img_color, (e_max_x_left,top), (e_max_x_left + patch_width,top + patch_height), (0,255,255), 1)   # yellow
    print(e_max_x)
    show_debug('ev_img_find',ev_img_color)







    quit()
    epipolar_line = patch_c[1]   # right after the rectify
    epi_result = result[epipolar_line]  # TODO according to the epi result generate the new rectangle on ev_img

    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
    cv2.line(ev_img,(0, patch_c[1]),(640, patch_c[1]),(255,0,0),1)  # blue draw the epipolar line on the event image
    # cv2.rectangle(ev_img, top_left, bottom_right, (0,255,255), 2)   # yellow
    cv2.imshow('ev_img_find',ev_img)

    NCC_norm_result = img_normalization(result)  # shape 441 561 and np
    cv2.imshow('NCC_norm_result',NCC_norm_result)

    x = np.array([i + int(patch_width * 0.5) for i in range(result.shape[1])]) # len 561
    plot(x, epi_result)  # plot NCC result on the epipolar line

    k = cv2.waitKey(0)
    i = 1
    if k == 27:         # ESC
        # cv2.imwrite(save_path + '/' + 'ev_img_find' + str(i) + '.png',ev_img)
        # cv2.imwrite(save_path + '/' + 'NCC_result' + str(i) + '.png',norm_result)
        cv2.destroyAllWindows()

