import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt
'''
until 11.12.2022 google document
use correct rectify pair images 
compute NCC on the whole image
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
    # print('show')
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     # cv2.imwrite(save_path + '/' + 'ev_img_find' + str(i) + '.png',ev_img)
    #     # cv2.imwrite(save_path + '/' + 'NCC_result' + str(i) + '.png',norm_result)
    #     cv2.destroyAllWindows()
    '''
    NCC cross correlation  do it only on the epipolar line 
    '''
    result = cv2.matchTemplate(ev_img, patch, cv2.TM_CCOEFF_NORMED) # (origin(big_img), patch(Template), mat_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print('max_val',max_val)
    top_left = max_loc
    bottom_right = (top_left[0] + patch_width, top_left[1] + patch_height)
    print('top_left',top_left,' bottom_right',bottom_right)  # top_left (223, 185)  bottom_right (243, 205) right!

    if patch_c[1] - (top_left[1] + int(patch_width * 0.5)) != 0:
        print('patch_c[1] - epipolar_line = ',patch_c[1] - (top_left[1] + int(patch_width * 0.5)))
    print('epipolar_line',top_left[1] + int(patch_width * 0.5))
    epi_result = result[top_left[1]]

    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
    cv2.line(ev_img,(0, patch_c[1]),(640, patch_c[1]),(255,0,0),1)  # blue draw the epipolar line on the event image
    cv2.rectangle(ev_img, top_left, bottom_right, (0,255,255), 2)   # yellow
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

