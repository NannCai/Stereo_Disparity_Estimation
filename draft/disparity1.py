import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt
'''
until 20.12.2022 google document update 
compute disparity on all samples but some problem maybe
change the rectifid image from alfha=0✅
no 整理
'''

def img_normalization(img,percentile_low = 0.1,percentile_high = 99.9):
    rmin,rmax = np.percentile(img,(percentile_low,percentile_high))
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
    return img

def plot(template_width, ypoints):
    xpoints = np.array([i + int(template_width * 0.5) for i in range(ypoints.shape[0])])
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

def compute_disp(pre_frame_img,ev_img,template_c, template_size): # for one point
    template_width, template_height = template_size 
    # template_c = [259, 197]   # [x,y] 
    # template_height = 20 
    # template_width = 20
    top =template_c[1] -int( template_height * 0.5)
    left = template_c[0] - int(template_width * 0.5)
    template = pre_frame_img[top:top + template_height, left:left + template_width]   # top down left right    
    # color_frame_template = cv2.cvtColor(pre_frame_img, cv2.COLOR_GRAY2BGR)
    # cv2.rectangle(color_frame_template,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
    # cv2.imshow('color_frame_template',color_frame_template)
    
    # NCC cross correlation   only on the epipolar line 
    ev_epi_img = ev_img[top:top + template_height,:] # top down left right; shape (20,640)
    epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
    # minMaxLoc_epi_result = cv2.minMaxLoc(epi_result)    # min_val, max_val, min_loc, max_loc
    _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
    # print('epi_val min_val, max_val, min_loc, max_loc',minMaxLoc_epi_result)  # (-0.18111075460910797, 0.49978014826774597, (384, 0), (222, 0))
    # e_max_x_left = max_loc[0]
    e_max_x = max_loc[0] + int(template_width * 0.5) 
    disp = e_max_x - template_c[0]
    plot(template_width, epi_result)
    return disp

def compute_disparity(pre_frame_img,ev_img,template_c, template_size):   # input stepsize template_size
    '''
    disparity_map = np.zeros((m,n,steps), np.uint8)
    for point in range ....:  # stepsize
        point = template_c
        disp = compute_disp(template_c, template_size)
    disparity_map = 0
    return disparity_map
    '''
    img_height,img_width = pre_frame_img.shape
    # print('img_height,img_width',img_height,img_width)  # 480 640
    stepsize = 1
    disparity_map = np.zeros((img_height,img_width), np.uint8)
    template_width, template_height = template_size 
    for x in range(template_width,img_width,stepsize):
        for y in range(template_height,img_height,stepsize):
            template_c = (x,y)
            # print(template_c)
            top = template_c[1] -int( template_height * 0.5)
            left = template_c[0] - int(template_width * 0.5)
            template = pre_frame_img[top:top + template_height, left:left + template_width]   # top down left right    
            # NCC cross correlation   only on the epipolar line 
            ev_epi_img = ev_img[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            e_max_x = max_loc[0] + int(template_width * 0.5) 
            # disp = e_max_x - template_c[0]
            disp = template_c[0] - e_max_x
            disparity_map[y,x] = disp
    print('1')
    return disparity_map    # ? 0-255

if __name__ == '__main__':
    save_path = '/Users/cainan/Desktop/Project/data/processed'
    '''
    LOAD DATA
    '''
    # load event preprocessed img
    ev_img_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
    ev_img_filename = 'cam1-0noline107ev_rectified_alpha_0.png'
    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640
    # load frame preprocessed img
    prepro_frame_img_path = ev_img_path
    prepro_frame_img_filename = 'cam1-0noline107fr_rectified_alpha_0.png'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # # shape (1536,2048)  3.2times of 480
    '''
    processing (resize)frame and pick the template from frame
    '''
    # set

    template_c = [259, 197]   # [x,y] 
    template_height = 10 
    template_width = 10
    disparity_map = compute_disparity(pre_frame_img,ev_img,template_c, (template_height,template_width))
    # disparity_map_norm = img_normalization(disparity_map,percentile_low = 1,percentile_high = 99.5)  # TODO 热力图
    # show_debug("disparity_map_norm",disparity_map_norm)  # img_name,img 34s
    show_debug('disparity_map',disparity_map)

    # disp = compute_disp(template_c, (template_height,template_width))
    # print(disp)

    # top =template_c[1] -int( template_height * 0.5)
    # left = template_c[0] - int(template_width * 0.5)
    # template = pre_frame_img[top:top + template_height, left:left + template_width]   # top down left right    
    # color_frame_template = cv2.cvtColor(pre_frame_img, cv2.COLOR_GRAY2BGR)
    # cv2.rectangle(color_frame_template,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
    # cv2.imshow('color_frame_template',color_frame_template)
    # '''
    # NCC cross correlation   only on the epipolar line 
    # '''
    # ev_epi_img = ev_img[top:top + template_height,:] # top down left right; shape (20,640)
    # epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
    # # minMaxLoc_epi_result = cv2.minMaxLoc(epi_result)    # min_val, max_val, min_loc, max_loc
    # _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
    # # print('epi_val min_val, max_val, min_loc, max_loc',minMaxLoc_epi_result)  # (-0.18111075460910797, 0.49978014826774597, (384, 0), (222, 0))
    # # e_max_x_left = max_loc[0]
    # e_max_x = max_loc[0] + int(template_width * 0.5) 
    # disp = e_max_x - template_c[0]



    # min_val, max_val, min_loc, max_loc (-0.4046742618083954, 0.5434345602989197, (310, 394), (224, 186))
    # epi_val min_val, max_val, min_loc, max_loc (-0.18111075460910797, 0.49978014826774597, (384, 0), (222, 0))
    # 234





    quit()

    epipolar_line = template_c[1]   # right after the rectify
    epi_result = result[epipolar_line]  # TODO according to the epi result generate the new rectangle on ev_img

    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
    cv2.line(ev_img,(0, template_c[1]),(640, template_c[1]),(255,0,0),1)  # blue draw the epipolar line on the event image
    # cv2.rectangle(ev_img, top_left, bottom_right, (0,255,255), 2)   # yellow
    cv2.imshow('ev_img_find',ev_img)

    NCC_norm_result = img_normalization(result)  # shape 441 561 and np
    cv2.imshow('NCC_norm_result',NCC_norm_result)


    plot(template_width, epi_result)  # plot NCC result on the epipolar line

    k = cv2.waitKey(0)
    i = 1
    if k == 27:         # ESC
        # cv2.imwrite(save_path + '/' + 'ev_img_find' + str(i) + '.png',ev_img)
        # cv2.imwrite(save_path + '/' + 'NCC_result' + str(i) + '.png',norm_result)
        cv2.destroyAllWindows()

