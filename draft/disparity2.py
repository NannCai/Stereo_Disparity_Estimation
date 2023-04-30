import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt

'''
until 11.12.2022 google document update TODO
compute disparity on all samples✅ but some problem maybe-- use good dataset to test and correct the code 
change the rectifed image from alfha=0✅
solution:
disparity_map from uint8 to float64✅
try to use good dataset to make sure the algorithm is right✅
the result on data tsukuba is wrong --12.29 seems right but still have a lot of things to improve the matching result:
1. the metric to see whether the code is improving the result or not
    remove the mismatching
        can do the left right yizhixing  but is too expensive maybe
2. the result on hybrid sys is bad -- need to find the problem (maybe the method of matching or the quality of ev_img is too bad)
'''

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
    plot(template_width, epi_result[0])
    return disp

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

def show_debug(img_name,img,save_pa = None):  # img_name,img
    cv2.imshow(img_name,img)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        if save_pa is not None:
            cv2.imwrite(save_path + '/' + 'dis_norm_h5' + '.png',img)
        cv2.destroyAllWindows()    
    quit()

def img_normalization(img,percentile_low = 0.05,percentile_high = 99.95):
    rmin,rmax = np.percentile(img,(percentile_low,percentile_high))
    # image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
    return img

def compute_disparity(pre_frame_img,ev_img, template_size):   # input stepsize template_size
    img_height,img_width = pre_frame_img.shape
    stepsize = 1
    step_num = 0
    wrong_num = 0
    test_wr_num =0
    disparity_map = np.zeros((img_height,img_width), np.float64)
    template_width, template_height = template_size 
    for x in range(int(template_width * 0.5),int(img_width - template_width * 0.5),stepsize):
        for y in range(int(template_height * 0.5),int(img_height - template_height * 0.5),stepsize):
    # for x in range(10,int(img_width - template_width * 0.5),stepsize):
    #     for y in range(60,int(img_height - template_height * 0.5),stepsize):
            step_num = step_num + 1
            template_c = (x,y)
            # template_c = (97,186)
            # print(template_c)
            top = template_c[1] -int( template_height * 0.5)
            left = template_c[0] - int(template_width * 0.5)
            template = pre_frame_img[top:top + template_height, left:left + template_width]   # top down left right    

            # NCC cross correlation   only on the epipolar line 
            ev_epi_img = ev_img[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            e_max_x = max_loc[0] + int(template_width * 0.5) 
            
            # draw the template on the image
            # color_frame_patch = cv2.cvtColor(pre_frame_img, cv2.COLOR_GRAY2BGR)
            # ev_img_color = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)

            # cv2.rectangle(color_frame_patch,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
            # cv2.imshow('color_frame_patch',color_frame_patch)

            # cv2.rectangle(ev_img_color, (max_loc[0],top), (max_loc[0] + template_width,top + template_height), (0,255,255), 1)   # yellow
            # cv2.imshow('ev_img_find',ev_img_color)

            # k = cv2.waitKey(0)
            # if k == 27:         # ESC
            #     # if save_pa is not None:
            #     #     cv2.imwrite(save_path + '/' + 'dis_norm_h5' + '.png',img)
            #     cv2.destroyAllWindows()    

            # disp = e_max_x - template_c[0]
            disp = template_c[0] - e_max_x
            # if disp < 0:
            #     color_frame_patch = cv2.cvtColor(pre_frame_img, cv2.COLOR_GRAY2BGR)
            #     ev_img_color = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)

            #     cv2.rectangle(color_frame_patch,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
            #     cv2.imshow('color_frame_patch',color_frame_patch)

            #     cv2.rectangle(ev_img_color, (max_loc[0],top), (max_loc[0] + template_width,top + template_height), (0,255,255), 1)   # yellow
            #     cv2.imshow('ev_img_find',ev_img_color)

            #     k = cv2.waitKey(0)
            #     if k == 27:         # ESC
            #         # if save_pa is not None:
            #         #     cv2.imwrite(save_path + '/' + 'dis_norm_h5' + '.png',img)
            #         cv2.destroyAllWindows()  
            if max_val < 0.6 or disp < 0 or disp > 70:
                # print(disp)
                disp = 0
                test_wr_num = test_wr_num+1
            # if disp > 50 and max_val < 0.7:  # 视差较大而相关度一般时
            #     # if disp < 0:
            #     #     print("max_val",max_val)
            #     #     print(disp)
            #     wrong_num = wrong_num + 1
            #     # print('template_c',template_c,'disp',disp)
                disp = 0
                # color_frame_patch = cv2.cvtColor(pre_frame_img, cv2.COLOR_GRAY2BGR)
                # ev_img_color = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)

                # cv2.rectangle(color_frame_patch,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
                # cv2.imshow('color_frame_patch',color_frame_patch)

                # cv2.rectangle(ev_img_color, (max_loc[0],top), (max_loc[0] + template_width,top + template_height), (0,255,255), 1)   # yellow
                # cv2.imshow('ev_img_find',ev_img_color)

                # k = cv2.waitKey(0)
                # if k == 27:         # ESC
                #     # if save_pa is not None:
                #     #     cv2.imwrite(save_path + '/' + 'dis_norm_h5' + '.png',img)
                #     cv2.destroyAllWindows()    
            disparity_map[y,x] = disp
            # print('1')


    print('step_num',step_num,'wrong_num',wrong_num,'max_val < 0.6',test_wr_num)
    # print(np.max(disparity_map))
    return disparity_map    # ? 0-255 cause fault unit8


if __name__ == '__main__':
    save_path = '/Users/cainan/Desktop/Project/data/processed/disparity'
    '''
    LOAD DATA
    '''
    # load event preprocessed img
    # ev_img_path ='/Users/cainan/Desktop/Project/data/tsukuba'
    ev_img_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
    ev_img_filename = 'cam1-0noline107ev_rectified_alpha_0.png'
    # ev_img_filename= 'scene1.row3.col5.ppm'

    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640

    # load frame preprocessed img
    prepro_frame_img_path = ev_img_path
    prepro_frame_img_filename = 'cam1-0noline107fr_rectified_alpha_0.png'
    # prepro_frame_img_filename = 'scene1.row3.col3.ppm'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # shape (1536,2048)  3.2times of 480
    # cv2.imshow('ev_img',ev_img)
    # show_debug("pre_frame_img",pre_frame_img) 
    if ev_img is None or pre_frame_img is None:
        print('Error: Could not load image')
        quit()
    
    '''
    processing (resize)frame and pick the template from frame
    '''
    # set
    template_height = 7
    # template_height = 20
    template_width = template_height

    disparity_map = compute_disparity(pre_frame_img,ev_img, (template_height,template_width))
    disparity_map_norm = img_normalization(disparity_map) 
    # disparity_map_norm = img_normalization(disparity_map,percentile_low = 0.1,percentile_high = 99.5)  # TODO 热力图
    show_debug("disparity_map_norm5",disparity_map_norm,save_pa = save_path)  # img_name,img 34s
    # show_debug('disparity_map',disparity_map,save_pa = save_path)


    # template_c = [117,186]
    # template_c = [97,186]   # get 118
    # disp = compute_disp(pre_frame_img,ev_img,template_c, (template_height,template_width))
    # print(disp)



