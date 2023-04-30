import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt
import os

'''
until 11.1.2022 google document update 
done the two layer's pyramid part✅
need to fill in the hole(blanck space) TODO

'''

def img_normalization(img,percentile_low = 0.05,percentile_high = 99.95):
    norm_img = img.copy()
    rmin,rmax = np.percentile(norm_img,(percentile_low,percentile_high))
    print('rmin,rmax',rmin,rmax)
    scale = 255/(rmax - rmin)
    print('min' ,rmin,'max',rmax,'scale',scale)
    norm_img = (norm_img - rmin) * scale
    norm_img = np.uint8(norm_img)
    return norm_img  

def load_data(show_FLAG = False):
    # load event preprocessed img
    # ev_img_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
    # ev_img_filename = 'cam1-0noline105ev_canny_rectified1.png'
    # ev_file = ev_img_path + '/' + ev_img_filename
    ev_file = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105ev_rectified_10.png'
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640

    # load frame preprocessed img
    # prepro_frame_img_path = ev_img_path
    # prepro_frame_img_filename = 'cam1-0noline105fr_canny_rectified1.png'
    # pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_f_file = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105fr_rectified.png'
    pre_frame_img = cv2.imread(pre_f_file, 0)  # shape (1536,2048)  3.2times of 480

    if ev_img is None or pre_frame_img is None:
        print('Error: Could not load image')
        quit()
    
    if show_FLAG == True:
        cv2.imshow('pre_frame_img',pre_frame_img)
        cv2.imshow('ev_img',ev_img)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 
    return pre_frame_img, ev_img

def compute_NCC(pre_frame_img,ev_img, template_size):   # input stepsize template_size
    img_height,img_width = pre_frame_img.shape
    stepsize = 1
    step_num = 0
    wrong_num = 0
    test_wr_num =0
    disparity_map = np.zeros((img_height,img_width), np.float64)
    template_width, template_height = template_size 
    for x in range(int(template_width * 0.5),int(img_width - template_width * 0.5),stepsize):
        for y in range(int(template_height * 0.5),int(img_height - template_height * 0.5),stepsize):
    # for x in range(175,int(img_width - template_width * 0.5),stepsize):
    #     for y in range(159,int(img_height - template_height * 0.5),stepsize):
            step_num = step_num + 1
            template_c = (x,y)
            top = template_c[1] -int(template_height * 0.5)
            left = template_c[0] - int(template_width * 0.5)
            template = pre_frame_img[top:top + template_height, left:left + template_width]   # TODO top down left right    

            # NCC_cross correlation   only on the epipolar line 
            ev_epi_img = ev_img[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            e_max_x = max_loc[0] + int(template_width * 0.5) 
            disp = template_c[0] - e_max_x

            if disp < 0 or disp > 100 :   # TODO need to fill in the hole
            # if max_val < 0.45 or disp < 0 or disp > 100 :
                disp = 0
                test_wr_num = test_wr_num+1

            disparity_map[y,x] = disp

    print('step_num',step_num,'wrong_num',wrong_num,'max_val < 0.45',test_wr_num)
    return disparity_map

def compu_NCC(pre_frame_img,ev_img,template_height = 20):
    # template_height = 20
    template_width = template_height
    disparity_map = compute_NCC(pre_frame_img,ev_img, (template_height,template_width))  
    disp_norm = img_normalization(disparity_map) 
    return disp_norm

def compare_window_size(pre_frame_img,ev_img):  # the window size is the template size 
    window20 = compu_NCC(pre_frame_img,ev_img,template_height=20)
    window40 = compu_NCC(pre_frame_img,ev_img,template_height=40)
    window60 = compu_NCC(pre_frame_img,ev_img,template_height=60)
    window80 = compu_NCC(pre_frame_img,ev_img,template_height=80)
    return [window20,window40,window60,window80]

# 左视差图是左边缺一块 右视差图是右边缺一块 
def compute_pyramid_NCC(pre_frame_img,ev_img, template_size = (60,60)):   # 60 & 40
    img_height,img_width = pre_frame_img.shape
    stepsize = 1
    step_num = 0
    wrong_num = 0
    test_wr_num =0
    disparity_map = np.zeros((img_height,img_width), np.float16)  #  change from flaot64 to float16
    template_width, template_height = template_size 
    mask = np.zeros((img_height,img_width), np.uint8)
    for x in range(int(template_width * 0.5),int(img_width - template_width * 0.5),stepsize):
        for y in range(int(template_height * 0.5),int(img_height - template_height * 0.5),stepsize):
    # for x in range(175,int(img_width - template_width * 0.5),stepsize):
    #     for y in range(159,int(img_height - template_height * 0.5),stepsize):
            step_num = step_num + 1
            template_c = (x,y)
            # print(template_c)
            top = template_c[1] -int(template_height * 0.5)
            left = template_c[0] - int(template_width * 0.5)
            template = pre_frame_img[top:top + template_height, left:left + template_width]   # TODO top down left right    

            # NCC_cross correlation   only on the epipolar line 
            ev_epi_img = ev_img[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            e_max_x = max_loc[0] + int(template_width * 0.5) 
            disp = template_c[0] - e_max_x
            if disp < 0 or disp > 100 :
                disparity_map[y,x] = 0
                mask[y,x] = 1
                # print("disp",disp,'================')
                continue  

            # #debug
            # ev_img_color = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
            # cv2.rectangle(ev_img_color, (max_loc[0],top), (max_loc[0] + template_width,top + template_height), (0,255,255), 1)   # yellow
            # # cv2.imshow('ev_img_find',ev_img_color)

            # color_frame_patch = cv2.cvtColor(pre_frame_img, cv2.COLOR_GRAY2BGR)
            # cv2.circle(color_frame_patch,template_c,2,(0,0,255),-1)
            # cv2.rectangle(color_frame_patch,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
            # # cv2.imshow('color_frame_patch',color_frame_patch)

            template_height_small = int(template_height * 2 / 3) # smaller window size to compute finer disparity
            template_width_small = template_height_small
            top_small = template_c[1] -int(template_height_small * 0.5)
            left_small = template_c[0] - int(template_width_small * 0.5)
            template_small = pre_frame_img[top_small:top_small + template_height_small, left_small:left_small + template_width_small]

            finer_range = 10 + int(template_width_small * 0.5)
            ev_epi_img_small = ev_img[top_small:top_small+template_height_small,e_max_x-finer_range:e_max_x+finer_range]  # top down left right
            # cv2.imshow('ev_epi_img_small',ev_epi_img_small)

            epi_result_small = cv2.matchTemplate(ev_epi_img_small, template_small, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val_small, _, max_loc_small = cv2.minMaxLoc(epi_result_small)
            e_max_x_small = e_max_x-finer_range + int(template_width_small * 0.5) + max_loc_small[0]    # left + half_template + max
            # print("e_max_x",e_max_x,'e_max_x_small',e_max_x_small)

            # #debug
            # ev_img_color = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
            # cv2.circle(ev_img_color,([e_max_x_small,template_c[1]]),2,(0,255,255),-1)
            # cv2.rectangle(ev_img_color, (e_max_x_small-int(template_width_small * 0.5),top_small), (e_max_x_small + int(template_width_small * 0.5),top_small + template_height_small), (0,255,255), 1)   # yellow
            # cv2.imshow('ev_img_find1',ev_img_color)

            # # color_frame_patch = cv2.cvtColor(pre_frame_img, cv2.COLOR_GRAY2BGR)
            # cv2.rectangle(color_frame_patch,  (left_small, top_small), (left_small + template_width_small, top_small + template_height_small),(0,0,255), 1)  # red
            # cv2.imshow('color_frame_patch1',color_frame_patch)
            # cv2.imshow('combine',cv2.hconcat([color_frame_patch,ev_img_color]))

            # k = cv2.waitKey(0)
            # if k == 27:         # ESC
            #     cv2.destroyAllWindows() 

            
            disp2 = template_c[0] - e_max_x_small
            # print('disp',disp,'disp2',disp2)

            if disp2 < 0 or disp2 > 100 :   # TODO need to fill in the hole
            # if max_val < 0.45 or disp < 0 or disp > 100 :
                disp2 = 0
                test_wr_num = test_wr_num+1

            disparity_map[y,x] = disp2

    print('step_num',step_num,'wrong_num',wrong_num,'disp < 0 or disp > 100',test_wr_num)
    ret, bi_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    disparity_map_uint8 = np.uint8(disparity_map)
    cv2.imwrite(save_path + '/' + 'origin_disparitymap' + '.png',disparity_map_uint8)

    # inpainted = cv2.inpaint(disparity_map,mask,3,cv2.INPAINT_TELEA)
    # inpainted_norm = img_normalization(inpainted)
    # cv2.imshow('inpainted_norm',inpainted_norm)
    # cv2.imshow('bi_mask',bi_mask)
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     # cv2.imwrite(save_path + '/' + 'pyramid460_medianblur' + '.png',disp_norm)
    #     cv2.destroyAllWindows()   

    return disparity_map    # ? 0-255 cause fault unit8

if __name__ == '__main__':
    save_path = '/Users/cainan/Desktop/Project/data/processed/disparity'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    '''
    LOAD DATA
    '''
    # pre_frame_img, ev_img_10 = load_data()
    # pre_frame_img, ev_img_120 = load_data2()
    pre_frame_img, ev_img = load_data()

    '''
    processing 
    '''
    disparity_map = compute_pyramid_NCC(pre_frame_img,ev_img)
    disp_norm = img_normalization(disparity_map) 


    # medianblur_disp = cv2.medianBlur(disparity_map,3)
    # disp_norm = img_normalization(medianblur_disp) 

    # img_list =  compare_window_size(pre_frame_img,ev_img_10)
    # img_combine = []
    # for i,img in enumerate(img_list):
    #     if i % 2 == 1:
    #         print(i)
    #         vconcat_img = cv2.vconcat([img_list[i-1],img_list[i]]) 
    #         img_combine.append(vconcat_img)
    #         # cv2.imshow('vconcat_img',vconcat_img)
    #         # k = cv2.waitKey(0)
    #         # if k == 27:         # ESC
    #         #    cv2.destroyAllWindows()
    #     # else:
    # img_combine = cv2.hconcat(img_combine)
    
    cv2.imshow("disp_norm",disp_norm)
    
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        cv2.imwrite(save_path + '/' + 'pyramid460_medianblur' + '.png',disp_norm)
        cv2.destroyAllWindows()   








