import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt

'''
until 02.01.2023 google document update 

compute disparity on all samples✅ but some problem maybe-- use good dataset to test and correct the code 
change the rectifed image from alfha=0✅
solution:
disparity_map from uint8 to float64✅
try to use good dataset to make sure the algorithm is right✅
the result on data tsukuba still have a lot of things to improve the matching result:
1. the metric to see whether the code is improving the result or not
    remove the mismatching
        can do the left right yizhixing  too expensive? ✅  --- make compare into one image using the same scale
2. the result on hybrid sys is bad -- need to find the problem (maybe the method of matching or the quality of ev_img is too bad)

only NCC is ok and it is not that computaional expensive
'''

def show_debug(img_name,img,save_pa = None):  # img_name,img
    cv2.imshow(img_name,img)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        if save_pa is not None:
            cv2.imwrite(save_path + '/' + 'ev_fr_105' + '.png',img)
        cv2.destroyAllWindows()    
    quit()

def img_normalization(img,percentile_low = 0.05,percentile_high = 99.95):
    norm_img = img.copy()
    rmin,rmax = np.percentile(norm_img,(percentile_low,percentile_high))
    scale = 255/(rmax - rmin)
    print('min' ,rmin,'max',rmax,'scale',scale)
    norm_img = (norm_img - rmin) * scale
    norm_img = np.uint8(norm_img)
    return norm_img  

def compute_disparity(pre_frame_img,ev_img, template_size):   # input stepsize template_size
    img_height,img_width = pre_frame_img.shape
    stepsize = 1
    step_num = 0
    wrong_num = 0
    test_wr_num =0
    disparity_map = np.zeros((img_height,img_width), np.float64)
    # nonmisremove_map = np.zeros((img_height,img_width), np.float64)
    template_width, template_height = template_size 
    for x in range(int(template_width * 0.5),int(img_width - template_width * 0.5),stepsize):
        for y in range(int(template_height * 0.5),int(img_height - template_height * 0.5),stepsize):
    # for x in range(175,int(img_width - template_width * 0.5),stepsize):
    #     for y in range(159,int(img_height - template_height * 0.5),stepsize):
            step_num = step_num + 1
            template_c = (x,y)
            top = template_c[1] -int(template_height * 0.5)
            left = template_c[0] - int(template_width * 0.5)
            template = pre_frame_img[top:top + template_height, left:left + template_width]   # top down left right    

            # NCC cross correlation   only on the epipolar line 
            ev_epi_img = ev_img[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            e_max_x = max_loc[0] + int(template_width * 0.5) 
            disp = template_c[0] - e_max_x

            if max_val < 0.45 or disp < 0 or disp > 100 :
                # print(disp)
                disp = 0
                # disparity_map[y,x] = 0
                test_wr_num = test_wr_num+1

            disparity_map[y,x] = disp

    print('step_num',step_num,'wrong_num',wrong_num,'max_val < 0.45',test_wr_num)
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
    # ev_img_filename = 'cam1-0noline107ev_rectified_alpha_0.png'
    ev_img_filename = 'cam1-0noline105ev_canny_rectified1.png'
    # ev_img_filename= 'scene1.row3.col5.ppm'

    ev_file = ev_img_path + '/' + ev_img_filename
    ev_file = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105ev_rectified_10.png'
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640

    # load frame preprocessed img
    prepro_frame_img_path = ev_img_path
    # prepro_frame_img_filename = 'cam1-0noline107fr_rectified_alpha_0.png'
    prepro_frame_img_filename = 'cam1-0noline105fr_canny_rectified1.png'
    # prepro_frame_img_filename = 'scene1.row3.col3.ppm'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_f_file = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105fr_rectified.png'
    pre_frame_img = cv2.imread(pre_f_file, 0)  # shape (1536,2048)  3.2times of 480
    if ev_img is None or pre_frame_img is None:
        print('Error: Could not load image')
        quit()
    
    # cv2.imshow("pre_frame_img",pre_frame_img)
    # cv2.imshow("ev_img",ev_img)
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     cv2.destroyAllWindows()   

    '''
    processing (resize)frame and pick the template from frame
    '''
    # set
    # template_height = 7
    template_height = 20
    template_width = template_height

    disparity_map = compute_disparity(pre_frame_img,ev_img, (template_height,template_width))  

    # disp_norm = cv2.normalize(disparity_map, disparity_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_norm = img_normalization(disparity_map) 
    # Median Filter
    # cv2.medianBlur(disparity_map_norm,3)
    # disparity_map_norm = img_normalization(disparity_map,percentile_low = 0.1,percentile_high = 99.5)  # TODO 热力图
    # im_h = cv2.hconcat([disparity_map, nonmisremove_map])
    # print('-----------------')
    # im_h_norm = img_normalization(im_h) 

    # disparity_map_norm, nonmisremove_map_norm = together_normalization(disparity_map,nonmisremove_map,percentile_low = 0.05,percentile_high = 99.95)

    # im_norm_h = cv2.hconcat([nonmisremove_map_norm, disparity_map_norm])
    cv2.imshow("disparity_map_norm",disp_norm)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        # cv2.imwrite(save_path + '/' + 'ev_fr_canny_temp10' + '.png',disp_norm)
        cv2.destroyAllWindows()   
    # show_debug("disparity_map,nonmisremove_map",im_norm_h,save_pa = save_path)  # img_name,img 34s
    # show_debug('disparity_map',disparity_map,save_pa = save_path)







'''
(in doc)I generate the disparity map for a famous standard dataset: tsukuba
As it is shown on the right of the figure, 
the disparity map only using the ncc algorithm has plenty of wrong points due to mismatching. 
In the left figure I find some mismatching points using the Left-Right Consistency(LRC) check. 
And I set them to zero. Do I need to set values to these points? 
'''