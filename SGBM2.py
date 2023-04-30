import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt

'''
until 11.12.2022 google document update 
'''

def img_normalization(img,percentile_low = 0.05,percentile_high = 99.95):
    norm_img = img.copy()
    rmin,rmax = np.percentile(norm_img,(percentile_low,percentile_high))
    scale = 255/(rmax - rmin)
    print('min' ,rmin,'max',rmax,'scale',scale)
    norm_img = (norm_img - rmin) * scale
    norm_img = np.uint8(norm_img)
    return norm_img  

def compute_NCC(pre_frame_img,ev_img, template_size):   # input stepsize template_size
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

            # NCC_cross correlation   only on the epipolar line 
            ev_epi_img = ev_img[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            e_max_x = max_loc[0] + int(template_width * 0.5) 
            disp = template_c[0] - e_max_x

            if disp < 0 or disp > 90 :
                # print(disp)
                disp = 0
                # disparity_map[y,x] = 0
                # test_wr_num = test_wr_num+1

            disparity_map[y,x] = disp

    print('step_num',step_num,'wrong_num',wrong_num)
    # print(np.max(disparity_map))
    return disparity_map    # ? 0-255 cause fault unit8

def SGBM_disparity(pre_frame_img, ev_img):
    blockSize = 8 # 15
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,  
                                   numDisparities=64,  # 16
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)   # STEREO_SGBM_MODE_HH4 fast  STEREO_SGBM_MODE_HH slow
    # compute disparity
    disparity = stereo.compute(pre_frame_img, ev_img)
    disparity = disparity.astype(np.float32) / 16.
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 -> max_disparity
    # as disparity=-1 means no disparity available
    # max_disparity = 128
    # _, disparity = cv2.threshold(
    #     disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    # disparity_scaled = (disparity / 16.).astype(np.uint8)
    
    return disp

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

def load_data2(show_FLAG = False):  # cutoff 120
    # load event preprocessed img
    ev_file = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105ev_rectified_120.png'
    ev_img_120 = cv2.imread(ev_file, 0)  # shape (480,640

    # load frame preprocessed img
    pre_f_file = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105fr_rectified.png'
    pre_frame_img = cv2.imread(pre_f_file, 0)  # shape (1536,2048)  3.2times of 480

    if ev_img_120 is None or pre_frame_img is None:
        print('Error: Could not load image')
        quit()
    
    if show_FLAG == True:
        cv2.imshow('pre_frame_img',pre_frame_img)
        cv2.imshow('ev_img_120',ev_img_120)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 
    return pre_frame_img, ev_img_120

def compu_NCC(pre_frame_img,ev_img,template_height = 20):
    # template_height = 20
    template_width = template_height
    disparity_map = compute_NCC(pre_frame_img,ev_img, (template_height,template_width))  
    disp_norm = img_normalization(disparity_map) 
    return disp_norm

def compare_10_120_NCC_SGBM(pre_frame_img,ev_img_10,ev_img_120):
    SGBM_disp_10 = SGBM_disparity(pre_frame_img, ev_img_10)
    SGBM_disp_120 = SGBM_disparity(pre_frame_img, ev_img_120)

    # template_height = 20
    # template_width = template_height
    # disparity_map = compute_NCC(pre_frame_img,ev_img_10, (template_height,template_width))  
    # disp_norm = img_normalization(disparity_map) 
    NCC_disp_10 = compu_NCC(pre_frame_img,ev_img_10)
    NCC_disp_120 = compu_NCC(pre_frame_img,ev_img_120)
    return [SGBM_disp_10,SGBM_disp_120,NCC_disp_10,NCC_disp_120]
    
def compare_window_size(pre_frame_img,ev_img):
    window20 = compu_NCC(pre_frame_img,ev_img,template_height=20)
    window40 = compu_NCC(pre_frame_img,ev_img,template_height=40)
    window60 = compu_NCC(pre_frame_img,ev_img,template_height=60)
    window80 = compu_NCC(pre_frame_img,ev_img,template_height=80)
    return [window20,window40,window60,window80]

    # return [window20,window40,window60,window80]

if __name__ == '__main__':
    save_path = '/Users/cainan/Desktop/Project/data/processed/disparity'
    '''
    LOAD DATA
    '''
    pre_frame_img, ev_img_10 = load_data()
    # pre_frame_img, ev_img_120 = load_data2()

    '''
    processing 
    '''
    # img_list = compare_10_120_NCC_SGBM(pre_frame_img,ev_img_10,ev_img_120)
    img_list =  compare_window_size(pre_frame_img,ev_img_10)
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
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        cv2.imwrite(save_path + '/' + 'compareWindow24680' + '.png',img_combine)
        cv2.destroyAllWindows()   








