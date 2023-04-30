import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt

'''
chinese version
until 11.12.2022 google document update 
'''

def show_debug(img_name,img,save_pa = None):  # img_name,img
    cv2.imshow(img_name,img)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        if save_pa is not None:
            cv2.imwrite(save_path + '/' + 'tsukuba_NCC,LRC_check' + '.png',img)
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

def together_normalization(img1,img2,percentile_low = 0.05,percentile_high = 99.95):
    rmin1,rmax1= np.percentile(img1,(percentile_low,percentile_high))
    rmin2,rmax2= np.percentile(img2,(percentile_low,percentile_high))
    print('rmin1,rmax1',rmin1,rmax1)
    print('rmin2,rmax2',rmin2,rmax2)
    scale1 = 255/(rmax1 - rmin1)
    scale2 = 255/(rmax2 - rmin2)
    
    norm_img1 = img1.copy()
    norm_img2 = img2.copy()
    if scale1 < scale2:  # 用差距小的那个
        print('1')
        norm_img1 = (norm_img1 - rmin2) * scale2
        norm_img1 = np.uint8(norm_img1)
        norm_img2 = (norm_img2 - rmin2) * scale2
        norm_img2 = np.uint8(norm_img2)
    if scale1 >= scale2:  # 用差距小的那个
        print('2')
        norm_img1 = (norm_img1 - rmin1) * scale1
        norm_img1 = np.uint8(norm_img1)
        norm_img2 = (norm_img2 - rmin1) * scale1
        norm_img2 = np.uint8(norm_img2)
    return norm_img1,norm_img2

def compute_disparity(pre_frame_img,ev_img, template_size):   # input stepsize template_size
    img_height,img_width = pre_frame_img.shape
    stepsize = 1
    step_num = 0
    wrong_num = 0
    test_wr_num =0
    disparity_map = np.zeros((img_height,img_width), np.float64)
    nonmisremove_map = np.zeros((img_height,img_width), np.float64)
    template_width, template_height = template_size 
    for x in range(int(template_width * 0.5),int(img_width - template_width * 0.5),stepsize):
        for y in range(int(template_height * 0.5),int(img_height - template_height * 0.5),stepsize):
    # for x in range(175,int(img_width - template_width * 0.5),stepsize):
    #     for y in range(159,int(img_height - template_height * 0.5),stepsize):
            step_num = step_num+1
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

            # check disp 
            FLAG = True

            if max_val < 0.5 or disp < 0 or disp > 100 :
                # print(disp)
                disp = 0
                disparity_map[y,x] = 0
                test_wr_num = test_wr_num+1
                # FLAG = False
            elif FLAG == True:
                # 左右一致性检测 threshold一般取为1或者2。
                # print('1')
                # nonmisremove_map[y,x] = disp
                check_patch = ev_epi_img[:,max_loc[0]:max_loc[0] + template_width]
                frame_epi_img = pre_frame_img[top:top + template_height,:]
                check_result = cv2.matchTemplate(frame_epi_img, check_patch, cv2.TM_CCOEFF_NORMED)
                _, check_max_val, _, check_max_loc = cv2.minMaxLoc(check_result)
                threshold = 2 + int(img_width * 0.003) # tsukuba 3
                if abs(check_max_loc[0] -left) >= threshold:
                    # print("check_max_loc",check_max_loc,'(left,top)',(left,top))
                    # print("check_max_loc",check_max_loc,'(left,top)',(left,top))
                    # print('max_val',max_val)
                    # print('check_max_val',check_max_val)
                    # print("check_max_loc[0] -left",check_max_loc[0] -left)
                    # print(disp)
                    # if max_val > 0.9 and check_max_val > 0.9:
                    #     ev_img_color = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
                    #     cv2.rectangle(ev_img_color, (max_loc[0],top), (max_loc[0] + template_width,top + template_height), (0,255,255), 1)   # yellow
                    #     cv2.imshow('ev_img_find',ev_img_color)

                    #     color_frame_patch = cv2.cvtColor(pre_frame_img, cv2.COLOR_GRAY2BGR)
                    #     # cv2.rectangle(color_frame_patch,  (check_max_loc[0], top), (check_max_loc[0] + template_width, top + template_height),(255,255,0), 1)  # blue
                    #     cv2.rectangle(color_frame_patch,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
                    #     cv2.imshow('color_frame_patch',color_frame_patch)
                    #     k = cv2.waitKey(0)
                    #     if k == 27:         # ESC
                    #         cv2.destroyAllWindows()    
                    wrong_num = wrong_num + 1  # 1227
                    disparity_map[y,x] = 0
                    # disp = 0
                else:
                    # print("2")
                    # disparity_map[y,x] = disp
                    disparity_map[y,x] = disp
            # print('1')
            nonmisremove_map[y,x] = disp

    print('step_num',step_num,'wrong_num',wrong_num,'max_val < 0.45',test_wr_num)
    # print(np.max(disparity_map))
    return disparity_map,nonmisremove_map    # ? 0-255 cause fault unit8

if __name__ == '__main__':
    save_path = '/Users/cainan/Desktop/Project/data/processed/disparity'
    '''
    LOAD DATA
    '''
    # load event preprocessed img
    ev_img_path ='/Users/cainan/Desktop/Project/data/tsukuba'
    # ev_img_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
    # ev_img_filename = 'cam1-0noline107ev_rectified_alpha_0.png'
    ev_img_filename = 'scene1.row3.col5.ppm'

    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640

    # load frame preprocessed img
    prepro_frame_img_path = ev_img_path
    # prepro_frame_img_filename = 'cam1-0noline107fr_rectified_alpha_0.png'
    prepro_frame_img_filename = 'scene1.row3.col3.ppm'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # shape (1536,2048)  3.2times of 480
    # pre_frame_img = cv2.cvtColor(pre_frame_img, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    if ev_img is None or pre_frame_img is None:
        print('Error: Could not load image')
        quit()
    

    # cv2.imshow('pre_frame_img',pre_frame_img)
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     cv2.destroyAllWindows() 
    '''
    processing 
    '''
    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,  
                                   numDisparities=64,  # 调小numDisparities会降低精度，但提高速度。注意：numDisparities需能被16整除
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
    # disparity = disparity.astype(np.float32) / 16.
    # disparity = stereo.compute(ev_img, pre_frame_img)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # disp = np.divide(disparity.astype(np.float32), 16.) #除以16得到真实视差图

    # disp = img_normalization(disparity)
    cv2.imshow('disp',disp)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        cv2.imwrite(save_path + '/' + 'SGBM' + '.png',disp)
        cv2.destroyAllWindows()   


    quit()



    # set
    # template_height = 7
    template_height = 20
    template_width = template_height

    disparity_map,nonmisremove_map = compute_disparity(pre_frame_img,ev_img, (template_height,template_width))  # TODO 两张图联合normalization
    # disparity_map_norm = img_normalization(disparity_map) 
    # nonmisremove_map_norm = img_normalization(nonmisremove_map) 
    # disparity_map_norm = img_normalization(disparity_map,percentile_low = 0.1,percentile_high = 99.5)  # TODO 热力图
    # im_h = cv2.hconcat([disparity_map, nonmisremove_map])
    # print('-----------------')
    # im_h_norm = img_normalization(im_h) 


    disparity_map_norm, nonmisremove_map_norm = together_normalization(disparity_map,nonmisremove_map,percentile_low = 0.05,percentile_high = 99.95)

    im_norm_h = cv2.hconcat([nonmisremove_map_norm, disparity_map_norm])
    # cv2.imshow("disparity_map,nonmisremove_map",im_norm_h)
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     cv2.destroyAllWindows()   
    show_debug("disparity_map,nonmisremove_map",im_norm_h,save_pa = save_path)  # img_name,img 34s
    # show_debug('disparity_map',disparity_map,save_pa = save_path)






