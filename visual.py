import numpy as np
import cv2
import src.HyStereo.utilities as ut
from SGBM2 import compare_window_size
# from src.HyStereo.preprocess6 import Preprocess
# import os
from stereo_RecDisWarp_EF import sobel_x,loadEvFrame,rectifyEF
import matplotlib.pyplot as plt



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

def epi_visual():
    # !!use gray frame, not sobel ✅
    save_path = '/Users/cainan/Desktop/Project/prophesee-automotive-dataset-toolbox-master/visual4report/cross-correlation'
    save_path= save_path + '/'

    line_width = 5
    red = (0,0,255)
    blue = (255,0,0)

    # same area in two images with red line
    [frame_image, event_image] = loadEvFrame(show_FLAG = False)  # original images with different resolution
    # Sobel_x_frame = sobel_x(frame_image, show_FLAG = False) # sobel fileter for frame

    rectifiedImgList,rectifiedParas1,rectifiedParas2  = rectifyEF(frame_image, event_image, show_FLAG = False)
    [frame_image_rectified,event_image_rectified] = rectifiedImgList
    # cv2.imwrite(save_path + '/' + 'rectified_frame.png',frame_image_rectified)
    # cv2.imwrite(save_path + '/' + 'rectified_event.png',event_image_rectified)
    # quit()

    template_c = [131, 201]   # [x,y] 
    # template_c = (97,186)
    template_height = 40 
    template_width = 40
    top =template_c[1] -int( template_height * 0.5)
    left = template_c[0] - int(template_width * 0.5)
    template = frame_image_rectified[top:top + template_height, left:left + template_width]   # top down left right    

    ## big image
    color_frame = cv2.cvtColor(frame_image_rectified, cv2.COLOR_GRAY2BGR)
    cv2.line(color_frame,(0, top),(640, top ),red,line_width) 
    cv2.line(color_frame,(0, top+ template_height),(640, top + template_height),red,line_width)  
    # cv2.rectangle(color_frame,  (left, top), (left + template_width, top + template_height),red, line_width)  # red
    # cv2.imshow('color_frame',color_frame)

    color_event = cv2.cvtColor(event_image_rectified, cv2.COLOR_GRAY2BGR)
    cv2.line(color_event,(0, top),(640, top ),red,line_width) 
    cv2.line(color_event,(0, top+ template_height),(640, top + template_height),red,line_width)  

    cv2.imwrite(save_path + 'fr_search_area.png',color_frame)
    cv2.imwrite(save_path + 'ev_search_area.png',color_event)
    # quit()

    ## line image
    ev_epi_img = event_image_rectified[top:top + template_height,:] # top down left right; shape (20,640)
    color_ev_epi_img = cv2.cvtColor(ev_epi_img, cv2.COLOR_GRAY2BGR)

    fr_epi_img = frame_image_rectified[top:top + template_height,:] # top down left right; shape (20,640)
    color_fr_epi_img = cv2.cvtColor(fr_epi_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color_fr_epi_img,  (left+2, 1), (left + template_width-2, template_height-1),red, 2)  # red
    # cv2.line(color_fr_epi_img,(0, top),(640, top ),red,line_width) 
    cv2.circle(color_fr_epi_img,(template_c[0],int(template_width * 0.5)),4,(0,0,255),-1)

    cv2.imwrite(save_path + 'ev_epi_img.png',ev_epi_img)
    cv2.imwrite(save_path + 'fr_epi_img.png',color_fr_epi_img)
    # quit()


    epi_result = cv2.matchTemplate(ev_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
    _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
    e_max_x = max_loc[0] + int(template_width * 0.5) 
    disp = e_max_x - template_c[0]
    
    cv2.line(color_ev_epi_img,(e_max_x, 5),(e_max_x, template_height ),blue,2) 
    cv2.imwrite(save_path + 'ev_epi_img.png',color_ev_epi_img)


    # cv2.imshow('fr_epi_img',color_fr_epi_img)
    # cv2.imshow('color_ev_epi_img',color_ev_epi_img)
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     cv2.destroyAllWindows()    

    plot(template_width, epi_result[0])

def warp_visual():
    '''
    sobelx frame✅
    left disparity  -normalise ✅
    warped frame need sobel-x to warp
    '''
    save_path = '/Users/cainan/Desktop/Project/prophesee-automotive-dataset-toolbox-master/visual4report/warp'
    save_path = save_path + '/'
    ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/cutoff_10/event105.png'
    frame_imgFile = '/Users/cainan/Desktop/Project/data/01_simple/png/105.png'

    leftImageFile = frame_imgFile  # left!!
    rightImageFile = ev_imgFile     # right!!
    # get them into same size
    [frame_image, event_image] = ut.parse_StereoImage(leftImageFile, rightImageFile)
    Sobel_x_frame = sobel_x(frame_image, show_FLAG = False)
    rectifiedImgList,rectifiedParas1,rectifiedParas2  = rectifyEF(frame_image, event_image, show_FLAG = True)
    [frame_image_rectified,event_image_rectified] = rectifiedImgList


    # left_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_left_disparity.png'
    # left_disparity_EF = cv2.imread(left_disparity_EF_file,0)
    # left_disparity_EF_norm = ut.img_normalization(left_disparity_EF)

    cv2.imwrite(save_path + 'frame_image_rectified.png',frame_image_rectified)
    # cv2.imwrite(save_path + 'left_disparity_EF_norm.png',left_disparity_EF_norm)

def disp_visual():
    right_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_right_disparity.png'
    right_disparity_EF = cv2.imread(right_disparity_EF_file,0)  
    right_disparity_EF_norm = ut.img_normalization(right_disparity_EF) 

def compare_visual():
    save_path = '/Users/cainan/Desktop/Project/prophesee-automotive-dataset-toolbox-master/visual4report/compare_windowsize'
    save_path = save_path + '/'
    ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/cutoff_10/event105.png'
    frame_imgFile = '/Users/cainan/Desktop/Project/data/01_simple/png/105.png'

    leftImageFile = frame_imgFile  # left!!
    rightImageFile = ev_imgFile     # right!!
    # get them into same size
    [frame_image, event_image] = ut.parse_StereoImage(leftImageFile, rightImageFile)
    # Sobel_x_frame = sobel_x(frame_image, show_FLAG = False)
    rectifiedImgList,rectifiedParas1,rectifiedParas2  = rectifyEF(frame_image, event_image, show_FLAG = False)
    [frame_image_rectified,event_image_rectified] = rectifiedImgList

    # ut.left_disparityPy(frame_image_rectified, event_image_rectified,show_FLAG=True)
    img_list = compare_window_size(frame_image_rectified,event_image_rectified)
    name_list = ['window20','window40','window60','window80']
    # name_list = ['window80']    
    for name,img in zip(name_list,img_list):
        cv2.imwrite(save_path + name + '.png',img)

def block_diagram():
    save_path = '/Users/cainan/Desktop/Project/prophesee-automotive-dataset-toolbox-master/visual4report/block_diagram'
    save_path = save_path + '/'

    ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/cutoff_10/event105.png'
    frame_imgFile = '/Users/cainan/Desktop/Project/data/01_simple/png/105.png'

    leftImageFile = frame_imgFile  # left!!
    rightImageFile = ev_imgFile     # right!!
    # get them into same size
    
    [frame_image, event_image] = ut.parse_StereoImage(leftImageFile, rightImageFile)
    rightImg = cv2.imread(rightImageFile,0)
    cv2.imwrite(save_path + '2_resized_frame_img.png',frame_image)  # ②
    cv2.imwrite(save_path + '3_recon_ev.png',event_image)  # ③

    rectifiedImgList,rectifiedParas1,rectifiedParas2  = rectifyEF(frame_image, event_image, show_FLAG = False)
    for i,img in enumerate(rectifiedImgList):  # ④ ⑤
        cv2.imwrite(save_path + str(i+4) + '_rectification' + '.png',img)

    # ⑥ NCC use the result from compare_visual

    # ⑦ image pyramid
    left_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_left_disparity.png'
    left_disparity_EF = cv2.imread(left_disparity_EF_file,0)
    right_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_right_disparity.png'
    right_disparity_EF = cv2.imread(right_disparity_EF_file,0)   
    left_disparity_EF_norm = ut.img_normalization(left_disparity_EF)
    right_disparity_EF_norm = ut.img_normalization(right_disparity_EF)
    cv2.imwrite(save_path + '7_left_disparity_EF_norm.png',left_disparity_EF_norm)
    cv2.imwrite(save_path + '7_right_disparity_EF_norm.png',right_disparity_EF_norm)

    # ⑧ warp  use load_sobel_rect_warp()

def rectify_visual():

    ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/cutoff_10/event105.png'
    frame_imgFile = '/Users/cainan/Desktop/Project/data/01_simple/png/105.png'
    # frame_imgFile = '/Users/cainan/Desktop/Project/data/recon_evImg/complex/frame231.png'

    leftImageFile = frame_imgFile
    rightImageFile = ev_imgFile
    leftImg = cv2.imread(leftImageFile,0)
    rightImg = cv2.imread(rightImageFile,0)
    leftImg = cv2.resize(leftImg,(rightImg.shape[1],rightImg.shape[0]), interpolation=cv2.INTER_AREA)   # INTER_AREA
    save_path = '/Users/cainan/Desktop/Project/prophesee-automotive-dataset-toolbox-master/visual4report/rectify'
    save_path = save_path + '/'
    cv2.imwrite(save_path + 'resized_frame.png', leftImg)
    cv2.imwrite(save_path + 'ev_recon.png', rightImg) 

def resize_img_readme():
    imfuse_file = '/Users/cainan/Desktop/Stereo_event_Disparity_Estimation/images/imfuse_8.png'
    imfuse = cv2.imread(imfuse_file)
    h, w = imfuse.shape[:2]
    resized_imfuse = cv2.resize(imfuse, (int(w/2),int(h/2)), interpolation=cv2.INTER_AREA)

    illus_file = '/Users/cainan/Desktop/Stereo_event_Disparity_Estimation/images/illustration_warping.png'
    illustration_warping = cv2.imread(illus_file)
    h, w = illustration_warping.shape[:2]
    resized_illustration_warping = cv2.resize(illustration_warping, (int(w/2),int(h/2)), interpolation=cv2.INTER_AREA)
    # cv2.imshow('new_img', resized_imfuse)

    save_path = '/Users/cainan/Desktop/Stereo_event_Disparity_Estimation/images'
    save_path = save_path + '/'

    cv2.imwrite(save_path + 'resized_illustration_warping.png', resized_illustration_warping)
    cv2.imwrite(save_path + 'resized_imfuse2.png',resized_imfuse)


if __name__ == '__main__':

    # epi_visual()
    # warp_visual()
    # compare_visual()
    # block_diagram()
    # rectify_visual()
    resize_img_readme()
    print('end')


