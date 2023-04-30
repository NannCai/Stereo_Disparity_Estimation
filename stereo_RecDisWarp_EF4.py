import numpy as np
import cv2
import src.HyStereo.utilities as ut
from preprocess6 import Preprocess
import os

'''
⭕️ Event Frame System
TODO preprocesing event function (reconstruction using high pass filter)
TODO set disparity part to be optional and optimize the disparity fuction
loadEvFrame ✅
sobel_x ✅
rectifyEF ✅  : theres two set of rectify for left2right warp and right2left warp  to get two P2
load_lrDisp_EF  ✅
load_rect_warped_EF ✅   
bgr_fusing ✅

load_sobel_rect_warp
'''

def loadEvFrame(show_FLAG = False):
    print('loading Ev Frame...')
    # ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105ev_rectified_10.png'
    ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/cutoff_10/event105.png'
    # frame_imgFile = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105fr_rectified.png'
    frame_imgFile = '/Users/cainan/Desktop/Project/data/01_simple/png/105.png'

    leftImageFile = frame_imgFile  # left!!
    rightImageFile = ev_imgFile     # right!!
    # get them into same size
    [frame_image, event_image] = ut.parse_StereoImage(leftImageFile, rightImageFile, show_FLAG = show_FLAG)
    return [frame_image, event_image] 

def load_PreproImgs(img_dir,image_number, show_FLAG= False):
    files = os.listdir(img_dir)
    ev_file = None
    frame_file = None
    for file in files:
        if 'event' in file and str(image_number) in file:
            ev_file = img_dir + '/' +file
        if 'frame' in file and str(image_number) in file:
            frame_file = img_dir + '/' +file
    if frame_file is None:
        frame_dir = '/Users/cainan/Desktop/Project/data/01_simple/png'
        for file in os.listdir(frame_dir):
            if 'frame' and str(image_number) in file:
                frame_file = frame_dir + '/' +file          

    frame_image = cv2.imread(frame_file,0)
    event_image = cv2.imread(ev_file,0)
    ut.img_show([frame_image,event_image],['frame_image','event_image'],show_FLAG = True)
    return frame_image, event_image

def sobel_x(img, show_FLAG = False):
    print('convert to sobel-x')
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1,0)
    absX = cv2.convertScaleAbs(grad_x)  # back to uint8

    if show_FLAG == True:
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0,1)
        absY = cv2.convertScaleAbs(grad_y)
        combine = cv2.hconcat([absX,absY])
        cv2.imshow('sobel-x,sobel-y',combine)
        # cv2.imshow('sobel-x',absX)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 

    return absX

def rectifyEF(frame_image, event_image, show_FLAG = False):  # TODO theres two set of rectify for left2right warp and right2left warp
    print('rectify EF')
    ev_intri = np.array([[5.549697628637119e+02, 0, 3.195838836743635e+02],
                                                             [0., 5.558964313825462e+02, 2.172033768792723e+02],
                                                              [0., 0., 1.]])
    fr_intri = np.array([[4.054135566221585e+02, 0, 3.142290421594836e+02],
                                                               [0., 4.049340581100494e+02,  2.445772553059500e+02],
                                                               [0., 0., 1.]])
    ev_dist = np.array([[-1.156664910088120e-01, 1.946241433904505e-01,3.025107052357226e-03, 4.913881837816321e-04, 1.768629600684745e-02]])
    fr_dist = np.array([[-2.405663351829244e-01, 1.804700878515662e-01, 3.107546803597012e-03, -4.817144966179097e-04, -9.362895292153335e-02]])
    cam_e2f_R = np.array([[9.998806194689772e-01, 7.303636302427013e-03 , -1.361630298929278e-02],
                                     [-7.653466954692024e-03, 9.996373237376646e-01, -2.581947780597325e-02],
                                     [1.342278860400440e-02, 2.592060738797566e-02, 9.995738846422163e-01]])
    cam_e2f_R = cam_e2f_R.T                         
    cam_e2f_t = np.array([[6.490402887577009e+01], [-4.946543212569138e+00], [-7.130161883182569e+00]])  # ?? 难道不是左右平移吗
    # rectify_w = 640
    # rectify_h = 480
    rectify_h, rectify_w = frame_image.shape
    # print(cam0_image.shape)
    # f2e rectify      
    # cam_e2f_R_inv = np.linalg.inv(cam_e2f_R)
    # cam_e2f_t_inv = -cam_e2f_t
    cam_f2e_R = cam_e2f_R.T
    cam_f2e_t = - np.dot(cam_e2f_R.T,cam_e2f_t)
    [frame_image_rectified,event_image_rectified],[R1, R2, P1, P2, Q] = ut.rectifyImage(fr_intri, fr_dist, 
                                                                                                ev_intri, ev_dist, 
                                                                                                cam_f2e_R, 
                                                                                                cam_f2e_t, 
                                                                                                frame_image, event_image, 
                                                                                                rectify_w , rectify_h, 
                                                                                                show_FLAG = show_FLAG)


    # e2f rectify
    R1_, R2_, P1_, P2_, Q_, _, _ = cv2.stereoRectify(ev_intri, ev_dist,    
                                                                        fr_intri, fr_dist, 
                                                                        (rectify_w, rectify_h),
                                                                        cam_e2f_R, 
                                                                        cam_e2f_t, 
                                                                        # flags=cv2.CALIB_ZERO_DISPARITY,
                                                                        alpha=0)

    return [frame_image_rectified,event_image_rectified],[R1, R2, P1, P2, Q],[R1_, R2_, P1_, P2_, Q_]

def load_lrDisp_EF(show_FLAG = False):
    print('load lrDisp EF')
    left_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_left_disparity.png'
    left_disparity_EF = cv2.imread(left_disparity_EF_file,0)
    right_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_right_disparity.png'
    right_disparity_EF = cv2.imread(right_disparity_EF_file,0)   
    if left_disparity_EF is None or right_disparity_EF is None:
        print('Error: Could not load image')
        quit()
    if show_FLAG == True:
        left_disparity_EF_norm = ut.img_normalization(left_disparity_EF)
        right_disparity_EF_norm = ut.img_normalization(right_disparity_EF)
        combine = cv2.hconcat([left_disparity_EF_norm,right_disparity_EF_norm])
        ut.img_show(combine,'left_disparity_EF_norm,right_disparity_EF_norm',show_FLAG)
        cv2.imshow('left_disparity_EF_norm,right_disparity_EF_norm',combine)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 
    return [left_disparity_EF,right_disparity_EF]

def bgr_fusing(warped_img,rectified_img, show_FLAG = False):
    green = warped_img
    purple = rectified_img  # sobel-x
    imfuse = np.dstack((purple,green,purple))  # the middle one will be green  bgr
    imfuse2 = np.dstack((green,purple,green))
    # C = cv2.merge((purple,green,purple)) # same as dstack

    save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
    # cv2.imwrite(save_path + '/' + 'sobel_x_frame' + '.png',frame_image_rectified)

    if show_FLAG == True:
        # print(green.shape)
        # print(imfuse.shape)
        green = cv2.cvtColor(green,cv2.COLOR_GRAY2BGR)
        combine = cv2.hconcat([green,imfuse,imfuse2])
        cv2.imshow('green,imfuse,imfuse2',combine)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 
    return imfuse


def load_sobel_rect_warp():
    # print('load sobel rect warp')
    pre_FLAG = False
    
    if pre_FLAG == True:
        # preprocess and load a pair of images
        preprocess_ = Preprocess(processe_number = 0)
        img_dir = preprocess_.get_savedir()
        image_number = 100
        frame_image, event_image = load_PreproImgs(img_dir, image_number, show_FLAG= False)
    else:
        # loading
        [frame_image, event_image] = loadEvFrame(show_FLAG = False)

    # generate sobel-x frame image  ？ before rectify or after the rectify
    Sobel_x_frame = sobel_x(frame_image, show_FLAG = False)

    # rectification 
    rectifiedImgList,rectifiedParas1,rectifiedParas2  = rectifyEF(Sobel_x_frame, event_image, show_FLAG = True)
    [frame_image_rectified,event_image_rectified] = rectifiedImgList
    [_, _, _, P2, Q] = rectifiedParas1
    [_, _, _, P2_, _] = rectifiedParas2
    # quit()
    # computing disparity map
    #...

    # warping
    [left_disparity_EF,right_disparity_EF] = load_lrDisp_EF(show_FLAG = True)  # use gray frame to compute but use sobel to compare
    warped_img_rl = ut.warp(right_disparity_EF,Q,P2_,event_image_rectified,show_FLAG = True) # better
    warped_img_lr = ut.warp(left_disparity_EF,Q,P2,frame_image_rectified,show_FLAG = True)  # worse
    # save_path = '/Users/cainan/Desktop/Project/prophesee-automotive-dataset-toolbox-master/visual4report/warp'
    save_path = '/Users/cainan/Desktop/Project/data/processed/warped/3.20'
    save_path = save_path + '/'
    # cv2.imwrite(save_path + '8_warped_img_rl.png',warped_img_rl)
    # cv2.imwrite(save_path + '8_warped_img_lr.png',warped_img_lr)

    print('end')


    # fusing
    imfuse_r2l = bgr_fusing(warped_img_rl,frame_image_rectified,show_FLAG=True)  # real left vs warped left  
    imfuse_l2r = bgr_fusing(warped_img_lr,event_image_rectified,show_FLAG=True)
    cv2.imwrite(save_path + '8_imfuse_left.png',imfuse_r2l)
    cv2.imwrite(save_path + '8_imfuse_right.png',imfuse_l2r)



if __name__ == '__main__':
    # camparing the warp image and the original one
    load_sobel_rect_warp()
    
    
    # preprocess and load a pair of images


    












