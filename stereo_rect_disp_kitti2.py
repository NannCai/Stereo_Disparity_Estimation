import numpy as np
import cv2
import src.HyStereo.utilities as ut
# import matplotlib.pyplot as plt
# import os

'''
load data  (option:preprocessing)✅
do the rectify first✅  --save img
then compute the disparity✅  --save img
finnally  2D-3D-2D point  --save img

first use kitti to make sure the 
'''
'''
⭕️  2D-3D-2D point 
1. 2D to 3D     reprojectImageTo3D
        Input left disparity, Q, left image
        Output 3d point
        Q	Output 4x4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
        ?? if I want to use the right disp and right image how about Q
        ?? Q inverse?
2. 3D to 2D     using P2
        P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, 
        i.e. it projects points_3d given in the ...rectified first camera coordinate system into the rectified second camera's image...
'''


def loadKitti(show_FLAG = False):
    # cam0 - left   cam1 - right
    cam0_imgFile = '/Users/cainan/Desktop/Project/data/kitti/kitti_2011_09_26/2011_09_26_drive_0001_extract/image_00/data/0000000000.png'
    cam1_imgFile = '/Users/cainan/Desktop/Project/data/kitti/kitti_2011_09_26/2011_09_26_drive_0001_extract/image_01/data/0000000000.png'

    [cam0_image, cam1_image] = ut.loadStereoImage(cam0_imgFile, cam1_imgFile, show_FLAG = show_FLAG)
    return [cam0_image, cam1_image]

def rectifyKitti(cam0_image, cam1_image, show_FLAG = False):
    print('rectifyKitti')
    print(cam0_image.shape)
    cam0 = np.array([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                                             [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                                              [0.000000e+00, 0.000000e+00, 1.000000e+00 ]])
    cam1 = np.array([[9.895267e+02, 0.000000e+00, 7.020000e+02],
                                                               [0.000000e+00, 9.878386e+02, 2.455590e+02],
                                                               [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    dist0 = np.array([[-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02]])
    dist1 = np.array([[-3.644661e-01, 1.790019e-01, 1.148107e-03, -6.298563e-04, -5.314062e-02]])
    cam_0_1_R = np.array([[9.993513e-01, 1.860866e-02, -3.083487e-02],
                                     [-1.887662e-02, 9.997863e-01, -8.421873e-03],
                                     [3.067156e-02, 8.998467e-03, 9.994890e-01]])                              
    cam_0_1_t = np.array([[-5.370000e-01], [4.822061e-03], [-1.252488e-02]])
    # rectify_w = 1392
    # rectify_h = 512
    rectify_h, rectify_w = cam0_image.shape
    print('cam0_image.shape',cam0_image.shape)
    
    # height,width,_ = points_3d.shape

    [cam0_image_rectified,cam1_image_rectified],[R0, R1, P0, P1, Q] = ut.rectifyImage(cam0, dist0, 
                                                                                                cam1, dist1, 
                                                                                                cam_0_1_R, 
                                                                                                cam_0_1_t, 
                                                                                                cam0_image, cam1_image, 
                                                                                                rectify_w , rectify_h, 
                                                                                                show_FLAG = show_FLAG)

    return [cam0_image_rectified,cam1_image_rectified],[R0, R1, P0, P1, Q]

def load_rect_lrDisp_kitti(stepsize = 2):
    print('load rect lrDisp_kitti')
    [cam0_image, cam1_image] = loadKitti(show_FLAG = False)

    rectifiedImgList,rectifiedParas = rectifyKitti(cam0_image, cam1_image, show_FLAG=False)
    [cam0_image_rectified,cam1_image_rectified] = rectifiedImgList
    [R0, R1, P0, P1, Q] = rectifiedParas

    # [left_disparity,right_disparity] = ut.compute_leftright_disparity(cam0_image_rectified, 
    #                                                                                     cam1_image_rectified,
    #                                                                                     stepsize,
    #                                                                                     show_FLAG = True)
    save_path = '/Users/cainan/Desktop/Project/data/processed/disparity/kitti'
    left_disparity = ut.left_disparityPy(cam0_image_rectified, 
                                                                cam1_image_rectified, 
                                                                template_size = (60,60), 
                                                                show_FLAG = True, 
                                                                verbosity = False, 
                                                                save_path = save_path,
                                                                stepsize = stepsize)


def load_rect_warped_kitti():
    print('load rect warped_kitti')
    '''
    ⭕️  2D-3D-2D point 
    1. 2D to 3D     reprojectImageTo3D
            Input left disparity, Q, left image
            Output 3d point
            Q	Output 4x4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
            ?? if I want to use the right disp and right image how about Q
            ?? Q inverse?
    2. 3D to 2D     using P2
            P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, 
            i.e. it projects points_3d given in the ...rectified first camera coordinate system into the rectified second camera's image...
    '''

    # cam0 - left   cam1 - right
    show_FLAG = False
    [cam0_image, cam1_image] = loadKitti(show_FLAG = show_FLAG)

    rectifiedImgList,rectifiedParas = rectifyKitti(cam0_image, cam1_image, show_FLAG=show_FLAG)
    [cam0_image_rectified,cam1_image_rectified] = rectifiedImgList
    [R0, R1, P0, P1, Q] = rectifiedParas

    # left_disparity_kitti_file = '/Users/cainan/Desktop/Project/data/processed/disparity/left_disparity_origin.png'
    left_disparity_kitti_file = '/Users/cainan/Desktop/Project/data/processed/disparity/kitti/origin_left_disparity_2.png'
    left_disparity_kitti = cv2.imread(left_disparity_kitti_file,0)
    if left_disparity_kitti is None :
        print('Error: Could not load image')
        quit()
    show_FLAG = False
    if show_FLAG == True:
        left_disparity_kitti_norm = ut.img_normalization(left_disparity_kitti)
        cv2.imshow('left_disparity_EF_norm',left_disparity_kitti_norm)
        # cv2.imshow('right_disparity_EF',left_disparity_kitti)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            # warp_save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
            # # cv2.imwrite(warp_save_path + '/' + 'warped_kitti' + '.png',warped_img)
            cv2.destroyAllWindows() 
    # 
    ut.warp(left_disparity_kitti,Q,P1,cam0_image_rectified) 

def test(cam0_image, cam1_image):
    # input para change or not
    # cam0 --- left   cam1---right
    cam0 = np.array([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                                             [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                                              [0.000000e+00, 0.000000e+00, 1.000000e+00 ]])
    cam1 = np.array([[9.895267e+02, 0.000000e+00, 7.020000e+02],
                                                               [0.000000e+00, 9.878386e+02, 2.455590e+02],
                                                               [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    dist0 = np.array([[-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02]])
    dist1 = np.array([[-3.644661e-01, 1.790019e-01, 1.148107e-03, -6.298563e-04, -5.314062e-02]])
    cam_0_1_R = np.array([[9.993513e-01, 1.860866e-02, -3.083487e-02],
                                     [-1.887662e-02, 9.997863e-01, -8.421873e-03],
                                     [3.067156e-02, 8.998467e-03, 9.994890e-01]])                              
    cam_0_1_t = np.array([[-5.370000e-01], [4.822061e-03], [-1.252488e-02]])
    rectify_w = 1392
    rectify_h = 512

    # rectify_h, rectify_w = cam0_image.shape
    '''
    the normal one
    '''
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(cam0, dist0,
                                                                    cam1, dist1,
                                                                    (rectify_w, rectify_h),
                                                                    cam_0_1_R,
                                                                    cam_0_1_t,
                                                                    # flags=cv2.CALIB_ZERO_DISPARITY,
                                                                    alpha=0)
    print('Q',Q)
    cam0MapX, cam0MapY = cv2.initUndistortRectifyMap(cam0, dist0, R0, P0, (rectify_w, rectify_h), cv2.CV_32FC1)
    cam1MapX, cam1MapY = cv2.initUndistortRectifyMap(cam1, dist1, R1, P1, (rectify_w, rectify_h), cv2.CV_32FC1)
    cam0_image_rectified = cv2.remap(cam0_image, cam0MapX, cam0MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    cam1_image_rectified = cv2.remap(cam1_image, cam1MapX, cam1MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    cam0_image_rectified_show = cam0_image_rectified.copy()
    cam1_image_rectified_show = cam1_image_rectified.copy()
    list = [100,159,200,250,300,400]
    for height in list:
        cv2.line(cam0_image_rectified_show, (0, height), (cam1_image.shape[1], height), (255, 255, 255))
        cv2.line(cam1_image_rectified_show, (0, height), (cam1_image.shape[1], height), (255, 255, 255))
    combine = cv2.hconcat([cam0_image_rectified_show,cam1_image_rectified_show])
    # combine2 = cv2.vconcat([cam0_image_rectified_show,cam1_image_rectified_show])
    cv2.imshow('cam0_image_rectified_show,cam1_image_rectified_show',combine)
    save_path = '/Users/cainan/Desktop/Project/data/processed/rectify/with_line'
    cv2.imwrite(save_path + '/' + 'no_flag' + '.png',combine)
    print(combine.shape)
    # cv2.imshow('combine2',combine2)
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     cv2.destroyAllWindows() 


    '''
    the whole inverse one
    '''
    cam_0_1_R_inv = np.linalg.inv(cam_0_1_R)
    cam_0_1_t_inv = - cam_0_1_t
    cam_f2e_R = cam_0_1_R.T
    cam_f2e_t = - np.dot(cam_0_1_R.T,cam_0_1_t)
    R1_, R0_, P1_, P0_, Q_, _, _ = cv2.stereoRectify(cam1, dist1,
                                                                    cam0, dist0,
                                                                    (rectify_w, rectify_h),
                                                                    cam_f2e_R,
                                                                    cam_f2e_t,
                                                                    flags=cv2.CALIB_ZERO_DISPARITY,
                                                                    alpha=0)
    print('Q_',Q_)

    cam0MapX, cam0MapY = cv2.initUndistortRectifyMap(cam0, dist0, R0_, P0_, (rectify_w, rectify_h), cv2.CV_32FC1)
    cam1MapX, cam1MapY = cv2.initUndistortRectifyMap(cam1, dist1, R1_, P1_, (rectify_w, rectify_h), cv2.CV_32FC1)
    cam0_image_rectified = cv2.remap(cam0_image, cam0MapX, cam0MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    cam1_image_rectified = cv2.remap(cam1_image, cam1MapX, cam1MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    cam0_image_rectified_show = cam0_image_rectified.copy()
    cam1_image_rectified_show = cam1_image_rectified.copy()
    list = [100,159,200,250,300,400]
    for height in list:
        cv2.line(cam0_image_rectified_show, (0, height), (cam1_image.shape[1], height), (255, 255, 255))
        cv2.line(cam1_image_rectified_show, (0, height), (cam1_image.shape[1], height), (255, 255, 255))
    combine = cv2.hconcat([cam0_image_rectified_show,cam1_image_rectified_show])
    # combine2 = cv2.vconcat([cam0_image_rectified_show,cam1_image_rectified_show])
    cv2.imshow('inverse',combine)
    print('inverse')
    print(combine.shape)
    # cv2.imshow('combine2',combine2)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        cv2.destroyAllWindows() 


if __name__ == '__main__':
    # load_rect_lrDisp_kitti()  # ✅
    # # use ut.warp in load_rect_warped_kitti to cheke the code
    # load_rect_warped_kitti() # ✅
    # test()

    show_FLAG = True
    [cam0_image, cam1_image] = loadKitti(show_FLAG = False)
    test(cam0_image, cam1_image)
    # rectifiedImgList,rectifiedParas = rectifyKitti(cam0_image, cam1_image, show_FLAG=show_FLAG)








