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
    # ev_file = '/Users/cainan/Desktop/Project/data/kitti_2011_09_26/2011_09_26_drive_0001_extract/image_00/data/0000000000.png'
    cam0_imgFile = '/Users/cainan/Desktop/Project/data/kitti_2011_09_26/2011_09_26_drive_0001_extract/image_00/data/0000000000.png'
    # pre_f_file = '/Users/cainan/Desktop/Project/data/kitti_2011_09_26/2011_09_26_drive_0001_extract/image_01/data/0000000000.png'
    cam1_imgFile = '/Users/cainan/Desktop/Project/data/kitti_2011_09_26/2011_09_26_drive_0001_extract/image_01/data/0000000000.png'

    [cam0_image, cam1_image] = ut.loadStereoImage(cam0_imgFile, cam1_imgFile, show_FLAG = show_FLAG)
    return [cam0_image, cam1_image]

def loadEvFrame(show_FLAG = False):
    print('load Ev Frame')
    # ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105ev_rectified_10.png'
    ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/cutoff_10/event105.png'
    # frame_imgFile = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105fr_rectified.png'
    frame_imgFile = '/Users/cainan/Desktop/Project/data/01_simple/png/105.png'

    cam0_imgFile = frame_imgFile  # left!!
    cam1_imgFile = ev_imgFile     # right!!
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
    
    # height,width,_ = points_3d.shape

    [cam0_image_rectified,cam1_image_rectified],[R0, R1, P0, P1, Q] = ut.rectifyImage(cam0, dist0, 
                                                                                                cam1, dist1, 
                                                                                                cam_0_1_R, 
                                                                                                cam_0_1_t, 
                                                                                                cam0_image, cam1_image, 
                                                                                                rectify_w , rectify_h, 
                                                                                                show_FLAG = show_FLAG)

    return [cam0_image_rectified,cam1_image_rectified],[R0, R1, P0, P1, Q]

def rectifyEF(cam0_image, cam1_image, show_FLAG = False):
    # print('rectify EF')
    # cam0 FR  cam1 EV
    cam1 = np.array([[5.549697628637119e+02, 0, 3.195838836743635e+02],
                                                             [0., 5.558964313825462e+02, 2.172033768792723e+02],
                                                              [0., 0., 1.]])
    cam0 = np.array([[4.054135566221585e+02, 0, 3.142290421594836e+02],
                                                               [0., 4.049340581100494e+02,  2.445772553059500e+02],
                                                               [0., 0., 1.]])
    dist1 = np.array([[-1.156664910088120e-01, 1.946241433904505e-01,3.025107052357226e-03, 4.913881837816321e-04, 1.768629600684745e-02]])
    dist0 = np.array([[-2.405663351829244e-01, 1.804700878515662e-01, 3.107546803597012e-03, -4.817144966179097e-04, -9.362895292153335e-02]])
    cam_0_1_R = np.array([[9.998806194689772e-01,     7.303636302427013e-03 ,   -1.361630298929278e-02],
                                     [-7.653466954692024e-03,     9.996373237376646e-01  ,  -2.581947780597325e-02],
                                     [1.342278860400440e-02  ,   2.592060738797566e-02 ,    9.995738846422163e-01]])
    cam_0_1_t = np.array([[6.490402887577009e+01], [-4.946543212569138e+00], [-7.130161883182569e+00]])
    # rectify_w = 640
    # rectify_h = 480
    rectify_h, rectify_w = cam0_image.shape
    # print(cam0_image.shape)

    [cam0_image_rectified,cam1_image_rectified],[R0, R1, P0, P1, Q] = ut.rectifyImage(cam0, dist0, 
                                                                                                cam1, dist1, 
                                                                                                cam_0_1_R, 
                                                                                                cam_0_1_t, 
                                                                                                cam0_image, cam1_image, 
                                                                                                rectify_w , rectify_h, 
                                                                                                show_FLAG = show_FLAG)

    return [cam0_image_rectified,cam1_image_rectified],[R0, R1, P0, P1, Q]

def Q_prime():
    cam1 = np.array([[5.549697628637119e+02, 0, 3.195838836743635e+02],
                                                             [0., 5.558964313825462e+02, 2.172033768792723e+02],
                                                              [0., 0., 1.]])
    cam0 = np.array([[4.054135566221585e+02, 0, 3.142290421594836e+02],
                                                               [0., 4.049340581100494e+02,  2.445772553059500e+02],
                                                               [0., 0., 1.]])
    dist1 = np.array([[-1.156664910088120e-01, 1.946241433904505e-01,3.025107052357226e-03, 4.913881837816321e-04, 1.768629600684745e-02]])
    dist0 = np.array([[-2.405663351829244e-01, 1.804700878515662e-01, 3.107546803597012e-03, -4.817144966179097e-04, -9.362895292153335e-02]])
    cam_0_1_R = np.array([[9.998806194689772e-01,     7.303636302427013e-03 ,   -1.361630298929278e-02],
                                     [-7.653466954692024e-03,     9.996373237376646e-01  ,  -2.581947780597325e-02],
                                     [1.342278860400440e-02  ,   2.592060738797566e-02 ,    9.995738846422163e-01]])
    cam_0_1_t = np.array([[6.490402887577009e+01], [-4.946543212569138e+00], [-7.130161883182569e+00]])
    # rectify_w = 640
    # rectify_h = 480
    rectify_h = 480
    rectify_w = 640
    R_1_0 = np.linalg.inv(cam_0_1_R)
    T_1_0 = -cam_0_1_t
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(cam1, dist1,
                                                                cam0, dist0,
                                                                (rectify_w, rectify_h),
                                                                R_1_0,
                                                                T_1_0,
                                                                flags=cv2.CALIB_ZERO_DISPARITY,
                                                                alpha=0)
    # Q[2,3] = 5.549697628637119e+02                                                    
    return Q,P1

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
 

def load_rect_disp_EF():
    show_FLAG = False
    [cam0_image, cam1_image] = loadEvFrame(show_FLAG = show_FLAG)
    # old_rectifyEF(cam1_image, cam0_image, show_FLAG = False)   # load is alreeady rectified✅
    rectifiedImgList,rectifiedParas  = rectifyEF(cam0_image, cam1_image, show_FLAG = show_FLAG)
    [cam0_image_rectified,cam1_image_rectified] = rectifiedImgList
    [R0, R1, P0, P1, Q] = rectifiedParas
    stepsize = 1
    [left_disparity,right_disparity] = ut.compute_leftright_disparity(cam0_image_rectified, 
                                                                                        cam1_image_rectified,
                                                                                        stepsize = stepsize,
                                                                                        show_FLAG = True)

    # left_disparity = ut.left_disparityPy(cam0_image_rectified, 
    #                                                             cam1_image_rectified, 
    #                                                             template_size = (60,60), 
    #                                                             show_FLAG = True, 
    #                                                             verbosity = False, 
    #                                                             save_path = None,
    #                                                             stepsize = 1)




    # right_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_disparitymap.png'
    # right_disparity_EF = cv2.imread(right_disparity_EF_file,0)

def load_rect_warped_EF_rl():   # 对的
    print('load_rect_warped_EF')
    show_FLAG = True
    [cam0_image, cam1_image] = loadEvFrame(show_FLAG = show_FLAG)
    # old_rectifyEF(cam1_image, cam0_image, show_FLAG = False)   # load is alreeady rectified✅
    rectifiedImgList,rectifiedParas  = rectifyEF(cam0_image, cam1_image, show_FLAG = show_FLAG)
    [cam0_image_rectified,cam1_image_rectified] = rectifiedImgList
    [R0, R1, P0, P1, Q] = rectifiedParas

    # left_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/kitti/origin_left_disparity_10.png'
    left_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_left_disparity.png'
    left_disparity_EF = cv2.imread(left_disparity_EF_file,0)

    right_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_right_disparity.png'
    right_disparity_EF = cv2.imread(right_disparity_EF_file,0)   
    if left_disparity_EF is None :
        print('Error: Could not load image')
        quit()
    if show_FLAG == True:
        left_disparity_EF_norm = ut.img_normalization(left_disparity_EF)
        cv2.imshow('left_disparity_EF_norm',left_disparity_EF_norm)
        cv2.imshow('right_disparity_EF',right_disparity_EF)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            # warp_save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
            # # cv2.imwrite(warp_save_path + '/' + 'warped_kitti' + '.png',warped_img)
            cv2.destroyAllWindows() 
    # TODO youwenti
    Q_,P1_ = Q_prime()
    warped_img = ut.warp(right_disparity_EF,Q_,P1_,cam1_image_rectified) 

    warp_save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
    cv2.imwrite(warp_save_path + '/' + 'warped_EF_rl' + '.png',warped_img)
 
def load_rect_warped_EF_lr(): 
    print('load_rect_warped_EF')
    show_FLAG = True
    [cam0_image, cam1_image] = loadEvFrame(show_FLAG = show_FLAG)
    # old_rectifyEF(cam1_image, cam0_image, show_FLAG = False)   # load is alreeady rectified✅
    rectifiedImgList,rectifiedParas  = rectifyEF(cam0_image, cam1_image, show_FLAG = show_FLAG)
    [cam0_image_rectified,cam1_image_rectified] = rectifiedImgList
    [R0, R1, P0, P1, Q] = rectifiedParas

    # left_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/kitti/origin_left_disparity_10.png'
    left_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_left_disparity.png'
    left_disparity_EF = cv2.imread(left_disparity_EF_file,0)

    right_disparity_EF_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_right_disparity.png'
    right_disparity_EF = cv2.imread(right_disparity_EF_file,0)   
    if left_disparity_EF is None :
        print('Error: Could not load image')
        quit()
    show_FLAG = True
    if show_FLAG == True:
        left_disparity_EF_norm = ut.img_normalization(left_disparity_EF)
        cv2.imshow('left_disparity_EF',left_disparity_EF)
        cv2.imshow('right_disparity_EF',right_disparity_EF)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            # warp_save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
            # # cv2.imwrite(warp_save_path + '/' + 'warped_kitti' + '.png',warped_img)
            cv2.destroyAllWindows() 
    # TODO youwenti
    Q_,P1_ = Q_prime()
    warped_img = ut.warp(left_disparity_EF,Q_,P1_,cam0_image_rectified) 
    warp_save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
    cv2.imwrite(warp_save_path + '/' + 'warped_EF_lr' + '.png',warped_img)


if __name__ == '__main__':
    # load_rect_lrDisp_kitti()  # ✅
    # use ut.warp in load_rect_warped_kitti to cheke the code
    # load_rect_warped_kirtti() # ✅
    # use ev frame   from left to right
    # load_rect_disp_EF()  # ✅
    # TODO create load_rect_warped_EF
    load_rect_warped_EF_rl() # 对的
    load_rect_warped_EF_lr()







