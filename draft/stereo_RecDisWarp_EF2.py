import numpy as np
import cv2
import src.HyStereo.utilities as ut
# import matplotlib.pyplot as plt
# import os

'''
整理代码
⭕️ Event Frame System
'''

'''
✅  2D-3D-2D point 
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

def loadEvFrame(show_FLAG = False):
    print('load Ev Frame')
    # ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105ev_rectified_10.png'
    ev_imgFile = '/Users/cainan/Desktop/Project/data/processed/cutoff_10/event105.png'
    # frame_imgFile = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105fr_rectified.png'
    frame_imgFile = '/Users/cainan/Desktop/Project/data/01_simple/png/105.png'

    leftImageFile = frame_imgFile  # left!!
    rightImageFile = ev_imgFile     # right!!
    [frame_image, event_image] = ut.loadStereoImage(leftImageFile, rightImageFile, show_FLAG = show_FLAG)
    return [frame_image, event_image] 

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

def rectify_right2left():
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
    rectify_w = 640
    rectify_h = 480
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(cam1, dist1,
                                                                    cam0, dist0,
                                                                    (rectify_w, rectify_h),
                                                                    cam_0_1_R,
                                                                    cam_0_1_t,
                                                                    flags=cv2.CALIB_ZERO_DISPARITY,)

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

def load_rect_warped_EF():   # 对的
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
    if left_disparity_EF is None or right_disparity_EF is None:
        print('Error: Could not load image')
        quit()
    if show_FLAG == True:
        left_disparity_EF_norm = ut.img_normalization(left_disparity_EF)
        cv2.imshow('left_disparity_EF_norm',left_disparity_EF_norm)
        cv2.imshow('right_disparity_EF',right_disparity_EF)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 
    Q_,P1_ = Q_prime()
    warped_img_rl = ut.warp(right_disparity_EF,Q_,P1_,cam1_image_rectified) 
    warped_img_lr = ut.warp(left_disparity_EF,Q_,P1_,cam0_image_rectified) 
    warp_save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
    cv2.imwrite(warp_save_path + '/' + 'warped_EF_rl' + '.png',warped_img_rl)

def rectify(frame_image, event_image, show_FLAG = False):
    # print('rectify EF')
    # cam1 EV  cam2 FR -----from stereo_params.txt  
    # ✅left image: frame;    right image: event   ---- for the images
    # -->FR: cam2&left     EV: cam1&right
    cam1 = np.array([[5.549697628637119e+02, 0, 3.195838836743635e+02],
                                                             [0., 5.558964313825462e+02, 2.172033768792723e+02],
                                                              [0., 0., 1.]])
    cam2 = np.array([[4.054135566221585e+02, 0, 3.142290421594836e+02],
                                                               [0., 4.049340581100494e+02,  2.445772553059500e+02],
                                                               [0., 0., 1.]])
    dist1 = np.array([[-1.156664910088120e-01, 1.946241433904505e-01,3.025107052357226e-03, 4.913881837816321e-04, 1.768629600684745e-02]])
    dist2 = np.array([[-2.405663351829244e-01, 1.804700878515662e-01, 3.107546803597012e-03, -4.817144966179097e-04, -9.362895292153335e-02]])
    cam_1_2_R = np.array([[9.998806194689772e-01,     7.303636302427013e-03 ,   -1.361630298929278e-02],
                                     [-7.653466954692024e-03,     9.996373237376646e-01  ,  -2.581947780597325e-02],
                                     [1.342278860400440e-02  ,   2.592060738797566e-02 ,    9.995738846422163e-01]])
    cam_1_2_t = np.array([[6.490402887577009e+01], [-4.946543212569138e+00], [-7.130161883182569e+00]])
    # rectify_w = 640
    # rectify_h = 480
    rectify_h, rectify_w = frame_image.shape
    
    # TODO want to rectify from image left to image right -- let left camera(cam2) coordinate be the reference
    # usually the cam1 is the reference. If we want to change the reference camera coordinate systems, the only thing we need to change is R&T
    # the output is related to rotation matrix and translation matrix
    # the order of parameter in input doesn't matter
    R_2to1 = np.linalg.inv(cam_1_2_R)
    T_2to1 = -cam_1_2_t
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam1, dist1,
                                                                cam2, dist2,
                                                                (rectify_w, rectify_h),
                                                                R_2to1,  # the rotation matrix and translation matrix must use the image order
                                                                T_2to1,
                                                                flags=cv2.CALIB_ZERO_DISPARITY,
                                                                alpha=0)


    # -->FR: cam2&left     EV: cam1&right
    if frame_image is not None: 
        lMapX, lMapY = cv2.initUndistortRectifyMap(cam2, dist2, R2, P2, (rectify_w, rectify_h), cv2.CV_32FC1)
        rMapX, rMapY = cv2.initUndistortRectifyMap(cam1, dist1, R1, P1, (rectify_w, rectify_h), cv2.CV_32FC1)
        frame_image_rectified = cv2.remap(frame_image, lMapX, lMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        event_image_rectified = cv2.remap(event_image, rMapX, rMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        if show_FLAG == True:
            frame_image_rectified_show = frame_image_rectified.copy()
            event_image_rectified_show = event_image_rectified.copy()
            list = [100,159,200,250,300,400]
            for height in list:
                cv2.line(frame_image_rectified_show, (0, height), (frame_image_rectified_show.shape[1], height), (255, 255, 255))
                cv2.line(event_image_rectified_show, (0, height), (event_image_rectified_show.shape[1], height), (255, 255, 255))
            combine = cv2.hconcat([frame_image_rectified_show,event_image_rectified_show])
            cv2.imshow('frame_image_rectified_show,event_image_rectified_show]',combine)
            k = cv2.waitKey(0)
            if k == 27:         # ESC
                cv2.destroyAllWindows() 

    return [frame_image_rectified,event_image_rectified],[R1, R2, P1, P2, Q]

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
        cv2.imshow('left_disparity_EF_norm,right_disparity_EF_norm',combine)
        # k = cv2.waitKey(0)
        # if k == 27:         # ESC
        #     cv2.destroyAllWindows() 
    return [left_disparity_EF,right_disparity_EF]

def Load_RecInvRT_Warped_EF():
    show_FLAG = True
    save_FLAG = True

    # LOAD
    [frame_image, event_image] = loadEvFrame(show_FLAG = show_FLAG)  # left-frame  right-event

    # Rectify_invRT
    [frame_image_rectified,event_image_rectified],[R1, R2, P1, P2, Q] = rectify(frame_image, event_image, show_FLAG = show_FLAG)
    # P1 to left    P2 to right
    print('Q',Q,'P2',P2)
    # Warped_EF
    [left_disparity_EF,right_disparity_EF] = load_lrDisp_EF(show_FLAG = show_FLAG)
    # 要把右边作为first 用原本的 q 和p2  ❎
    warped_EF_r2l = ut.warp(right_disparity_EF,Q,P2,event_image_rectified) 
    # 把左边作为first 用inv q  和p2    ✅
    warped_EF_l2r = ut.warp(left_disparity_EF,Q,P2,frame_image_rectified) 
    if show_FLAG != False:
        # warped_EF_r2l_norm = ut.img_normalization(warped_EF_r2l)
        # warped_EF_l2r_norm = ut.img_normalization(warped_EF_l2r)
        warped_EF_r2l_show = warped_EF_r2l.copy()
        warped_EF_l2r_show = warped_EF_l2r.copy()
        list = [100,159,200,250,300,400]
        for height in list:
            cv2.line(warped_EF_l2r_show, (0, height), (warped_EF_l2r_show.shape[1], height), (255, 255, 255))
            cv2.line(warped_EF_r2l_show, (0, height), (warped_EF_r2l_show.shape[1], height), (255, 255, 255))
        combine = cv2.vconcat([warped_EF_r2l_show,warped_EF_l2r_show])
        cv2.imshow('warped_EF_r2l,warped_EF_l2r line',combine)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 
    if save_FLAG != False:
        warp_save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
        cv2.imwrite(warp_save_path + '/' + 'warped_EF_r2l' + '.png',warped_EF_r2l)
        cv2.imwrite(warp_save_path + '/' + 'warped_EF_l2r' + '.png',warped_EF_l2r)


if __name__ == '__main__':
    # use ev frame   from left to right
    # load_rect_disp_EF()  # ✅

    # warp 
    # load_rect_warped_EF() # 对的
    # load_rect_warped_EF()

    # TODO use all inverse parameters to rectify the stereo images 
    # Load_RecInvRT_Warped_EF()




    show_FLAG = True
    # save_FLAG = True

    # LOAD
    [frame_image, event_image] = loadEvFrame(show_FLAG = show_FLAG)  # left-frame  right-event

    # Rectify_invRT
    [frame_image_rectified,event_image_rectified],[R1, R2, P1, P2, Q] = rectify_invRT(frame_image, event_image, show_FLAG = show_FLAG)






    # TODO check if disparity is exist. If not generate it.







