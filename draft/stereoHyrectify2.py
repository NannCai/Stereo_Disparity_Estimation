import numpy as np
import cv2
# reference code from ESL
def rectifyImage(cam0, dist0, cam1, dist1, cam_0_1_R, cam_0_1_t, cam0_image=None, cam1_image=None):
    rectify_w = 960
    rectify_h = 640
    R1, R0, P1, P0, Q, _, _ = cv2.stereoRectify(cam1, dist1,
                                                                    cam0, dist0,
                                                                    (rectify_w, rectify_h),
                                                                    cam_0_1_R,
                                                                    cam_0_1_t,
                                                                    flags=cv2.CALIB_ZERO_DISPARITY,
                                                                    alpha=1)
    
    # not the same
    # R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(cam0, dist0,
    #                                                                 cam1, dist1,
    #                                                                 (rectify_w, rectify_h),
    #                                                                 cam_0_1_R,
    #                                                                 cam_0_1_t,
    #                                                                 flags=cv2.CALIB_ZERO_DISPARITY,
    #                                                                 alpha=-1)
    # cv2.imshow("cam0_image", cam0_image)
    # cv2.imshow("cam1_image", cam1_image)
    # cv2.waitKey(0)
    cam0MapX, cam0MapY = cv2.initUndistortRectifyMap(cam0, dist0, R0, P0, (rectify_w, rectify_h), cv2.CV_32FC1)
    cam1MapX, cam1MapY = cv2.initUndistortRectifyMap(cam1, dist1, R1, P1, (rectify_w, rectify_h), cv2.CV_32FC1)
    if cam0_image is not None: # flag None指的是没有输入图片 只输出rectify的map
        cam0_image_rectified = cv2.remap(cam0_image, cam0MapX, cam0MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        cam1_image_rectified = cv2.remap(cam1_image, cam1MapX, cam1MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    else: # 没有对image做rectify
        cam0_image_rectified = None
        cam1_image_rectified = None

    return R0, R1, P0, P1, cam0_image_rectified, cam1_image_rectified, Q

if __name__ == '__main__':
    # cam0, dist0, cam1, dist1, cam_0_1_R, cam_0_1_t, cam0_image=None, cam1_image=None
    cam0 = np.array([[5.549697628637119e+02, 0, 3.195838836743635e+02],   # EV 
                                                             [0., 5.558964313825462e+02, 2.172033768792723e+02],
                                                              [0., 0., 1.]])
    cam1 = np.array([[4.054135566221585e+02, 0, 3.142290421594836e+02],   # FR
                                                               [0., 4.049340581100494e+02,  2.445772553059500e+02],
                                                               [0., 0., 1.]])
    # EV 
    dist0 = np.array([[-1.156664910088120e-01, 1.946241433904505e-01,3.025107052357226e-03, 4.913881837816321e-04, 1.768629600684745e-02]])
    # FR
    dist1 = np.array([[-2.405663351829244e-01, 1.804700878515662e-01, 3.107546803597012e-03, -4.817144966179097e-04, -9.362895292153335e-02]])
    cam_0_1_R = np.array([[9.998806194689772e-01,     7.303636302427013e-03 ,   -1.361630298929278e-02],
                                     [-7.653466954692024e-03,     9.996373237376646e-01  ,  -2.581947780597325e-02],
                                     [1.342278860400440e-02  ,   2.592060738797566e-02 ,    9.995738846422163e-01]])
    cam_0_1_t = np.array([[6.490402887577009e+01], [-4.946543212569138e+00], [-7.130161883182569e+00]])
    rectify_w = 640
    rectify_h = 480

    '''
    LOAD DATA
    '''
    # load event preprocessed img  cam0---evnet 
    # ev_img_path = '/Users/cainan/Desktop/Project/data/correlation_test'
    # ev_img_path = '/Users/cainan/Desktop/Project/data/processed/pair_binary'
    ev_img_path = '/Users/cainan/Desktop/Project/data/processed/pair_gray'
    num = 105
    ev_img_filename = 'event_recon'+str(num)+'.png'
    ev_file = ev_img_path + '/' + ev_img_filename
    # ev_file = '/Users/cainan/Desktop/Project/data/processed/canny/cannyevent_canny_open105.png'
    ev_file = '/Users/cainan/Desktop/Project/data/processed/cutoff_10/event105.png'

    ev_img = cv2.imread(ev_file, 0)  # shape (480,640)
    cam0_image = ev_img

    # load frame preprocessed img  cam1----frame 
    prepro_frame_img_path = ev_img_path
    prepro_frame_img_filename = 'frame_gray'+str(num)+'.png'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_f_file = '/Users/cainan/Desktop/Project/data/01_simple/png/105.png'
    pre_frame_img = cv2.imread(pre_f_file, 0)  # åshape (1536,2048)  3.2times of 480
    resize_frame_img = cv2.resize(pre_frame_img, (rectify_w,rectify_h), interpolation=cv2.INTER_AREA)   # INTER_AREA
    cam1_image = resize_frame_img
    cv2.imshow('ev_img',ev_img)

    list = [100,150,200,250,300,400]
    for height in list:
        cv2.line(cam0_image, (0, height), (cam1_image.shape[1], height), (255, 255, 255))
        cv2.line(cam1_image, (0, height), (cam1_image.shape[1], height), (255, 255, 255))

    cv2.imshow("cam0_image", cam0_image)
    cv2.imshow("cam1_image", cam1_image)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        save_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
        # cv2.imwrite(save_path + '/' + 'cam0_image' + '.png',cam0_image)
        # cv2.imwrite(save_path + '/' + 'cam1_image' + '.png',cam1_image)
        cv2.destroyAllWindows()

    '''
    alpha = 0
    在立体校正阶段需要设置alpha = 0才能完成对图像的裁剪,否则会有黑边。
    '''
    R1, R0, P1, P0, Q, _, _ = cv2.stereoRectify(cam1, dist1,
                                                                    cam0, dist0,
                                                                    (rectify_w, rectify_h),
                                                                    cam_0_1_R,
                                                                    cam_0_1_t,
                                                                    flags=cv2.CALIB_ZERO_DISPARITY,
                                                                    alpha=0)

    cam0MapX, cam0MapY = cv2.initUndistortRectifyMap(cam0, dist0, R0, P0, (rectify_w, rectify_h), cv2.CV_32FC1)
    cam1MapX, cam1MapY = cv2.initUndistortRectifyMap(cam1, dist1, R1, P1, (rectify_w, rectify_h), cv2.CV_32FC1)
    if cam0_image is not None: 
        cam0_image_rectified = cv2.remap(cam0_image, cam0MapX, cam0MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        cam1_image_rectified = cv2.remap(cam1_image, cam1MapX, cam1MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    else: 
        cam0_image_rectified = None
        cam1_image_rectified = None


    # cam1_image_rectified = cv2.Canny(cam1_image_rectified, 90, 200)
    # cam0_image_rectified = cv2.Canny(cam0_image_rectified, 100, 200)
    cam0_image_rectified_show = cam0_image_rectified.copy()
    cam1_image_rectified_show = cam1_image_rectified.copy()
    list = [100,159,200,250,300,400]
    for height in list:
        cv2.line(cam0_image_rectified_show, (0, height), (cam1_image.shape[1], height), (255, 255, 255))
        cv2.line(cam1_image_rectified_show, (0, height), (cam1_image.shape[1], height), (255, 255, 255))

    save_flag = 'origin' + str(num)
    cv2.imshow(save_flag + 'ev_rectified_', cam0_image_rectified_show)
    cv2.imshow(save_flag + 'fr_rectified_', cam1_image_rectified_show)
    k = cv2.waitKey(0)
    
    if k == 27:         # ESC
        save_path = '/Users/cainan/Desktop/Project/data/processed/origin_rectify'
        # cv2.imwrite(save_path + '/' + save_flag + 'ev_rectified_10' + '.png',cam0_image_rectified)
        # cv2.imwrite(save_path + '/' + save_flag + 'fr_rectified' + '.png',cam1_image_rectified)
        cv2.destroyAllWindows()




