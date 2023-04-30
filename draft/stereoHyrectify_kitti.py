import numpy as np
import cv2

if __name__ == '__main__':
    # cam0, dist0, cam1, dist1, cam_0_1_R, cam_0_1_t, cam0_image=None, cam1_image=None
    # S_00: 1.392000e+03 5.120000e+02 ✅
    # K_00: 9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00
    # D_00: -3.728755e-01 2.037299e-01 2.219027e-03 1.383707e-03 -7.233722e-02
    # R_00: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00
    # T_00: 2.573699e-16 -1.059758e-16 1.614870e-16
    # S_rect_00: 1.242000e+03 3.750000e+02
    # R_rect_00: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
    # P_rect_00: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
    # S_01: 1.392000e+03 5.120000e+02
    # K_01: 9.895267e+02 0.000000e+00 7.020000e+02 0.000000e+00 9.878386e+02 2.455590e+02 0.000000e+00 0.000000e+00 1.000000e+00
    # D_01: -3.644661e-01 1.790019e-01 1.148107e-03 -6.298563e-04 -5.314062e-02
    # R_01: 9.993513e-01 1.860866e-02 -3.083487e-02 -1.887662e-02 9.997863e-01 -8.421873e-03 3.067156e-02 8.998467e-03 9.994890e-01
    # T_01: -5.370000e-01 4.822061e-03 -1.252488e-02
    # S_rect_01: 1.242000e+03 3.750000e+02
    # R_rect_01: 9.996878e-01 -8.976826e-03 2.331651e-02 8.876121e-03 9.999508e-01 4.418952e-03 -2.335503e-02 -4.210612e-03 9.997184e-01
    # P_rect_01: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00


    # K_00: 9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00    
    cam0 = np.array([[9.842439e+02, 0.000000e+00, 6.900000e+02],
                                                             [0.000000e+00, 9.808141e+02, 2.331966e+02],
                                                              [0.000000e+00, 0.000000e+00, 1.000000e+00 ]])
    # K_01: 9.895267e+02 0.000000e+00 7.020000e+02 0.000000e+00 9.878386e+02 2.455590e+02 0.000000e+00 0.000000e+00 1.000000e+00
    cam1 = np.array([[9.895267e+02, 0.000000e+00, 7.020000e+02],
                                                               [0.000000e+00, 9.878386e+02, 2.455590e+02],
                                                               [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    # D_00: -3.728755e-01 2.037299e-01 2.219027e-03 1.383707e-03 -7.233722e-02
    dist0 = np.array([[-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02]])
    # D_01: -3.644661e-01 1.790019e-01 1.148107e-03 -6.298563e-04 -5.314062e-02
    dist1 = np.array([[-3.644661e-01, 1.790019e-01, 1.148107e-03, -6.298563e-04, -5.314062e-02]])
    # R_01: 9.993513e-01 1.860866e-02 -3.083487e-02 -1.887662e-02 9.997863e-01 -8.421873e-03 3.067156e-02 8.998467e-03 9.994890e-01
    cam_0_1_R = np.array([[9.993513e-01, 1.860866e-02, -3.083487e-02],
                                     [-1.887662e-02, 9.997863e-01, -8.421873e-03],
                                     [3.067156e-02, 8.998467e-03, 9.994890e-01]])
    # T_01: -5.370000e-01 4.822061e-03 -1.252488e-02                                 
    cam_0_1_t = np.array([[-5.370000e-01], [4.822061e-03], [-1.252488e-02]])
    # rectify_w = 1.392000e+03 
    # rectify_h = 5.120000e+02
    rectify_w = 1392
    rectify_h = 512


    '''
    LOAD DATA
    '''
    # load event preprocessed img  cam0---evnet 
    # ev_img_path = '/Users/cainan/Desktop/Project/data/processed'
    ev_img_path = '/Users/cainan/Downloads/2011_09_26 2/2011_09_26_drive_0001_extract/image_00/data'
    ev_img_filename = '0000000000.png'
    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640)
    cam0_image = ev_img

    # load frame preprocessed img  cam1----frame 
    prepro_frame_img_path = '/Users/cainan/Downloads/2011_09_26 2/2011_09_26_drive_0001_extract/image_01/data'
    prepro_frame_img_filename = ev_img_filename
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # # shape (1536,2048)  3.2times of 480
    # resize_frame_img = cv2.resize(pre_frame_img, (640,480))
    cam1_image = pre_frame_img

    # list = [100,150,200,250,300,400,455]
    # for height in list:
    #     cv2.line(cam0_image, (0, height), (cam1_image.shape[1], height), (255, 255, 255))
    #     cv2.line(cam1_image, (0, height), (cam1_image.shape[1], height), (255, 255, 255))

    cv2.imshow("cam0_image", cam0_image)
    cv2.imshow("cam1_image", cam1_image)
    k = cv2.waitKey(0)
    if k == 27:         # ESC
        # save_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
        # cv2.imwrite(save_path + '/' + 'cam0_image_rectified_kitti' + '.png',cam0_image_rectified)
        # cv2.imwrite(save_path + '/' + 'cam1_image_rectified_kitti' + '.png',cam1_image_rectified)
        cv2.destroyAllWindows()

    # R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(cam0, dist0,
    #                                                                 cam1, dist1,
    #                                                                 (rectify_w, rectify_h),
    #                                                                 cam_0_1_R,
    #                                                                 cam_0_1_t,
    #                                                                 flags=cv2.CALIB_ZERO_DISPARITY,
    #                                                                 alpha=0)
    # maybe flags fault? paras fault   config = 'change_cam_oder'  --is wrong!!
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

    list = [100,150,200,250,300,400,455]
    for height in list:
        cv2.line(cam0_image_rectified, (0, height), (cam1_image.shape[1], height), (255, 255, 255))
        cv2.line(cam1_image_rectified, (0, height), (cam1_image.shape[1], height), (255, 255, 255))


    cv2.imshow("cam0_image_rectified_kitti", cam0_image_rectified)
    cv2.imshow("cam1_image_rectified_kitti", cam1_image_rectified)
    k = cv2.waitKey(0)
    # config = 'change_cam_oder'
    config = ''
    # config = ''
    if k == 27:         # ESC
        save_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
        cv2.imwrite(save_path + '/' + config + 'cam0_image_rectified_kitti' + '.png',cam0_image_rectified)
        cv2.imwrite(save_path + '/' + config + 'cam1_image_rectified_kitti' + '.png',cam1_image_rectified)
        cv2.destroyAllWindows()
