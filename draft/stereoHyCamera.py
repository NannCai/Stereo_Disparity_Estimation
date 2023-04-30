import numpy as np
import cv2
 
# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参  camera1 done
        # FocalLength: [5.549697628637119e+02 5.558964313825462e+02]
        # PrincipalPoint: [3.195838836743635e+02 2.172033768792723e+02]
        self.cam_matrix_left = np.array([[5.549697628637119e+02, 0, 3.195838836743635e+02],
                                                             [0., 5.558964313825462e+02, 2.172033768792723e+02],
                                                              [0., 0., 1.]])
        # 右相机内参  camera2 done
        # FocalLength: [4.054135566221585e+02 4.049340581100494e+02]
        # PrincipalPoint: [3.142290421594836e+02 2.445772553059500e+02]
        self.cam_matrix_right = np.array([[4.054135566221585e+02, 0, 3.142290421594836e+02],
                                                               [0., 4.049340581100494e+02,  2.445772553059500e+02],
                                                               [0., 0., 1.]])
 
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]  # 畸变系数,K1、K2、K3为径向畸变Radial,P1、P2为切向畸变tangential
        # cam1 left                    
        # RadialDistortion: [-1.156664910088120e-01 1.946241433904505e-01 1.768629600684745e-02]   k1 k2 k3
        # TangentialDistortion: [3.025107052357226e-03 4.913881837816321e-04]  p1 p2
        self.distortion_l = np.array([[-1.156664910088120e-01, 1.946241433904505e-01,3.025107052357226e-03, 4.913881837816321e-04, 1.768629600684745e-02]])
        # cam2 right
        # RadialDistortion: [-2.405663351829244e-01 1.804700878515662e-01 -9.362895292153335e-02]
        # TangentialDistortion: [3.107546803597012e-03 -4.817144966179097e-04]
        self.distortion_r = np.array([[-2.405663351829244e-01, 1.804700878515662e-01, 3.107546803597012e-03, -4.817144966179097e-04, -9.362895292153335e-02]])
 
        # 旋转矩阵  stereoParams.RotationOfCamera2 done
        self.R = np.array([[9.998806194689772e-01,     7.303636302427013e-03 ,   -1.361630298929278e-02],
                                     [-7.653466954692024e-03,     9.996373237376646e-01  ,  -2.581947780597325e-02],
                                     [1.342278860400440e-02  ,   2.592060738797566e-02 ,    9.995738846422163e-01]])
#       9.998806194689772e-01     7.303636302427013e-03    -1.361630298929278e-02
    # -7.653466954692024e-03     9.996373237376646e-01    -2.581947780597325e-02
    #  1.342278860400440e-02     2.592060738797566e-02     9.995738846422163e-01


        # 平移矩阵  done
        #   6.490402887577009e+01    -4.946543212569138e+00    -7.130161883182569e+00
        self.T = np.array([[6.490402887577009e+01], [-4.946543212569138e+00], [-7.130161883182569e+00]])

        # size
        self.size = (640, 480)
 
        # # 主点列坐标的差
        # self.doffs = 0.0
 
        # # 指示上述内外参是否为经过立体校正后的结果
        # self.isRectified = False
 
    # def setMiddleBurryParams(self):
    #     self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
    #                                                         [0., 3997.684, 187.5],
    #                                                         [0., 0., 1.]])
    #     self.cam_matrix_right =  np.array([[3997.684, 0, 225.0],
    #                                                             [0., 3997.684, 187.5],
    #                                                             [0., 0., 1.]])
    #     self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
    #     self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
    #     self.R = np.identity(3, dtype= np.float64)
    #     self.T = np.array([[-193.001], [0.0], [0.0]])
    #     self.doffs = 131.111
    #     self.isRectified = True






if __name__ == '__main__':
# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
# left_camera_matrix = np.array([[516.5066236,-1.444673028,320.2950423],[0,516.5816117,270.7881873],[0.,0.,1.]])
    cam_matrix_left = np.array([[5.549697628637119e+02, 0, 3.195838836743635e+02],
                                                             [0., 5.558964313825462e+02, 2.172033768792723e+02],
                                                              [0., 0., 1.]])
# right_camera_matrix = np.array([[511.8428182,1.295112628,317.310253],[0,513.0748795,269.5885026],[0.,0.,1.]])
    cam_matrix_right = np.array([[4.054135566221585e+02, 0, 3.142290421594836e+02],
                                                               [0., 4.049340581100494e+02,  2.445772553059500e+02],
                                                               [0., 0., 1.]])
# # 畸变系数,K1、K2、K3为径向畸变Radial distortion,P1、P2为切向畸变 tangential distortion
# left_distortion = np.array([[-0.046645194,0.077595167, 0.012476819,-0.000711358,0]])
    distortion_l = np.array([[-1.156664910088120e-01, 1.946241433904505e-01,3.025107052357226e-03, 4.913881837816321e-04, 1.768629600684745e-02]])
# right_distortion = np.array([[-0.061588946,0.122384376,0.011081232,-0.000750439,0]])
    distortion_r = np.array([[-2.405663351829244e-01, 1.804700878515662e-01, 3.107546803597012e-03, -4.817144966179097e-04, -9.362895292153335e-02]])

# # 旋转矩阵
# R = np.array([[0.999911333,-0.004351508,0.012585312],
#               [0.004184066,0.999902792,0.013300386],
#               [-0.012641965,-0.013246549,0.999832341]])
    R = np.array([[9.998806194689772e-01,     7.303636302427013e-03 ,   -1.361630298929278e-02],
                                     [-7.653466954692024e-03,     9.996373237376646e-01  ,  -2.581947780597325e-02],
                                     [1.342278860400440e-02  ,   2.592060738797566e-02 ,    9.995738846422163e-01]])
# # 平移矩阵
# T = np.array([-120.3559901,-0.188953775,-0.662073075])
    T = np.array([[6.490402887577009e+01], [-4.946543212569138e+00], [-7.130161883182569e+00]])
# size = (640, 480)
    size = (640, 480)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cam_matrix_left, distortion_l,
                                                                    cam_matrix_right, distortion_r, size, R,
                                                                    T,alpha=0.9)

    # 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
    left_map1, left_map2 = cv2.initUndistortRectifyMap(cam_matrix_left, distortion_l, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cam_matrix_right, distortion_r, R2, P2, size, cv2.CV_16SC2)
    print(Q)

    '''
    LOAD DATA
    '''
    # load event preprocessed img  cam0---evnet 
    ev_img_path = '/Users/cainan/Desktop/Project/data/processed'
    ev_img_filename = 'event_binary0.png'
    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640)
    imgL = ev_img

    # load frame preprocessed img  cam1----frame 
    prepro_frame_img_path = ev_img_path
    prepro_frame_img_filename = 'frame_binary0.png'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # # shape (1536,2048)  3.2times of 480
    resize_frame_img = cv2.resize(pre_frame_img, (640,480))
    imgR = resize_frame_img


    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    list = [100,150,200,250,300,400]
    for height in list:
        cv2.line(imgL_rectified, (0, height), (imgL_rectified.shape[1], height), (255, 255, 255))
        cv2.line(imgR_rectified, (0, height), (imgR_rectified.shape[1], height), (255, 255, 255))
 
 
    cv2.imshow("imgL_rectified", imgL_rectified)
    cv2.imshow("imgR_rectified", imgR_rectified)

    k = cv2.waitKey(0)
    if k == 27:         # ESC
        save_path = '/Users/cainan/Desktop/Project/data/processed/rectify'
        cv2.imwrite(save_path + '/' + 'imgL_rectified' + 'paper2' + '.png',imgL_rectified)
        cv2.imwrite(save_path + '/' + 'imgR_rectified' + 'paper2' + '.png',imgR_rectified)
        cv2.destroyAllWindows()