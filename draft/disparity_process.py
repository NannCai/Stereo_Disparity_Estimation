import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def load_data(show_FLAG = False):
    # load event preprocessed img
    ev_file = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105ev_rectified_10.png'
    # ev_file = ''
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640

    # load frame preprocessed img
    pre_f_file = '/Users/cainan/Desktop/Project/data/processed/origin_rectify/origin105fr_rectified.png'
    # pre_f_file = ''
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

def stereoRectifyParameter(cam0_image=None, cam1_image=None):
    cam0 = np.array([[5.549697628637119e+02, 0, 3.195838836743635e+02],
                                                             [0., 5.558964313825462e+02, 2.172033768792723e+02],
                                                              [0., 0., 1.]])
    cam1 = np.array([[4.054135566221585e+02, 0, 3.142290421594836e+02],
                                                               [0., 4.049340581100494e+02,  2.445772553059500e+02],
                                                               [0., 0., 1.]])
    dist0 = np.array([[-1.156664910088120e-01, 1.946241433904505e-01,3.025107052357226e-03, 4.913881837816321e-04, 1.768629600684745e-02]])
    dist1 = np.array([[-2.405663351829244e-01, 1.804700878515662e-01, 3.107546803597012e-03, -4.817144966179097e-04, -9.362895292153335e-02]])
    cam_0_1_R = np.array([[9.998806194689772e-01,     7.303636302427013e-03 ,   -1.361630298929278e-02],
                                     [-7.653466954692024e-03,     9.996373237376646e-01  ,  -2.581947780597325e-02],
                                     [1.342278860400440e-02  ,   2.592060738797566e-02 ,    9.995738846422163e-01]])
    cam_0_1_t = np.array([[6.490402887577009e+01], [-4.946543212569138e+00], [-7.130161883182569e+00]])
    rectify_w = 640
    rectify_h = 480

    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(cam0, dist0,
                                                                    cam1, dist1,
                                                                    (rectify_w, rectify_h),
                                                                    cam_0_1_R,
                                                                    cam_0_1_t,
                                                                    flags=cv2.CALIB_ZERO_DISPARITY,
                                                                    alpha=0)

    if cam0_image is not None: 
        cam0MapX, cam0MapY = cv2.initUndistortRectifyMap(cam0, dist0, R0, P0, (rectify_w, rectify_h), cv2.CV_32FC1)
        cam1MapX, cam1MapY = cv2.initUndistortRectifyMap(cam1, dist1, R1, P1, (rectify_w, rectify_h), cv2.CV_32FC1)
        cam0_image_rectified = cv2.remap(cam0_image, cam0MapX, cam0MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        cam1_image_rectified = cv2.remap(cam1_image, cam1MapX, cam1MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    # else: 
    #     cam0_image_rectified = None
    #     cam1_image_rectified = None

    '''
    R1	Output 3x3 rectification transform (rotation matrix) for the first camera. 
        This matrix brings points_3d given in the unrectified first camera's coordinate system to points_3d in the rectified first camera's coordinate system. 
        In more technical terms, it performs a change of basis from the ...unrectified first camera's coordinate system to the rectified first camera's coordinate system...
    R2	Output 3x3 rectification transform (rotation matrix) for the second camera. 
        This matrix brings points_3d given in the unrectified second camera's coordinate system to points_3d in the rectified second camera's coordinate system. 
        In more technical terms, it performs a change of basis from the ...unrectified second camera's coordinate system to the rectified second camera's coordinate system...
    P1	Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera, 
        i.e. it projects points_3d given in the ...rectified first camera coordinate system into the rectified first camera's image...
    P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, 
        i.e. it projects points_3d given in the ...rectified first camera coordinate system into the rectified second camera's image...
    Q	Output 4x4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
    '''

    return R1, R0, P1, P0, Q

def stereoRectifyParameter_kitti(cam0_image=None, cam1_image=None):
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

    # R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(cam0, dist0,
    #                                                                 cam1, dist1,
    #                                                                 (rectify_w, rectify_h),
    #                                                                 cam_0_1_R,
    #                                                                 cam_0_1_t,
    #                                                                 flags=cv2.CALIB_ZERO_DISPARITY,
    #                                                                 alpha=0)
    R1, R0, P1, P0, Q, _, _ = cv2.stereoRectify(cam1, dist1,
                                                                    cam0, dist0,
                                                                    (rectify_w, rectify_h),
                                                                    cam_0_1_R,
                                                                    cam_0_1_t,
                                                                    flags=cv2.CALIB_ZERO_DISPARITY,
                                                                    alpha=0)

    '''
    R1	Output 3x3 rectification transform (rotation matrix) for the first camera. 
        This matrix brings points_3d given in the unrectified first camera's coordinate system to points_3d in the rectified first camera's coordinate system. 
        In more technical terms, it performs a change of basis from the ...unrectified first camera's coordinate system to the rectified first camera's coordinate system...
    R2	Output 3x3 rectification transform (rotation matrix) for the second camera. 
        This matrix brings points_3d given in the unrectified second camera's coordinate system to points_3d in the rectified second camera's coordinate system. 
        In more technical terms, it performs a change of basis from the ...unrectified second camera's coordinate system to the rectified second camera's coordinate system...
    P1	Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera, 
        i.e. it projects points_3d given in the ...rectified first camera coordinate system into the rectified first camera's image...
    P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, 
        i.e. it projects points_3d given in the ...rectified first camera coordinate system into the rectified second camera's image...
    Q	Output 4x4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
    '''

    return R1, R0, P1, P0, Q

# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points_3d):
    height, width = points_3d.shape[0:2]
 
    points_1 = points_3d[:, :, 0].reshape(height * width, 1)
    points_2 = points_3d[:, :, 1].reshape(height * width, 1)
    points_3 = points_3d[:, :, 2].reshape(height * width, 1)
 
    points_ = np.hstack((points_1, points_2, points_3))
 
    return points_

def getDepthMapWithQ(disparityMap : np.ndarray, Q : np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
 
    return depthMap.astype(np.float32)




if __name__ == '__main__':
    save_path = '/Users/cainan/Desktop/Project/data/processed/disparity'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # read disparity map and image left and right
    # use kitti first to run through the code

    # dis_file = '/Users/cainan/Desktop/Project/data/processed/disparity'
    dis_file = '/Users/cainan/Desktop/Project/data/processed/disparity/origin_disparitymap_tsukuba.png'
    dis_file = dis_file + '/' + 'origin_disparitymap.png'
    disparity_map = cv2.imread(dis_file, 0) 
    v_max = 80
    disparity_map[disparity_map>v_max]=0
    # disparity_map = np.zeros((480,640),np.float16)-disparity_map.astype(np.float16)
    # disp_f16 = disparity_map.astype(np.float16)

    pre_frame_img, ev_img = load_data(show_FLAG= False)
    if disparity_map is None :
        print('Error: Could not load disp image')
        quit()


    # 2D to 3D      reprojectImageTo3D
    '''
    Q	Output 4x4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
    '''
    R1, R0, P1, P0, Q = stereoRectifyParameter()

    points_3d = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues= True)   # shape (480,640,3)
    # points_f32 = points_3d.astype(np.float64)
    # fini = np.isfinite(points_3d)
    # points_3d = points_3d[np.isfinite(points_3d)]
    
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    depthMap = depthMap.astype(np.float32) # 把置零的给换成float


    # 3D to 2D   using P2
    '''
        P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, 
        i.e. it projects points_3d given in the ...rectified first camera coordinate system into the rectified second camera's image...
    '''

    pointsHomo = np.ones((480,640,4),np.float32)
    pointsHomo[:,:,:3] = points_3d
    # pointsHomo = np.stack([points_3d,np.ones((480,640),np.float32) ], axis=-1)

    for line in pointsHomo: # shape (640,3)
        # print(line)
        for point in line:
            # print(point)
            # print('np.isfinite(point)',np.isfinite(point))
            # print( np.array([True, True,True ]))
            # if np.array([True, True,True ]) == np.isfinite(point):  # the point we want
            if np.all(np.isfinite(point)[:] == True) and point[2] > 0:
                print(point)
            else:
                print('hole/false')


        















    # generate the depth map
    # ev_focal_x = 5.549697628637119e+02
    # # The baseline b = 65.44 mm, focal lengths f = 555 pixels
    # ev_b = 65.44
    # if pre_frame_img.shape != ev_img.shape:
    #     print('pre_frame_img.shape',pre_frame_img.shape,'ev_img.shape',ev_img.shape)
    # img_height,img_width = pre_frame_img.shape
    # depthMap = np.zeros((img_height,img_width), np.float64)
    # for x in range(img_width):
    #     for y in range(img_width):
    #         if ()
    #         depthMap[y,x] = ev_focal_x * ev_b / float(disparity_map[y,x])




		# for (int i = 0; i < height; i++)		
		# {			
		# 	for (int j = 0; j < width; j++)			
		# 	{				
		# 		int id = i * width + j;				
		# 		if (!dispData[id])				
		# 		{					
		# 			//防止0除					
		# 			//depthData[id] = 255;					
		# 			continue;				
		# 		}				
		# 		else					
		# 			depthData[id] = fx * baseline / float(dispData[id]);			
		# 	}	


    # cv2.imshow('inpainted_norm',inpainted_norm)
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     # cv2.imwrite(save_path + '/' + 'pyramid460_medianblur' + '.png',disp_norm)
    #     cv2.destroyAllWindows()   







'''
    float dep = pDepthV[i*imgW + j];
    if (dep < 0.001) continue;
    //反投影
    float X = (j - cx)*dep / fx;
    float Y = (i - cy)*dep / fy;
    float Z = dep;
    //平移
    float X2 = X + offsetX;
    float Y2 = Y;
    float Z2 = Z;
    //重投影
    float uf = (fx*X2 + cx* Z2) / dep;
    float vf = (fy*Y2 + cy * Z2) / dep;
'''