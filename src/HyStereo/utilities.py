import numpy as np
import cv2
'''
loadStereoImage
rectifyImage
left_disparityPy
rightDisparity
right_disparityPy
img_normalization
warp
'''
def img_show(img,img_name,show_FLAG):
    # print(str(img_name))
    if show_FLAG == True:
        if len(img) in range(1,4):
            for image,name in zip(img,img_name):
                cv2.imshow(name,image)

        else:
            cv2.imshow(img_name,img)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 

def parse_StereoImage(leftImageFile, rightImageFile, show_FLAG = False):
    leftImg = cv2.imread(leftImageFile,0)
    rightImg = cv2.imread(rightImageFile,0)

    if leftImg is None or rightImg is None:
        print('Error: Could not load image')
        quit()

    if leftImg.shape != rightImg.shape :
        leftImg = cv2.resize(leftImg,(rightImg.shape[1],rightImg.shape[0]), interpolation=cv2.INTER_AREA)   # INTER_AREA

    if show_FLAG == True:
        combine = cv2.vconcat([leftImg,rightImg])
        cv2.imshow('leftImg,rightImg',combine)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 
    return [leftImg,rightImg]

def rectifyImage(cam1, dist0, cam2, dist1, cam_0_1_R, cam_0_1_t, cam1_image, cam2_image, rectify_w , rectify_h, show_FLAG = False):
    # 1to2 so the R and T is from 1 to 2  !! the R must transpose  and the t is from ev to frame
    print('rectifing image...')
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam1, dist0,
                                                                    cam2, dist1,
                                                                    (rectify_w, rectify_h),
                                                                    cam_0_1_R,
                                                                    cam_0_1_t,
                                                                    # flags=cv2.CALIB_ZERO_DISPARITY,
                                                                    alpha=0)

    cam1MapX, cam1MapY = cv2.initUndistortRectifyMap(cam1, dist0, R1, P1, (rectify_w, rectify_h), cv2.CV_32FC1)
    cam2MapX, cam2MapY = cv2.initUndistortRectifyMap(cam2, dist1, R2, P2, (rectify_w, rectify_h), cv2.CV_32FC1)
    cam1_image_rectified = cv2.remap(cam1_image, cam1MapX, cam1MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    cam2_image_rectified = cv2.remap(cam2_image, cam2MapX, cam2MapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    if show_FLAG == True:
        cam1_image_rectified_show = cv2.cvtColor(cam1_image_rectified,cv2.COLOR_GRAY2BGR)
        cam2_image_rectified_show = cv2.cvtColor(cam2_image_rectified,cv2.COLOR_GRAY2BGR)

        list = [100,159,200,250,300,400]
        for height in list:
            cv2.line(cam1_image_rectified_show, (0, height), (cam2_image.shape[1], height), (0, 0, 255),thickness= 4)
            cv2.line(cam2_image_rectified_show, (0, height), (cam2_image.shape[1], height), (0, 0, 255),thickness= 4)
        combine = cv2.hconcat([cam1_image_rectified_show,cam2_image_rectified_show])
        # combine2 = cv2.vconcat([cam1_image_rectified_show,cam2_image_rectified_show])
        cv2.imshow('cam1_image_rectified_show,cam2_image_rectified_show',combine)
        save_path = '/Users/cainan/Desktop/Project/data/processed/rectify/with_line'
        cv2.imwrite(save_path+ '/' + 'cam1_image_rectified_show' + '.png',cam1_image_rectified_show)
        cv2.imwrite(save_path+ '/' + 'cam2_image_rectified_show' + '.png',cam2_image_rectified_show)
    # cv2.imwrite(save_path + '/' + 'sobel_x_frame' + '.png',frame_image_rectified)
        print(combine.shape)
        # cv2.imshow('combine2',combine2)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 

    print('rectify done')
    return [cam1_image_rectified,cam2_image_rectified],[R1, R2, P1, P2, Q]

# compute the disparity of left image 
# the left disparity have a blank on the left, the right disparity have a blank on the right
def left_disparityPy(leftImg, rightImg, template_size = (60,60), 
                     show_FLAG = False, verbosity = False, 
                     save_path = None, stepsize = 1):  
    print('compute left disparityPy')
    img_height,img_width = leftImg.shape
    disparity_map = np.zeros((img_height,img_width), np.float16)  
    template_width, template_height = template_size 
    mis_th = int(img_width/7)
    # for x in range(int(template_width * 0.5),int(img_width - template_width * 0.5),stepsize):
    #     for y in range(int(template_height * 0.5),int(img_height - template_height * 0.5),stepsize):
    for x in range(300,int(img_width - template_width * 0.5),stepsize):
        for y in range(200,int(img_height - template_height * 0.5),stepsize):
            template_c = (x,y)
            top = template_c[1] -int(template_height * 0.5)
            left = template_c[0] - int(template_width * 0.5)
            template = leftImg[top:top + template_height, left:left + template_width]   # top down left right    

            right_epi_img = rightImg[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(right_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            right_max_x = max_loc[0] + int(template_width * 0.5) 
            disp = template_c[0] - right_max_x
            if disp < 0 or disp > mis_th :
                disparity_map[y,x] = 0
                continue  

            template_height_small = int(template_height * 2 / 3) # smaller window size to compute finer disparity
            template_width_small = template_height_small
            top_small = template_c[1] -int(template_height_small * 0.5)
            left_small = template_c[0] - int(template_width_small * 0.5)
            template_small = leftImg[top_small:top_small + template_height_small, left_small:left_small + template_width_small]

            finer_range = int(template_height/6) + int(template_width_small * 0.5)  # TODO to avoid the range out of images
            right_epi_img_small = rightImg[top_small:top_small+template_height_small,right_max_x-finer_range:right_max_x+finer_range]  # top down left right
            epi_result_small = cv2.matchTemplate(right_epi_img_small, template_small, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val_small, _, max_loc_small = cv2.minMaxLoc(epi_result_small)
            right_max_x_small = right_max_x-finer_range + int(template_width_small * 0.5) + max_loc_small[0]    # left + half_template + max
            disp2 = template_c[0] - right_max_x_small

            if disp2 < 0 or disp2 > 100 :   # TODO need to fill in the hole
            # if max_val < 0.45 or disp < 0 or disp > 100 :
                disp2 = 0
            disparity_map[y,x] = disp2

            if verbosity == True:
                print('disp',disp,'disp2',disp2)
                # print("e_max_x",e_max_x,'e_max_x_small',e_max_x_small)
                # print(template_c)

            # if show_FLAG == True:
                #debug
                right_img_color = cv2.cvtColor(rightImg, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(right_img_color, (max_loc[0],top), (max_loc[0] + template_width,top + template_height), (0,255,255), 1)   # yellow
                # cv2.imshow('ev_img_find',ev_img_color)
                cv2.circle(right_img_color,([right_max_x_small,template_c[1]]),2,(0,255,255),-1)
                cv2.rectangle(right_img_color, (right_max_x_small-int(template_width_small * 0.5),top_small), (right_max_x_small + int(template_width_small * 0.5),top_small + template_height_small), (0,255,255), 1)   # yellow


                left_img_color = cv2.cvtColor(leftImg, cv2.COLOR_GRAY2BGR)
                cv2.circle(left_img_color,template_c,2,(0,0,255),-1)
                cv2.rectangle(left_img_color,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
                # cv2.imshow('left_img_color',left_img_color)
                cv2.rectangle(left_img_color,  (left_small, top_small), (left_small + template_width_small, top_small + template_height_small),(0,0,255), 1)  # red


                cv2.imshow('left_img_color,right_img_color',cv2.hconcat([left_img_color,right_img_color]))
                k = cv2.waitKey(0)
                if k == 27:         # ESC
                    cv2.destroyAllWindows() 

    print('compute done')
    if show_FLAG == True:
        left_disparity_norm = img_normalization(disparity_map)                                                        
        cv2.imshow("left_disparity_norm",left_disparity_norm)
        print('show left_disparity')
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            # cv2.imwrite(save_path + '/' + 'right_disparity_norm_kitti' + '.png',right_disparity_norm)
            # cv2.imwrite(save_path + '/' + 'left_disparity_norm_kitti' + '.png',left_disparity_norm)
            cv2.destroyAllWindows()           

    if save_path is not None:
        disparity_map_uint8 = np.uint8(disparity_map)
        cv2.imwrite(save_path + '/' + 'origin_left_disparity_part' + '.png',disparity_map_uint8) 

    return disparity_map    

def right_disparityPy(leftImg, rightImg, template_size = (60,60), 
                                                                show_FLAG = False, 
                                                                verbosity = False, 
                                                                save_path = None, 
                                                                stepsize = 1):  
    print('compute right disparity')
    img_height,img_width = leftImg.shape
    disparity_map = np.zeros((img_height,img_width), np.float16)  
    template_width, template_height = template_size 
    mis_th = int(img_width/7)
    # for x in range(int(template_width * 0.5),int(img_width - template_width * 0.5),stepsize):
    #     for y in range(int(template_height * 0.5),int(img_height - template_height * 0.5),stepsize):
    for x in range(600,int(img_width - template_width * 0.5),stepsize):
        for y in range(200,int(img_height - template_height * 0.5),stepsize):
            template_c = (x,y)
            # print(template_c)
            template_c = (x,y)
            top = template_c[1] -int(template_height * 0.5)
            left = template_c[0] - int(template_width * 0.5)
            template = rightImg[top:top + template_height, left:left + template_width]   # top down left right    

            left_epi_img = leftImg[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(left_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            left_max_x = max_loc[0] + int(template_width * 0.5) 
            disp = left_max_x - template_c[0] 
            if disp < 0 or disp > mis_th :
                disparity_map[y,x] = 0
                continue  

            template_height_small = int(template_height * 2 / 3) # smaller window size to compute finer disparity
            template_width_small = template_height_small
            top_small = template_c[1] -int(template_height_small * 0.5)
            left_small = template_c[0] - int(template_width_small * 0.5)
            template_small = rightImg[top_small:top_small + template_height_small, left_small:left_small + template_width_small]
            finer_range = int(template_height/6) + int(template_width_small * 0.5)  # TODO to avoid the range out of images
            left_epi_img_small = leftImg[top_small:top_small+template_height_small,left_max_x-finer_range:left_max_x+finer_range]  # top down left right
            epi_result_small = cv2.matchTemplate(left_epi_img_small, template_small, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val_small, _, max_loc_small = cv2.minMaxLoc(epi_result_small)
            left_max_x_small = left_max_x-finer_range + int(template_width_small * 0.5) + max_loc_small[0]    # left + half_template + max
            disp2 = left_max_x_small - template_c[0]

            if disp2 < 0 or disp2 > 100 :   # TODO need to fill in the hole
            # if max_val < 0.45 or disp < 0 or disp > 100 :
                disp2 = 0
            disparity_map[y,x] = disp2

            if verbosity == True:
                print('disp',disp,'disp2',disp2)
                # print("e_max_x",e_max_x,'e_max_x_small',e_max_x_small)
                # print(template_c)

                left_img_color = cv2.cvtColor(leftImg, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(left_img_color, (max_loc[0],top), (max_loc[0] + template_width,top + template_height), (0,255,255), 1)   # yellow
                # cv2.imshow('ev_img_find',ev_img_color)
                cv2.circle(left_img_color,([left_max_x_small,template_c[1]]),2,(0,255,255),-1)
                cv2.rectangle(left_img_color, (left_max_x_small-int(template_width_small * 0.5),top_small), (left_max_x_small + int(template_width_small * 0.5),top_small + template_height_small), (0,255,255), 1)   # yellow


                right_img_color = cv2.cvtColor(rightImg, cv2.COLOR_GRAY2BGR)
                cv2.circle(right_img_color,template_c,2,(0,0,255),-1)
                cv2.rectangle(right_img_color,  (left, top), (left + template_width, top + template_height),(0,0,255), 1)  # red
                # cv2.imshow('left_img_color',left_img_color)
                cv2.rectangle(right_img_color,  (left_small, top_small), (left_small + template_width_small, top_small + template_height_small),(0,0,255), 1)  # red


                cv2.imshow('left_img_color,right_img_color',cv2.hconcat([left_img_color,right_img_color]))
                k = cv2.waitKey(0)
                if k == 27:         # ESC
                    cv2.destroyAllWindows() 

    if show_FLAG == True:
        right_disparity_norm = img_normalization(disparity_map)                                                        
        cv2.imshow("right_disparity_norm",right_disparity_norm)
        print('show right_disparity')
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            # cv2.imwrite(save_path + '/' + 'right_disparity_norm_kitti' + '.png',right_disparity_norm)
            # cv2.imwrite(save_path + '/' + 'left_disparity_norm_kitti' + '.png',left_disparity_norm)
            cv2.destroyAllWindows()           

    if save_path is not None:
        disparity_map_uint8 = np.uint8(disparity_map)
        cv2.imwrite(save_path + '/' + 'origin_right_disparity' + '.png',disparity_map_uint8) 
    return disparity_map

def rightDisparity(leftImg, rightImg, template_size = (100,100), 
                                                                show_FLAG = False, 
                                                                verbosity = False, 
                                                                save_path = None, 
                                                                stepsize = 1): 
    print('compute right disparity')
    img_height,img_width = leftImg.shape
    disparity_map = np.zeros((img_height,img_width), np.float16)  
    template_width, template_height = template_size 
    mis_th = int(img_width/7)
    step_num = 0
    wrong_num = 0
    test_wr_num =0
    for x in range(int(template_width * 0.5),int(img_width - template_width * 0.5),stepsize):
        for y in range(int(template_height * 0.5),int(img_height - template_height * 0.5),stepsize):
    # for x in range(400,int(img_width - template_width * 0.5),stepsize):
    #     for y in range(200,int(img_height - template_height * 0.5),stepsize):
            step_num = step_num + 1
            template_c = (x,y)
            # print(template_c)
            template_c = (x,y)
            top = template_c[1] -int(template_height * 0.5)
            left = template_c[0] - int(template_width * 0.5)
            template = rightImg[top:top + template_height, left:left + template_width]   # TODO top down left right    

            left_epi_img = leftImg[top:top + template_height,:] # top down left right; shape (20,640)
            epi_result = cv2.matchTemplate(left_epi_img, template, cv2.TM_CCOEFF_NORMED) # shape (1,621)
            _, max_val, _, max_loc = cv2.minMaxLoc(epi_result)
            left_max_x = max_loc[0] + int(template_width * 0.5) 
            disp = left_max_x - template_c[0] 
            if disp < 0 or disp > mis_th :
                disparity_map[y,x] = 0
                continue  
            disparity_map[y,x] = disp

    if show_FLAG == True:
        right_disparity_norm = img_normalization(disparity_map)                                                        
        cv2.imshow("right_disparity_norm",right_disparity_norm)
        print('show right_disparity')
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            # cv2.imwrite(save_path + '/' + 'right_disparity_norm_kitti' + '.png',right_disparity_norm)
            # cv2.imwrite(save_path + '/' + 'left_disparity_norm_kitti' + '.png',left_disparity_norm)
            cv2.destroyAllWindows()           

    if save_path is not None:
        disparity_map_uint8 = np.uint8(disparity_map)
        cv2.imwrite(save_path + '/' + 'origin_right_disparity' + '.png',disparity_map_uint8) 
    return disparity_map



def compute_leftright_disparity(cam1_image_rectified, cam2_image_rectified,stepsize = 1,show_FLAG = False): # show_FLAG for the disp
    print('compute_leftright_disparity')
    save_path = '/Users/cainan/Desktop/Project/data/processed/disparity'
    save_path = None

    left_disparity = left_disparityPy(cam1_image_rectified, 
                                                                cam2_image_rectified, 
                                                                template_size = (60,60), 
                                                                show_FLAG = False, 
                                                                verbosity = False, 
                                                                save_path = save_path,
                                                                stepsize = stepsize)
    right_disparity = rightDisparity(cam1_image_rectified, 
                                                                cam2_image_rectified, 
                                                                template_size = (60,60), 
                                                                show_FLAG = False, 
                                                                verbosity = False, 
                                                                save_path = save_path,
                                                                stepsize = stepsize)
    print('compute done') 
    if show_FLAG == True:
        left_disparity_norm = img_normalization(left_disparity)      
        right_disparity_norm = img_normalization(right_disparity)                                                     
        # cv2.imshow("left_disparity_norm",left_disparity_norm)
        cv2.imshow('left_disparity_norm,right_disparity_norm',cv2.hconcat([left_disparity_norm,right_disparity_norm]))
        print('show left_disparity right_disparity')
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            # cv2.imwrite(save_path + '/' + 'right_disparity_norm_kitti' + '.png',right_disparity_norm)
            # cv2.imwrite(save_path + '/' + 'left_disparity_norm_kitti' + '.png',left_disparity_norm)
            cv2.destroyAllWindows()  


    return [left_disparity,right_disparity]


# def load_rect_disp_EF():
#     show_FLAG = False
#     [frame_image, event_image] = loadEvFrame(show_FLAG = show_FLAG)
#     # old_rectifyEF(cam2_image, cam1_image, show_FLAG = False)   # load is alreeady rectified✅
#     rectifiedImgList,rectifiedParas  = rectifyEF(frame_image, event_image, show_FLAG = show_FLAG)
#     [frame_image_rectified,event_image_rectified] = rectifiedImgList
#     [R1, R2, P1, P2, Q] = rectifiedParas
#     stepsize = 1
#     [left_disparity,right_disparity] = ut.compute_leftright_disparity(frame_image_rectified, 
#                                                                                         event_image_rectified,
#                                                                                         stepsize = stepsize,
#                                                                                         show_FLAG = True)



def img_normalization(img,percentile_low = 0.05,percentile_high = 99.95):
    norm_img = img.copy()
    rmin,rmax = np.percentile(norm_img,(percentile_low,percentile_high))
    # print('rmin,rmax',rmin,rmax)
    scale = 255/(rmax - rmin)
    # print('min' ,rmin,'max',rmax,'scale',scale)
    norm_img = (norm_img - rmin) * scale
    norm_img = np.uint8(norm_img)
    return norm_img  

def warp(left_disparity,Q,P2,cam1_image_rectified, show_FLAG = False, depth_FLAG = False):
    print('warping...')
    # 2D to 3D      reprojectImageTo3D
    points_3d = cv2.reprojectImageTo3D(left_disparity, Q, handleMissingValues= True)
    # print('points_3d.shape',points_3d.shape)
    # shape = points_3d.shape
    height,width,_ = points_3d.shape

    # looking into the depth map
    if depth_FLAG ==  True:
        depthMap = points_3d[:, :, 2]  # max 10000 min 5.2729836
        depthMap = np.zeros((480,640),np.float32) - depthMap
        # v_max = 100
        # depthMap[depthMap>= v_max]=0
        depthMap[depthMap <= 0]=0
        # depthMap = depthMap.astype(np.float32)  # max 527 min 0
        depthMap_norm = img_normalization(depthMap)        
        cv2.imshow('depthMap_norm',depthMap_norm)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            # depth_save_path = '/Users/cainan/Desktop/Project/data/processed/warped/depth'
            # cv2.imwrite(depth_save_path + '' + '.png',depthMap_norm)
            cv2.destroyAllWindows() 
    # EF max 0 min -44152

    # 3D to 2D   using P2 and left image
    pointsHomo = np.ones((height,width,4),np.float32)
    pointsHomo[:,:,:3] = points_3d
    warped_img = np.zeros((height,width),np.uint8)
    count = 0
    for x in range(width):
        for y in range(height):
            # print(pointsHomo[y,x])
            point = pointsHomo[y,x]
            if np.all(np.isfinite(point)[:] == True) and point[2] > 0:  # if the point is available
                # print(point)  # [-35.13595     3.7846594  52.729836    1.       ]
                warped_right_point = np.dot(P2,point)
                warped_right_point = warped_right_point / warped_right_point[2]
                if int(warped_right_point[1]) <= height and int(warped_right_point[0]) <= width:  # if the warped 2Dpoint in the image range
                    warped_img[int(warped_right_point[1]),int(warped_right_point[0])] = cam1_image_rectified[y,x]
                    count = count+1
            # else:
            #     print('hole/false')
    print('count',count) 
    print('warp done')
    if show_FLAG == True:
        cv2.imshow('warped_img',warped_img)
        # k = cv2.waitKey(0)
        # if k == 27:         # ESC
        #     # warp_save_path = '/Users/cainan/Desktop/Project/data/processed/warped'
        #     # cv2.imwrite(warp_save_path + '/' + 'warped' + '.png',warped_img)
        #     cv2.destroyAllWindows() 
    return warped_img



















