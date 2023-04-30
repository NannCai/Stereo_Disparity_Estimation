import numpy as np
# from src.io.psee_loader import PSEELoader
import cv2
import matplotlib.pyplot as plt

# want to see the ev_image frame_image 


def img_normalization(img,percentile_low = 0.1,percentile_high = 99.9):
    rmin,rmax = np.percentile(img,(percentile_low,percentile_high))
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
    return img  

def plot(xpoints, ypoints):
    plt.figure(num = 1,figsize = (8,4)) 
    if np.max(np.abs(ypoints)) > 1 :
        print('ypoints value is wrong!!!!')
    index_ymax = np.argmax(ypoints)
    # print(index_ymax)   # 33
    plt.ylim([-1.2, 1.2])  # 刻度范围
    plt.xticks([])  # 坐标刻度不可见
    plt.yticks([-1,-0.5,0,0.5,1])

    ax = plt.gca()
    ax.spines["top"].set_color('none')#上轴不显示
    ax.spines["right"].set_color('none')#右
    ax.spines["bottom"].set_color('blue')
    ax.spines["bottom"].set_position(('data', 0))
    plt.plot(xpoints, ypoints, color='blue', linewidth=2, linestyle='-')
    plt.plot(xpoints[index_ymax], ypoints[index_ymax], color='red', marker='o')
    plt.show()

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - 我们在img2相应位置绘制极点生成的图像 line zai img1 shang
        lines - 对应的极点 '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

if __name__ == '__main__':

    save_path = '/Users/cainan/Desktop/Project/data/processed'

    '''
    LOAD DATA
    '''
    # load event preprocessed img
    ev_img_path = '/Users/cainan/Desktop/Project/data/processed'
    ev_img_filename = 'event_binary0.png'
    ev_file = ev_img_path + '/' + ev_img_filename
    ev_img = cv2.imread(ev_file, 0)  # shape (480,640)
    # The baseline b = 65.44 mm, focal lengths f = 555 pixels
    ev_b = 65.44
    ev_f = 555.0

    # load frame preprocessed img
    prepro_frame_img_path = ev_img_path
    prepro_frame_img_filename = 'frame_binary0.png'
    pre_f_file = prepro_frame_img_path + '/' + prepro_frame_img_filename
    pre_frame_img = cv2.imread(pre_f_file, 0)  # # shape (1536,2048)  3.2times of 480

    '''
    processing (resize)frame and pick the patch from frame
    '''
    frame_f = 1301.0  # f = 1301 pixels for FLIR camera.
    ratio = ev_f / frame_f   # 0.4265949269792467
    # resize_size = (int(ratio * pre_frame_img.shape[1]),int(ratio * pre_frame_img.shape[0]))
    
    resize_frame_img = cv2.resize(pre_frame_img, (640,480))   # (width,height) xy
    patch_c = [324, 346]   # [x,y] TODO put patch in variable center and the height and width
    
    patch_height = 50
    patch_width = 20
    print('ph pw from variable', patch_height,patch_width)
    # patch = resize_frame_img[240:280, 110:190]   # height width
    top =patch_c[1] -int( patch_height * 0.5)
    left = patch_c[0] - int(patch_width * 0.5)
    # print('left' , left) 
    patch = resize_frame_img[top:top + patch_height, left:left + patch_width]   # top down left right
    # print('patch_height, patch_width from patch', patch.shape[:2])
    

    color_frame_patch = cv2.cvtColor(resize_frame_img, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_frame_patch,tuple(patch_c),2,(0,0,255),-1)
    cv2.rectangle(color_frame_patch,  (left, top), (left + patch_width, top + patch_height),(0,0,255), 2)  # red
    cv2.imshow('color_frame_patch',color_frame_patch)
    # print('show')
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     # cv2.imwrite(save_path + '/' + 'ev_img_find' + str(i) + '.png',ev_img)
    #     # cv2.imwrite(save_path + '/' + 'NCC_result' + str(i) + '.png',norm_result)
    #     cv2.destroyAllWindows()

    '''
    NCC cross correlation  TODO do it only on the epipolar line 
    '''

    # TODO 	cv2.computeCorrespondEpilines(	points, whichImage, F[, lines]	)
    # at first find out the right position of patch center that will ues in comp_epi's points ✅
    # find out F is corresponding to which camera(ev->fra or fra->ev)
    #   write F in correct form
    list =[5.308170932110807e-07 ,    3.219351527219740e-05,    -1.890233407724186e-02,
    -2.779092523559830e-05  ,   7.721963665410085e-06  ,  -1.532673732470708e-01,
    1.639650649737013e-02,     1.046480379983835e-01  ,   1.671559913658885e+01]
    F = np.array(list).reshape(3,3)
    alfa = ratio
    S = [alfa,0,0,0,alfa,0,0,0,1]
    S = np.array(S).reshape(3,3)
    # print(S)
    points = []
    points.append(patch_c)
    # points.append(patch_c)
    points = np.int32(points)
    # for point in points:
    #     print(point)
    lines = cv2.computeCorrespondEpilines(points, 2, F)  # points在frame上 line在event上
    lines = lines.reshape(-1,3)
    print(lines)  # [[-7.5023495e-02 -9.9718177e-01  2.7069153e+02]]
    r,c = ev_img.shape
    for r in lines:
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
    ev_img_epi = cv2.line(ev_img, (x0,y0), (x1,y1), (255,0,0),1)
    cv2.imshow('ev_img_epi',ev_img_epi)

    k = cv2.waitKey(0)
    if k == 27:         # ESC
        cv2.destroyAllWindows()


    quit()
    result = cv2.matchTemplate(ev_img, patch, cv2.TM_CCOEFF_NORMED) # (origin(big_img), patch(Template), mat_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    NCC_norm_result = img_normalization(result)  # shape 441 561 and np
    epipolar_line = patch_c[0]   # TODO  这里是错的

    epi_result = result[epipolar_line]  # size 561 array
    x = np.array([i + 40 for i in range(result.shape[1])]) # len 561

    plot(x, epi_result)  # plot NCC result on the epipolar line
    # TODO draw the epipolar line on the event image
    top_left = max_loc
    bottom_right = (top_left[0] + patch_width, top_left[1] + patch_height)
 
    ev_img = cv2.cvtColor(ev_img, cv2.COLOR_GRAY2BGR)
    print('top_left',top_left,' bottom_right',bottom_right)
    # cv2.line(ev_img,(0, patch_c[0]),(640, patch_c[0]),(255,0,0),5)
    cv2.rectangle(ev_img, top_left, bottom_right, (0,255,255), 2) # 在原图上画矩形
    cv2.imshow('ev_img_find',ev_img)
    cv2.imshow('NCC_norm_result',NCC_norm_result)

    k = cv2.waitKey(0)
    i = 1
    if k == 27:         # ESC
        # cv2.imwrite(save_path + '/' + 'ev_img_find' + str(i) + '.png',ev_img)
        # cv2.imwrite(save_path + '/' + 'NCC_result' + str(i) + '.png',norm_result)
        cv2.destroyAllWindows()





