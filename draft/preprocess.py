# import  pandas  as pd
# import loris
import numpy as np
from src.io.psee_loader import PSEELoader
import cv2
# from matplotlib import pyplot as plt

def e_file_load(e_filedir, e_filename):
    file = e_filedir + '/' + e_filename
    video = PSEELoader(file)
    # print(video)  # PSEELoader: -- config ---
    # print(video.event_count())  # number of events in the file
    # print(video.total_time())  # duration of the file in mus
    events = video.load_n_events(video.event_count())
    return events

def frame_ts_load(f_filedir, f_filename,):
    f_file = f_filedir + '/' + f_filename
    frame_timestamps = []
    with open(f_file,"r") as f:
        for line in f:
            # print(line)
            line=line.strip('\n')
            frame_timestamps.append(float(line))
    # print("frame_timestamps in[0]",frame_timestamps[0])
    # print("frame_timestamps in[2]",frame_timestamps[2])
    # first_frame = frame_timestamps[0]
    frame_timestamps = np.array(frame_timestamps)
    static_interval = 1000000
    # a = frame_timestamps - frame_timestamps[0]
    # b = a-static_interval
    start_move_index = np.argmin(np.abs(frame_timestamps - frame_timestamps[0] - static_interval))
    print('start_move_index',start_move_index)  # 500000-27 1000000-54
    frame_timestamps = frame_timestamps[start_move_index:]
    return frame_timestamps

def frame_img_load(frame_img_dir, frame_img_name):
    frame_img_file = frame_img_dir + '/' + frame_img_name
    frame_img = cv2.imread(frame_img_file, 0)
    grad_x = cv2.Sobel(frame_img, cv2.CV_64F, 1,0)
    grad_y = cv2.Sobel(frame_img, cv2.CV_64F, 0,1)
    
    absX = cv2.convertScaleAbs(grad_x)  # 转回uint8
    absY = cv2.convertScaleAbs(grad_y)
    # nms = NMS(absX, absY)
    # nms = NMS(grad_x, grad_y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    return dst

def img_normalization(img,percentile_low = 0.1,percentile_high = 99.9):
    rmin,rmax = np.percentile(img,(percentile_low,percentile_high))
    # print('min' ,rmin)
    # print('max',rmax)
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
    return img  

def NMS(gradientx, gradienty):
    """ Non-maxima suppression

    Args:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel

    Returns:
        the output image
    """
    W, H = gradientx.shape

    # absX = cv2.convertScaleAbs(gradientx)  # 转回uint8
    # absY = cv2.convertScaleAbs(gradienty)
    absX = gradientx
    absY = gradienty
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    gradients = dst
    nms = np.copy(gradients[1:-1, 1:-1])
    direction = np.zeros(gradients.shape)
    for i in range(W - 2):
        for j in range(H - 2):
            # dx = np.sum(image[i:i+3, j:j+3] * Sx)
            # dy = np.sum(image[i:i+3, j:j+3] * Sy)
            # gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
            dy = gradienty[i,j]
            dx = gradientx[i,j]
            if dx == 0:   # attention ！！when denominator is 0
                direction[i, j] = np.pi / 2
            else:
                direction[i, j] = np.arctan(dy / dx)


    # direction = 0
    for i in range(1, W - 1):
        for j in range(1, H - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
            print('theta = ',theta)
            if theta <= 0:
                print("neg")  # 有很少的neg
            if theta > np.pi / 4:
                d1 = [0, 1]
                d2 = [1, 1]
                weight = 1 / weight
            elif theta >= 0:
                d1 = [1, 0]
                d2 = [1, 1]
            elif theta >= - np.pi / 4:
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else:
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight

            g1 = gradients[i + d1[0], j + d1[1]]
            g2 = gradients[i + d2[0], j + d2[1]]
            g3 = gradients[i - d1[0], j - d1[1]]
            g4 = gradients[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight)
            grade_count2 = g3 * weight + g4 * (1 - weight)

            if grade_count1 > gradients[i, j] or grade_count2 > gradients[i, j]:
                nms[i - 1, j - 1] = 0

    return nms

def img_ternary(img):  # ternary 三元   only 1 0 -1
    rmin = np.min(img)  
    rmax = np.max(img)
    print('min' ,rmin)
    print('max',rmax)
    img[img > 0] = 1
    img[img < 0] = -1
    img = img_normalization(img)
    return img

if __name__ == '__main__':

    '''
    LOAD DATA
    '''
    # load events from .dat file
    e_filedir = '/Users/cainan/Desktop/Project/data/01_simple'
    e_filename = 'log_td.dat'   
    events = e_file_load(e_filedir, e_filename)
    # print(len((events)['t']))  # 154751804

    # load frame timestamp(in ev area)
    f_filedir = e_filedir
    f_filename = 'image_ts.txt'
    frame_timestamps = frame_ts_load(f_filedir, f_filename)
    # frame_timestamps = frame_timestamps[500:]
    # print('frame_timestamps[0]',frame_timestamps[0])
    # print('frame_timestamps[1]',frame_timestamps[1])

    # load frame img
    frame_img_dir = '/Users/cainan/Desktop/Project/data/01_simple/png'
    frame_img_name = '1.png'
    dst = frame_img_load(frame_img_dir, frame_img_name)
    ret, frame_binary = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    print("threshold value %s"%ret)   # threshold value 49.0  # don't know for what
    cv2.imshow("frame_binary", frame_binary)   # shape (1536,2048)


    # print('show')
    # # myimshow(grad_x)  # Image gradient x
    # # cv2.imshow("grad_y",grad_y)  # Image gradient y
    # # cv2.imshow('frame_img',frame_img)
    
    # cv2.imshow("binary0",binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    '''
    find nearest ev_ts and take the around event sub_package and get the recon img of sub_package
    '''
    for f_ts in frame_timestamps:

        # @find nearest ev_ts
        # print(type(f_ts))  # <class 'str'>
        print(f_ts)  # 9122644  9141145  9159645
        # f_ts = float(f_ts)
        abs_list = np.abs(events['t'] - f_ts - 2) 
        # print(type(events['t']))  # <class 'numpy.ndarray'>
        min_index = np.argmin(abs_list)    # TODO maybe bug here  a lot same ev-t of diff pix
        # print("min_index",min_index)  # 14228454
        # @take the around event sub_package
        num_events = 500000
        front = 1500000
        back = 0
        # events_sub = events[min_index - num_events:min_index + num_events]
        events_sub = events[min_index - front:min_index + back]
        t_first = events_sub['t'][0]
        # print('events_sub t [0]',events_sub['t'][0])
        # print('events_sub t [-1]',events_sub['t'][-1])
        
        # @get the recon img
        # Initial
        img_size = (480,640)   # (height, width)
        state_time_map_ = np.zeros(img_size, np.float64) + t_first   # ! here is mus!F);
        state_image_ = np.zeros(img_size, np.float64)
        # Process
        c_pos_ = 0.1
        alpha_cutoff_ = 120
        for i, ev in enumerate(events_sub):
            delta_ts = ev['t'] - state_time_map_[ev['y'],ev['x']]
            delta_ts = delta_ts * 1e-06
            l_beta = np.exp(-alpha_cutoff_ * delta_ts)
            if ev['p'] == 1:
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] + c_pos_
            else :
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] - c_pos_
            state_time_map_[ev['y'],ev['x']] = ev['t']   # min&max cha19111

        # state_time_map_ = img_normalization_debug(state_time_map_)
        # cv2.imshow("state_time_map_",state_time_map_)
        # cv2.waitKey(0)

        # Publish
        t_last_ = events_sub['t'][-1]
        last_delta = (t_last_ - state_time_map_) * 1e-06  # min max cha0.019111
        beta = np.exp(-alpha_cutoff_ * last_delta)
        decay_img = beta * state_image_

        # convert to appropriate range, [0,255]
        decay_img = img_normalization(decay_img)
        ret, event_binary = cv2.threshold(decay_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        # decay_img = img_ternary(decay_img)

        print('show')
        cv2.imshow('event_binary',event_binary)   # shape (480,640)
        # cv2.waitKey(0)


        save_path = '/Users/cainan/Desktop/Project/data/processed' + '/'
        i = 0

        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.imwrite(save_path + 'event_binary' + str(i) + '.png',event_binary)
            cv2.imwrite(save_path + 'frame_binary' + str(i) + '.png',frame_binary)
            cv2.destroyAllWindows()



        # if f_ts > 9141145:
        #     break



