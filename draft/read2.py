# import numpy as np
# import  pandas  as pd
# import loris
import numpy as np
from src.io.psee_loader import PSEELoader
import cv2


def NMS(gradients, direction):
    """ Non-maxima suppression

    Args:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel

    Returns:
        the output image
    """
    W, H = gradients.shape
    nms = np.copy(gradients[1:-1, 1:-1])

    for i in range(1, W - 1):
        for j in range(1, H - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
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


def e_file_load(e_filedir, e_filename):
    file = e_filedir + '/' + e_filename
    video = PSEELoader(file)
    # print(video)  # PSEELoader: -- config ---
    # # print(video.event_count())  # number of events in the file
    # # print(video.total_time())  # duration of the file in mus

    events = video.load_n_events(video.event_count())
    # print(np.sum(events['x'] > 480))   # 640
    # print(np.sum(events['y'] > 480))   # 480
    # print(events)
    # print(events['t'][154751803])  # this shows only the timestamps of events
    # for instance to count the events of positive polarity you can do :
    # print(np.sum(events['p'] > 0))
    # events = video.load_delta_t(10000)
    # print(events[0])
    # print(type(events))
    # print(events.shape)   # events[0] <class 'numpy.void'>  自定义记录类型

    # events[0]['t'] <class 'numpy.uint32'>
    # events = np.array(events)
    # print(events[:,0])

    # evs = events.reshape((-1,1))
    # print(type(evs[0]))
    # python numpy.array 与list类似，不同点：前者区分元素不用逗号，中间用空格,矩阵用[]代表行向量，两个行向量中间仍无逗号；
    
    return events
    # return evs

def frame_ts_load(f_filedir, f_filename):
    f_file = f_filedir + '/' + f_filename
    frame_timestamps = []
    with open(f_file,"r") as f:
        for line in f:
            # print(line)
            line=line.strip('\n')
            frame_timestamps.append(line)
  
    return frame_timestamps

def img_normalization(img):
    rmin,rmax = np.percentile(img,(0.1,99.9))
    # rmin = np.min(img)  # TODO robust-去掉极端值
    # rmax = np.max(img)
    print('min' ,rmin)
    print('max',rmax)
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
    # rmin = np.min(decay_img)  # TODO robust-去掉极端值
    # rmax = np.max(decay_img)
    # scale = 255/(rmax - rmin)
    # decay_img = (decay_img - rmin) * scale
    return img

def img_ternary(img):  # ternary 三元   only 1 0 -1
    rmin = np.min(img)  # TODO robust-去掉极端值
    rmax = np.max(img)
    print('min' ,rmin)
    print('max',rmax)
    img[img > 0] = 1
    img[img < 0] = -1
    img = img_normalization(img)
    return img

def img_normalization_debug(img, num = 0):
    rmin = np.min(img)  # TODO robust-去掉极端值
    rmax = np.max(img)
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
    show_name = 'debug' + str(num)
    # cv2.imshow(show_name,img)
    # cv2.waitKey()
    return img


if __name__ == '__main__':

    '''
    LOAD DATA
    '''
    # load events from .dat file
    e_filedir = '/Users/cainan/Desktop/Project/data/01_simple'
    e_filename = 'log_td.dat'   
    # file = '/Users/cainan/Desktop/ser/Project/data/01_simple/log_td.dat'
    events = e_file_load(e_filedir, e_filename)
    # print(len((events)['t']))  # 154751804

    # load frame timestamp
    f_filedir = e_filedir
    f_filename = 'image_ts.txt'
    frame_timestamps = frame_ts_load(f_filedir, f_filename)
    # print(frame_timestamps)

    '''
    find nearest ev_ts and take the around event sub_package and get the recon img of sub_package
    '''
    for f_ts in frame_timestamps:

        # @find nearest ev_ts
        # print(type(f_ts))  # <class 'str'>
        print(f_ts)  # 9122644  9141145  9159645
        f_ts = float(f_ts)
        # print(type(f_ts))  # <class 'float'>
        abs_list = np.abs(events['t'] - f_ts)
        min_index = np.argmin(abs_list)    # TODO maybe bug here  a lot same ev-t of diff pix
        # index_gt = np.argmin(np.abs(f_ts - events[:, 0]))
        
        # @take the around event sub_package
        # num_events = 15000
        num_events = 500000
        events_sub = events[min_index - num_events:min_index + num_events]
        t_first = events['t'][min_index - num_events]
        
        # print(events['t'][num_events * 2])  # 19176 for30000 the around interval of num_events
        # print(len(events_sub))  # 30000
        # print(events['t'][min_index])  # 9122644
        # print(len(abs_list))  

        # @get the recon img
        # Initial
        img_size = (480,640)   # (height, width)
        state_time_map_ = np.zeros(img_size, np.float64) + t_first   # ! here is mus!
        # state_time_map_ = cv::Mat(msg->height, msg->width, CV_64F,cv::Scalar(t_first_));
        # state_image_ = cv::Mat::zeros(msg->height, msg->width, CV_64F);
        state_image_ = np.zeros(img_size, np.float64)
        # Process
        c_pos_ = 0.1
        alpha_cutoff_ = 120
        for i, ev in enumerate(events_sub):
            # print(state_time_map_[ev['y'],ev['x']])
            # print('ev:',ev)
            delta_ts = ev['t'] - state_time_map_[ev['y'],ev['x']]
            delta_ts = delta_ts * 1e-06
            # if delta_ts > 0:
            #     print(delta_ts)
            l_beta = np.exp(-alpha_cutoff_ * delta_ts)
            if ev['p'] == 1:
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] + c_pos_
                # print("pos")
            elif ev['p'] == 0:
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] - c_pos_
                # print("neg")
            else:
                print("!!!!False")

            state_time_map_[ev['y'],ev['x']] = ev['t']   # min&max cha19111

        # state_time_map_ = img_normalization_debug(state_time_map_)
        # cv2.imshow("state_time_map_",state_time_map_)
        # cv2.waitKey(0)

        # Publish
        # cv::Mat last_delta = t_last_ - state_time_map_; //is pos value
        # cv::Mat decay_img;
        # cv::Mat img_beta;   // the beta of state_image_
        # cv::exp((-alpha_cutoff_ * last_delta) , img_beta);
        # decay_img = img_beta.mul(state_image_);  
        t_last_ = events_sub['t'][-1]
        last_delta = t_last_ - state_time_map_  # min max cha19111
        last_delta = last_delta * 1e-06     # min max cha0.019111
        # decay_img = -alpha_cutoff_ * last_delta
        beta = np.exp(-alpha_cutoff_ * last_delta)
        decay_img = beta * state_image_

        # convert to appropriate range, [0,255]
        decay_img = img_normalization(decay_img)
        # decay_img = img_ternary(decay_img)

        print('show')
        cv2.imshow('line',decay_img)
        cv2.waitKey(0)



        if f_ts > 9141145:
            break




