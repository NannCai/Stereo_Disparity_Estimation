# import  pandas  as pd
# import loris
import numpy as np
from src.io.psee_loader import PSEELoader
import cv2

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
            line=line.strip('\n')
            frame_timestamps.append(float(line))
    frame_timestamps = np.array(frame_timestamps)
    static_interval = 1000000

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
    print('min' ,rmin)
    print('max',rmax)
    scale = 255/(rmax - rmin)
    img = (img - rmin) * scale
    img = np.uint8(img)
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

    # load frame img
    frame_img_dir = '/Users/cainan/Desktop/Project/data/01_simple/png'
    frame_img_name = '1.png'
    dst = frame_img_load(frame_img_dir, frame_img_name)
    ret, frame_binary = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    # print("threshold value %s"%ret)   #TODO threshold value 49.0  # don't know for what
    cv2.imshow("frame_binary", frame_binary)   # shape (1536,2048)

    '''
    find nearest ev_ts and take the around event sub_package and get the recon img of sub_package
    '''
    # Initial
    img_size = (480,640)   # (height, width)
    t_first = events['t'][0]
    state_time_map_ = np.zeros(img_size, np.float64) + t_first
    state_image_ = np.zeros(img_size, np.float64)
    c_pos_ = 0.1
    alpha_cutoff_ = 120
    # last_index = 0  # 9122644 对应的index
    last_index = np.argmin(np.abs(events['t'] - 9122644 - 2))
    i = 8
    # 因为有time map所以要存ts 因为有event的sub拿取所以要存index
    for f_ts in frame_timestamps:

        # @find nearest ev_ts
        print(f_ts)     # 10121662.0
        # print(f_ts)  # 9122644  9141145  9159645    type(f_ts))  # <class 'str'>
        abs_list = np.abs(events['t'] - f_ts - 2) 
        min_index = np.argmin(abs_list)    # TODO maybe bug here  a lot same ev-t of diff pix
        print('last_index',last_index,'min_index',min_index)    # last_ 14141717 min_ 20325250
        events_sub = events[last_index:min_index]
        last_index = min_index

        for ev in events_sub:
            delta_ts = ev['t'] - state_time_map_[ev['y'],ev['x']]
            delta_ts = delta_ts * 1e-06
            l_beta = np.exp(-alpha_cutoff_ * delta_ts)
            if ev['p'] == 1:
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] + c_pos_
            else :
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] - c_pos_
            state_time_map_[ev['y'],ev['x']] = ev['t']   # min-max cha19111

        state_time_map_show = img_normalization(state_time_map_)   # state_time_map_ have mius in it
        cv2.imshow("state_time_map_show",state_time_map_show)
        print('debug')
        k = cv2.waitKey(0)
        if k == 27:        
            cv2.destroyAllWindows()
        # Publish
        t_last_ = events_sub['t'][-1]
        last_delta = (t_last_ - state_time_map_) * 1e-06  # min-max = 0.019111
        beta = np.exp(-alpha_cutoff_ * last_delta)
        decay_img = beta * state_image_
        state_time_map_ = np.zeros(img_size, np.float64) + t_last_

        # binary and convert to appropriate range, [0,255]
        # decay_img = np.uint8(decay_img)
        decay_img = img_normalization(decay_img)
        ret, event_binary = cv2.threshold(decay_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        # event_binary = img_normalization(event_binary)

        print('show end')
        cv2.imshow('event_binary',event_binary)   # shape (480,640)

        save_path = '/Users/cainan/Desktop/Project/data/processed' + '/'
        
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.imwrite(save_path + 'event_binary' + str(i) + '.png',event_binary)
            cv2.imwrite(save_path + 'frame_binary' + str(i) + '.png',frame_binary)
            i = i+1
            cv2.destroyAllWindows()

        # if f_ts > 9141145:
        #     break



