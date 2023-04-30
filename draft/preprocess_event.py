# import  pandas  as pd
# import loris
import numpy as np
from src.io.psee_loader import PSEELoader
import cv2
# implement the Ex4 intergrator of events

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

def img_normalization(img,percentile_low = 0.2,percentile_high = 99.8):
    norm_img = img.copy()
    rmin,rmax = np.percentile(norm_img,(percentile_low,percentile_high))
    scale = 255/(rmax - rmin)
    print('min' ,rmin,'max',rmax,'scale',scale)
    norm_img = (norm_img - rmin) * scale
    norm_img = np.uint8(norm_img)
    return norm_img  

if __name__ == '__main__':

    Flag = False

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

    '''
    find nearest ev_ts and take the around event sub_package and get the recon img of sub_package
    '''
    # Initial
    img_size = (480,640)   # (height, width)
    # the cam move after the first frame ts
    # t_first_index = np.argmin(np.abs(events['t'] - 9122644 - 2))  # 14141717
    t_first_index = np.argmin(np.abs(events['t'] - 9959645 - 2))
    t_first = events[t_first_index]['t']
    state_time_map_ = np.zeros(img_size, np.float64) + t_first
    # ?sth wrong? to set the state time map?
    # state_time_map_ = cv::Mat(msg->height, msg->width, CV_64F,cv::Scalar(t_first_));
    state_image_ = np.zeros(img_size, np.float64)
    c_pos_ = 0.1
    alpha_cutoff_ = 120
    i = 8
    # 因为有time map所以要存ts 因为有event的sub拿取所以要存index
    for f_ts in frame_timestamps:
        if i in [29,30,31]:
            Flag = True

        # @find nearest ev_ts
        print(f_ts,"=========")     # 10121662.0
        # print(f_ts)  # 9122644  9141145  9159645    type(f_ts))  # <class 'str'>
        t_last_index = np.argmin(np.abs(events['t'] - f_ts - 2))    # TODO maybe bug here  a lot same ev-t of diff pix
        print('t_first_index',t_first_index,'t_last_index',t_last_index,'event-number',t_last_index-t_first_index)    # t_first_index 14141717 t_last_index 20325250 event-number 6183533
        # if t_last_index - t_first_index <= 100000: # at least 100000 events per integrate
        #     print('index wrong!!!!')
            # quit()
        events_sub = events[t_first_index:t_last_index]
        t_first_index = t_last_index + 1

        for ev in events_sub:
            delta_ts = (ev['t'] - state_time_map_[ev['y'],ev['x']]) * 1e-06
            l_beta = np.exp(-alpha_cutoff_ * delta_ts)
            if ev['p'] == 1:
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] + c_pos_
            else :
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] - c_pos_
            state_time_map_[ev['y'],ev['x']] = ev['t']   # min-max cha19111

        state_time_map_show = img_normalization(state_time_map_)   # state_time_map_ have mius in it
        cv2.imshow("state_time_map_show",state_time_map_show)  # event timesurface 
        # print('debug')
        save_path = '/Users/cainan/Desktop/Project/data/processed/highpass/time_map' + '/'
        print(i)
        cv2.imwrite(save_path + 'state_time_map_show' + str(i) + '.png',state_time_map_show)
        if i >= 29:
            k = cv2.waitKey(0)
            if k == 27 :        
                # save_path = '/Users/cainan/Desktop/Project/data/processed/highpass/time_map' + '/'
                # print(i)
                # cv2.imwrite(save_path + 'state_time_map_show' + str(i) + '.png',state_time_map_show)
                cv2.destroyAllWindows()
        # Publish
        t_last_ = events_sub['t'][-1]
        last_delta = (t_last_ - state_time_map_) * 1e-06  # min-max = 0.019111
        beta = np.exp(-alpha_cutoff_ * last_delta)
        decay_img = state_image_.copy()
        decay_img = beta * decay_img
        # state_time_map_ = np.zeros(img_size, np.float64) + t_last_

        # binary and convert to appropriate range, [0,255]
        norm_decay_img = img_normalization(decay_img)
        ret, event_binary = cv2.threshold(norm_decay_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        # event_binary = img_normalization(event_binary)

        print('show end')
        cv2.imshow('event_binary',event_binary)   # shape (480,640)
        cv2.imshow('norm_decay_img',norm_decay_img) 
        # if Flag:
        #     print(norm_decay_img)
        #     cv2.imshow('norm_decay_img',norm_decay_img) 

        save_path = '/Users/cainan/Desktop/Project/data/processed/highpass/state_image' + '/'
        cv2.imwrite(save_path + 'event_binary' + str(i) + '.png',event_binary)
        cv2.imwrite(save_path + 'norm_decay_img' + str(i) + '.png',norm_decay_img)
        if i>=29:
            k = cv2.waitKey(0)
            if k == 27  :         # ESC
                # save_path = '/Users/cainan/Desktop/Project/data/processed/highpass/state_image' + '/'
                # cv2.imwrite(save_path + 'event_binary' + str(i) + '.png',event_binary)
                # cv2.imwrite(save_path + 'norm_decay_img' + str(i) + '.png',norm_decay_img)
                # i = i+1
                cv2.destroyAllWindows()
        i = i+1
        # if f_ts > 9141145:
        #     break



