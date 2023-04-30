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
    return frame_timestamps

def frame_img_load(frame_img_dir, frame_img_name):
    frame_img_file = frame_img_dir + '/' + frame_img_name
    frame_img = cv2.imread(frame_img_file, 0)
    grad_x = cv2.Sobel(frame_img, cv2.CV_64F, 1,0)
    grad_y = cv2.Sobel(frame_img, cv2.CV_64F, 0,1)
    
    absX = cv2.convertScaleAbs(grad_x)  # back to uint8
    absY = cv2.convertScaleAbs(grad_y)
    # nms = NMS(absX, absY)
    # nms = NMS(grad_x, grad_y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    return dst

def img_normalization(img,percentile_low = 0.1,percentile_high = 99.9):
    norm_img = img.copy()
    rmin,rmax = np.percentile(norm_img,(percentile_low,percentile_high))
    scale = 255/(rmax - rmin)
    print('min' ,rmin,'max',rmax,'scale',scale)
    norm_img = (norm_img - rmin) * scale
    norm_img = np.uint8(norm_img)
    return norm_img  

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
    static_interval_time = 4000000
    # f_start_index = np.argmin(np.abs(frame_timestamps - frame_timestamps[0] - static_interval_time))
    f_start_index = 98
    print('f_start_index',f_start_index,'f_ts_len',len(frame_timestamps))  # 500000-27 1000000-54

    # load frame img
    frame_img_dir = '/Users/cainan/Desktop/Project/data/01_simple/png'

    '''
    find nearest ev_ts and take the around event sub_package and get the recon img of sub_package
    '''
    # Initial
    img_size = (480,640)   # (height, width)
    print(f_start_index-2)
    t_first_index = np.argmin(np.abs(events['t'] - frame_timestamps[f_start_index-2] - 2))
    t_first = events[t_first_index]['t']
    state_time_map_ = np.zeros(img_size, np.float64) + t_first
    state_image_ = np.zeros(img_size, np.float64)
    c_pos_ = 0.1
    alpha_cutoff_ = 120
    # alpha_cutoff_ = 10
    f_index = f_start_index
    # time map need ts; event_sub need index
    for f_ts in frame_timestamps[f_start_index:]:
        # load frame img
        frame_img_name = str(f_index) + '.png'
        f_index = f_index + 1
        dst = frame_img_load(frame_img_dir, frame_img_name)
        ret, frame_binary = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        # print("threshold value %s"%ret)   # threshold value 49.0  # don't know for what
        cv2.imshow("frame_binary", frame_binary)   # shape (1536,2048)

        # @find nearest ev_ts
        print(f_ts,"=========")     # 10121662.0
        t_last_index = np.argmin(np.abs(events['t'] - f_ts - 2))    # TODO maybe bug here  a lot same ev-t of diff pix
        print('t_first_index',t_first_index,'t_last_index',t_last_index,'event-number',t_last_index-t_first_index)    # t_first_index 14141717 t_last_index 20325250 event-number 6183533
        if t_last_index - t_first_index <= 10000: # at least 100000 events per integrate
            print('index wrong!!!!')
            quit()
        events_sub = events[t_first_index:t_last_index]
        t_first_index = t_last_index + 1

        for ev in events_sub:
            delta_ts = ev['t'] - state_time_map_[ev['y'],ev['x']]
            delta_ts = delta_ts * 1e-06
            l_beta = np.exp(-alpha_cutoff_ * delta_ts)
            if ev['p'] == 1:
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] + c_pos_
            else :
                state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] - c_pos_
            state_time_map_[ev['y'],ev['x']] = ev['t']   # min-max cha19111

        # state_time_map_show = img_normalization(state_time_map_)   # state_time_map_ have mius in it
        # cv2.imshow("state_time_map_show",state_time_map_show)

        # Publish
        t_last_ = events_sub['t'][-1]
        last_delta = (t_last_ - state_time_map_) * 1e-06  # min-max = 0.019111
        beta = np.exp(-alpha_cutoff_ * last_delta)
        decay_img = beta * state_image_

        decay_img = img_normalization(decay_img)
        print('show end')

        save_path = '/Users/cainan/Desktop/Project/data/processed/cutoff_120' + '/'
        cv2.imwrite(save_path + 'event' + str(f_index) + '.png',decay_img)
        if f_index >= 109:
            k = cv2.waitKey(0)
            if k == 27:         # ESC

                cv2.destroyAllWindows()

        # if f_ts > 9141145:
        #     break



