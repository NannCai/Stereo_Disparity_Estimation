import numpy as np
from src.io.psee_loader import PSEELoader
import cv2
import os

def img_show(img,img_name,show_FLAG = True):
    # print(str(img_name))
    if show_FLAG == True:
        cv2.imshow(img_name,img)
        k = cv2.waitKey(0)
        if k == 27:         # ESC
            cv2.destroyAllWindows() 

def load_frame_img(frame_img_dir,index,frame_show = False):
    frame_img_name = str(index) + '.png'
    frame_img_file = frame_img_dir + '/' + frame_img_name
    frame_img = cv2.imread(frame_img_file, 0)  
    img_show(frame_img,'frame_img '+ str(index) ,show_FLAG = frame_show)   
    return frame_img


class Preprocess:
    def __init__(self, processe_number = 10) :
        # 在函数间会用到的
        # self.event_dir = '/Users/cainan/Desktop/Project/data/01_simple'
        self.event_dir = '/Users/cainan/Desktop/Project/data/raw_data/raw_01_complex'
        self.frame_ts_dir = self.event_dir
        self.f_start_index = 230  # 98
        self.img_size = (480,640) 
        self.processe_number = processe_number
        self.c_pos_ = 0.1
        self.alpha_cutoff_ = 120
        self.frame_ts = []
        self.frame_img_dir = self.event_dir + '/png'  # '/Users/cainan/Desktop/Project/data/01_simple/png'
        self.frame_FLAG = True
        # self.save_path = '/Users/cainan/Desktop/Project/data/recon_evImg'
        self.save_path = '/Users/cainan/Desktop/Project/data/recon_evImg/complex'
        print('save_path:',self.save_path)
        if self.processe_number != 0:
            self.parse_data()
            self.run_preprocess()
        
        
    def get_savedir(self):
        return self.save_path

    def parse_data(self):
        dirs = os.listdir(self.event_dir) 
        for file in dirs:
            if 'image_ts.txt' in file:
                frame_ts_file = self.frame_ts_dir + '/' + file
            if 'log_td.dat' in file:
                event_file = self.event_dir + '/' + file

        video = PSEELoader(event_file)
        # print(video)  # PSEELoader: -- config ---
        # print(video.event_count())  # number of events in the file
        # print(video.total_time())  # duration of the file in mus
        self.events = video.load_n_events(video.event_count())
        # print(self.event_file)

        with open(frame_ts_file,"r") as f:
            for line in f:
                line=line.strip('\n')
                self.frame_ts.append(float(line))
        self.frame_ts = np.array(self.frame_ts)

    def img_normalization(self,img,percentile_low = 0.1,percentile_high = 99.9):
        norm_img = img.copy()
        rmin,rmax = np.percentile(norm_img,(percentile_low,percentile_high))
        scale = 255/(rmax - rmin)
        print('min' ,rmin,'max',rmax,'scale',scale)
        norm_img = (norm_img - rmin) * scale
        norm_img = np.uint8(norm_img)
        return norm_img  

    def run_preprocess(self):
        
        t_first_index = np.argmin(np.abs(self.events['t'] - self.frame_ts[self.f_start_index-2]))
        t_first = self.events[t_first_index]['t']
        state_time_map_ = np.zeros(self.img_size, np.float64) + t_first
        state_image_ = np.zeros(self.img_size, np.float64)

        print('f_start_index:',self.f_start_index,'total timestamps:',len(self.frame_ts))
        if len(self.frame_ts) <= self.f_start_index:
            print("the frame index is out of range!")
        for f_ts_index in range(self.f_start_index,self.f_start_index + self.processe_number):  
            if self.frame_FLAG == True:
                frame_img = load_frame_img(self.frame_img_dir,f_ts_index,frame_show=False)
                cv2.imwrite(self.save_path + '/' +'frame' + str(f_ts_index) + '.png',frame_img)
            
            f_ts = self.frame_ts[f_ts_index]            
            print('f_ts_index:', f_ts_index,'f_ts:',f_ts,"=========")

            t_last_index = np.argmin(np.abs(self.events['t'] - f_ts ))
            print('t_first_index',t_first_index,'t_last_index',t_last_index,'event-number',t_last_index-t_first_index)    # t_first_index 14141717 t_last_index 20325250 event-number 6183533
            if t_last_index - t_first_index <= 10000: # at least 100000 events per integrate
                print('event index wrong!!!!')
                quit()
            events_sub = self.events[t_first_index:t_last_index]
            t_first_index = t_last_index + 1
            
            for ev in events_sub:
                delta_ts = ev['t'] - state_time_map_[ev['y'],ev['x']]
                delta_ts = delta_ts * 1e-06
                l_beta = np.exp(-self.alpha_cutoff_ * delta_ts)
                if ev['p'] == 1:
                    state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] + self.c_pos_
                else :
                    state_image_[ev['y'],ev['x']] = l_beta * state_image_[ev['y'],ev['x']] - self.c_pos_
                state_time_map_[ev['y'],ev['x']] = ev['t'] 
            
            # Publish
            t_last_ = events_sub['t'][-1]
            last_delta = (t_last_ - state_time_map_) * 1e-06 
            beta = np.exp(-self.alpha_cutoff_ * last_delta)
            img_out = beta * state_image_

            img_out = self.img_normalization(img_out)
            cv2.imwrite(self.save_path + '/' +'event' + str(f_ts_index) + '.png',img_out)
            img_show(img_out,'img_out ' + str(f_ts_index),show_FLAG = True)

            print('recon_evImg',f_ts_index,'finish')

if __name__ == '__main__':
    # have a struction to create a class
    # preprocess = Preprocess()

    # event_dir = '/Users/cainan/Desktop/Project/data/raw_data/raw_01_complex'
    event_dir = './data'

    dirs = os.listdir(event_dir) 
    for file in dirs:
        # if 'image_ts.txt' in file:
        #     frame_ts_file = self.frame_ts_dir + '/' + file
        if 'log_td.dat' in file:
            event_file = event_dir + '/' + file

    video = PSEELoader(event_file)


    print('end')






