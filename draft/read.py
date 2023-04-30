# import numpy as np
# import  pandas  as pd

import numpy as np
from src.io.psee_loader import PSEELoader


if __name__ == '__main__':

    e_filedir = '/Users/cainan/Desktop/ser/Project/data/01_simple'
    e_filename = 'log_td.dat'   # [(1, 568, 322, 1) (1, 214, 253, 1) (1, 266,  57, 1) (2, 191, 113, 0) 
    # (3, 404,  24, 1) (6, 205, 342, 1) (6, 601, 436, 1) (7, 435, 381, 0)
    # (7, 501,  94, 1) (8,  78, 343, 1)]

    # file = '/Users/cainan/Desktop/ser/Project/data/01_simple/log_td.dat'
    file = e_filedir + '/' + e_filename
    video = PSEELoader(file)
    # print(video) 
        # PSEELoader:
        # -----------
        # Event Type: Event2D
        # Event Size: 8 bytes
        # Event Count: 154751804
        # Duration: 31.243012 s 
        # -----------
    # print(video.event_count())  # number of events in the file
    # print(video.total_time())  # duration of the file in mus


    # events = video.load_n_events(154751804)  # this loads the 10 next events
    events = video.load_n_events(video.event_count())
    evs = events.reshape((-1,1))
    print(type(evs[0]))
    # python numpy.array 与list类似，不同点：前者区分元素不用逗号，中间用空格,矩阵用[]代表行向量，两个行向量中间仍无逗号；

    # e= np.arange(0 ,9)
    # print(e)   # [0 1 2 3 4 5 6 7 8]
    # print(events[:5])
    # print(events['t'][154751803])  # this shows only the timestamps of events
    # for instance to count the events of positive polarity you can do :
    # print(np.sum(events['p'] > 0))
    # events = video.load_delta_t(10000)
    # my_file = loris.read_file(file)
    # print(my_file)
    # events = my_file['events']

    # for event in events:
    #     print(event)
        # (1, 568, 157, True)
        # (1, 214, 226, True)
        # (1, 266, 422, True)
        # (2, 191, 366, False)
        # print("ts:", event.t, "x:", event.x, "y:", event.y, "p:", event.p)  # 'numpy.void' object has no attribute 't'