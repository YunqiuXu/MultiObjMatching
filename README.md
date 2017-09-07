# MultiObjMatching
+ COMP9517 Project: Multiple Objects Matching in Real-Time
+ Yunqiu Xu (z5096489), yunqiuxu1991@gmail.com

+ Dependencies
    + Python 2.7.11
    + OpenCV 3.3.0 with Contrib
    + Numpy 1.11.3
    + Numba 0.34.0
    + Pillow 3.1.2

+ There are 3 versions
    + multi_obj_match_v1.py: Both tracking and matching are performed in low resolution
    + multi_obj_match_v2.py: Tracking in low resolution and matching in original resolution
    + multi_obj_match_v3.py: Both tracking and matching are performed in original resolution
+ Run the code, then you will need to assign contours and bboxes for model frame, just follow the guide on command line
```shell
./multi_obj_match_v2.py L2_shaking.mp4 tracker_mode
```
+ Traker_mode: you can assign tracker mode for each object or just input one mode for all objects
    + 1: OpenCV KCF, default tracker
    + 2: KCF modified from https://github.com/uoip/KCFpy
    + 3: OpenCV MedianFlow, faster in high-resolution or when the object move fast
    + 4: OpenCV TLD, long term tracking to handle out-of-scene

    
