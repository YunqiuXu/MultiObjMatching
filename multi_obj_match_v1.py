#!/usr/bin/env python

###############################################################################
#                                                                             #
#                        Multi-Object Matching Project                        #
#                           Yunqiu Xu (z5096489)                              #
#                          yunqiuxu1991@gmail.com                             #
#                                                                             #
# How to run : sudo nice -n -19 ./multi_obj_match.py video_name tracker_mode  #
# E.G. sudo nice -n -19 ./multi_obj_match.py Video_sample_3.mp4 1 2 3         #
# For webcam, videoname = 0                                                   #
# Tracker_mode:                                                               #
#   1: KCF built-in, default                                                  #
#   2: Modified KCF: add HOG, better than built-in version, but will take     #
#   several seconds to init(numby module)                                     #
#   3: MedianFlow, faster in high-resolution or when the object move fast     #
#   4: TLD, long term tracking to handle occulusion, but very slow            #
# This is version 1: surf on small size                                       #
###############################################################################


import numpy as np
import cv2
import sys
from time import time
from PIL import Image, ImageEnhance
import kcftracker


# Function : get_tracker(tracker_mode)
# Input: tracker_mode, int
# Output: cv2.Tracker object
def get_tracker(tracker_mode):
    if tracker_mode == 1: # Built-in KCF
        return cv2.TrackerKCF_create()
    elif tracker_mode == 2: # Modified KCF
        return kcftracker.KCFTracker(True, True, True)
    elif tracker_mode == 3: # Median Flow
        return cv2.TrackerMedianFlow_create()
    else: # TLD
        return cv2.TrackerTLD_create()


# Function : get_position_obj
# Get the position for a left button double-click
# Input : mouse event
# Output : (x,y)
def get_position_obj(event,x,y,flags,param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        ix, iy = x, y
        print "Get current position ({}, {})".format(ix, iy)


# Function : get_all_objects_bboxes(img)
# Get all object contours and bboxes
# Input: image
# Output: objects, bboxes
# objects = [obj1, obj2, ...], obj = [(x1,y1), (x2,y2), ...]
# bboxes = [bbox1, bbox2, bbox3, ...], bbox = (xmin,ymin,w,h)
def get_all_objects_bboxes(img):
    print "Press 'a' to generate a new object, and save the old one."
    print "Double click left button to choose a position."
    print "Press 's' to save this position"
    print "Press 'q' to draw bboxes for all saved objects and quit"
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',get_position_obj)
    objects = []
    bboxes = []
    positions = []
    while True:
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('a'):
            if len(positions) > 0: # empty list will be ignored
                print "Object saved"
                img = cv2.polylines(img,[np.int32(positions)],True,(255,0,0),3, cv2.LINE_AA) # draw contour
                objects.append(positions)
                
                print "-----"
            print "New object created"
            positions = []
        elif k == ord('s'):
            print "Save position ({}, {})".format(ix, iy)
            cv2.circle(img, (ix,iy), 2, (255,0,0),3) # draw this point
            positions.append((ix,iy))
        elif k == ord('q'):
            print "Draw bounding boxes for {} objects".format(len(objects))
            for i in range(len(objects)):
                bbox = cv2.selectROI("image",img)
                img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0,0,255), 1)
                print "Bbox saved"
                bboxes.append(bbox)
            break
    print "Finish!"
    cv2.destroyAllWindows()
    return objects, bboxes


# Function : crop_box(img, bbox)
# Crop the img using bbox
# Input: img, bbox
# Output: subimg
def crop_bbox(img, bbox):
    h,w = img.shape[:2]
    h_min = min(max(int(bbox[1]),0), h)
    w_min = min(max(int(bbox[0]),0), w)
    h_max = min(max(int(bbox[1] + bbox[3]),0), h)
    w_max = min(max(int(bbox[0] + bbox[2]),0), w)
    # return img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2]),:]
    return img[h_min:h_max, w_min:w_max,:]


# Function : mapDown
# Map to lower resolution
# Input: bboxes list
# Output: bboxes list
def mapDown(bboxes):
    new_boxes = []
    for item in bboxes:
        item = [i / 2 for i in item]
        new_boxes.append(tuple(item))
    return new_boxes

# Remap the keypoints to original size
# input: kps, bbox on pyramid
# output: kps on original size
def remap_kps(kps, bbox):
    xmin = bbox[0]
    ymin = bbox[1]
    for kp in kps:
        kp.pt = ((kp.pt[0] + xmin) * 2, (kp.pt[1] + ymin) * 2)
    return kps


# Remap the contour to original size
# Input: contour, bbox on pyramid
# Output: contour on original size
def remap_contour(contour, bbox):
    xmin = bbox[0]
    ymin = bbox[1]
    contour[:,:,0] = (contour[:,:,0] + xmin) * 2
    contour[:,:,1] = (contour[:,:,1] + ymin) * 2
    return contour


# Function : sharpen()
# Input: img, sharpen_factor = 10
# Output: sharpened img
def sharpen(img, sharpen_factor):
    img_pil = Image.fromarray(img)
    img_enhanced = ImageEnhance.Sharpness(img_pil).enhance(sharpen_factor)
    img_cv = np.array(img_enhanced, dtype = np.uint8)
    return img_cv


# Function : match using orb + flann
# Input: 
#   img_template, img_test: the crop of small size
#   object_template: the object position for img_template in original size
#   bbox_template, bbox_test: the bbox of small size
# Output:
#   kp1, kp2: keypoints
#   good, matchesMask: refined matching pairs
#   object_test: the object position for img_test in original size
def match_orb_flann(img_template, img_test, object_template, bbox_template, bbox_test):

    # build orb
    orb = cv2.ORB_create()
    try:
        kp1, des1 = orb.detectAndCompute(img_template,None)
        kp2, des2 = orb.detectAndCompute(img_test,None)
    except:
        print "orb failed"
        return [],[],[],[],[]
    
    # build flann and get matches with recommanded parameters for orb 
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
               table_number = 12, # 6/12
               key_size = 20,     # 12/20
               multi_probe_level = 2) # 1/2
    search_params = dict(checks = 100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1,des2,k=2)
        if len(matches) == 0:
            print "No matches"
            return [],[],[],[],[]
    except:
        print "flann failed"
        return [],[],[],[],[]
    
    # refine matches: the threshold for orb should be higher to get more matches
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.7 * n.distance: 
            good.append(m)
    if len(good) == 0:
        print "No good"
        return [],[],[],[],[]

    print "The good pairs : " + str(len(good))

    # remap keypoints to original image
    kp1 = remap_kps(kp1, bbox_template)
    kp2 = remap_kps(kp2, bbox_test)
    
    # compute homography matrix
    img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    # perspective transform
    try:
        object_template = np.float32(object_template).reshape(-1,1,2)
        object_test = cv2.perspectiveTransform(object_template,M)
    except:
        print "Transform failed"
        return [],[],[],[],[]
    
    return kp1, kp2, good, matchesMask, object_test


# Function : match using surf + flann
# Input: 
#   img_template, img_test: the crop of small size
#   object_template: the object position for img_template in original size
#   bbox_template, bbox_test: the bbox of small size
# Output:
#   kp1, kp2: keypoints
#   good, matchesMask: refined matching pairs
#   object_test: the object position for img_test in original size
def match_surf_flann(img_template, img_test, object_template, bbox_template, bbox_test):

    # build surf
    surf = cv2.xfeatures2d.SURF_create()
    try:
        kp1, des1 = surf.detectAndCompute(img_template,None)
        kp2, des2 = surf.detectAndCompute(img_test,None)
    except:
        print "surf failed"
        return [],[],[],[],[]
    
    # build flann and get matches with recommanded parameters for surf
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    try:
        matches = flann.knnMatch(des1,des2,k=2)
        if len(matches) == 0:
            print "No matches"
            return [],[],[],[],[]
    except:
        print "flann failed"
        return [],[],[],[],[]
    
    # refine matches: the threshold for orb should be higher to get more matches
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.7 * n.distance: 
            good.append(m)
    if len(good) == 0:
        print "No good"
        return [],[],[],[],[]

    print "The good pairs : " + str(len(good))

    # remap keypoints to original image
    kp1 = remap_kps(kp1, bbox_template)
    kp2 = remap_kps(kp2, bbox_test)
    
    # compute homography matrix
    img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    # perspective transform
    try:
        object_template = np.float32(object_template).reshape(-1,1,2)
        object_test = cv2.perspectiveTransform(object_template,M)
    except:
        print "Transform failed"
        return [],[],[],[],[]
    
    return kp1, kp2, good, matchesMask, object_test


# Function : Draw object
def draw_obj(img, obj, color):
    img = cv2.polylines(img,[np.int32(obj)],True,color,3, cv2.LINE_AA)
    return img


# Function : Draw bounding box
def draw_bbox(img, bbox, color):
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 1)
    return img


# Function : Get largest contour
# If there is only one contour --> return the largest one
def get_largest_contour(img):
    largest_size = 0
    largest = None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 100, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        contour_size = cv2.contourArea(contour)
        if contour_size > largest_size:
            largest = contour
            largest_size = contour_size
    return largest


# Function : Build the frame of img_match: combination of img1 and img2
def build_new_img(img1, img2):
    new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    return new_img


# Function : draw matches
# rewrite cv2.drawMatches
def draw_matches(new_img, img1, kp1, kp2, matches, matchesMask, color): 
    for i in range(len(matchesMask)):
        if matchesMask[i]:
            end1 = tuple(np.round(kp1[matches[i].queryIdx].pt).astype(int))
            end2 = tuple(np.round(kp2[matches[i].trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
            cv2.line(new_img, end1, end2, color, 0)
            cv2.circle(new_img, end1, 2, color, 0)
            cv2.circle(new_img, end2, 2, color, 0)
    return new_img


if __name__ == "__main__":
    # Load parameters and global variables
    if len(sys.argv) == 2:
        video_name = sys.argv[1]
        tracker_modes = [1]
    elif len(sys.argv) > 2:
        video_name = sys.argv[1]
        tracker_modes = sys.argv[2:]
    else:
        print "Wrong arguments"
        exit(1)
    if video_name == '0': # web cam
        video_name = int(video_name)    
    ix, iy = -1, -1 # will be used in get_position_obj()
    colorList = [(255,144,30),(0,97,255),(240,32,160),(255,0,255),(255,255,0),(0,255,255),(0,0,255),(0,255,0),(255,0,0)] # BGR for common colors
    duration = 0.01 # compute FPS

    # Load the first frame
    camera = cv2.VideoCapture(video_name)
    ok, img_template=camera.read()
    if not ok:
        print('Failed to read video')
        exit(1)

    # Treat the first frame as model frame
    img_template_copy = img_template.copy()
    objects, bboxes = get_all_objects_bboxes(img_template_copy) # in original size
    
    # Draw object and bbox for model frame
    for i in range(len(objects)):
        img_template = draw_obj(img_template, objects[i], colorList[i])
        img_template = draw_bbox(img_template, bboxes[i], colorList[i])
    img_template_original = img_template.copy() # original size after drawing obj and bboxes

    # Use 1-level pyramide on model frame, then sharpen, and remap objects
    # From now on img_template means small image after pyramide
    img_template = cv2.pyrDown(img_template)
    img_template = sharpen(img_template, 10)
    print "Original bboxes : " + str(bboxes)
    bboxes = mapDown(bboxes)
    print "Map down bboxes : " + str(bboxes)
    # Now bboxes and img_template are in small size, but the objects are original

    # Init trackers for each bbox
    trackers = []
    if len(tracker_modes) == 1: # default KCF or only one tracker
        for i in range(len(bboxes)):
            tracker = get_tracker(int(tracker_modes[0]))
            tracker.init(img_template, bboxes[i])
            trackers.append(tracker)
    else: # each tracker_mode is assigned to a tracker
        for i in range(len(tracker_modes)):
            tracker = get_tracker(int(tracker_modes[i]))
            tracker.init(img_template, bboxes[i])
            trackers.append(tracker)

    # Init trace
    all_trace = []
    for i in range(len(objects)):
        all_trace.append([])

    # Start loop
    cv2.namedWindow("tracking")
    while camera.isOpened():
        # Get current frame
        ok, image=camera.read()
        if not ok:
            print "All finished!"
            break

        # Use 1-level pyramide
        img_test_original = image.copy()
        image = cv2.pyrDown(image)
        image = sharpen(image, 10)

        # Store the parameters for each object
        all_kp1 = []
        all_kp2 = []
        all_good = []
        all_matchesMask = []
        all_img_test_crop = []
        all_object_test = []
        all_bbox_test = []

        
        t0 = time()

        # Processs all objects for this frame
        for i in range(len(objects)):

            print "Object {}".format(i)

            # Get new bbox
            object_template = objects[i]
            bbox_template = bboxes[i]
            _, bbox_test = trackers[i].update(image)

            # Get crop on small image
            img_template_crop = crop_bbox(img_template, bbox_template)
            img_test_crop = crop_bbox(image, bbox_test)

            # get matches
            # kp1, kp2, good, matchesMask, object_test = match_orb_flann(img_template_crop, img_test_crop, object_template, bbox_template, bbox_test)
            kp1, kp2, good, matchesMask, object_test = match_surf_flann(img_template_crop, img_test_crop, object_template, bbox_template, bbox_test)
            # Note that now kp1, kp2, good and matchesMask are all for large image!!

            # collect matches
            all_kp1.append(kp1)
            all_kp2.append(kp2)
            all_good.append(good)
            all_matchesMask.append(matchesMask)
            all_img_test_crop.append(img_test_crop)
            all_object_test.append(object_test)
            all_bbox_test.append(bbox_test)


        # After processing all objects, draw objects and bboxes on img_test
        # only when there are more then 25 pairs can we make drawing, otherwise we draw the contour for second largest object
        for i in range(len(objects)):
            bbox_remap = [2*x for x in all_bbox_test[i]]
            img_test_original = draw_bbox(img_test_original, bbox_remap, colorList[i])
            
            # get current trace centre
            trace_center = (int(bbox_remap[0] + bbox_remap[2] / 2), int(bbox_remap[1] + bbox_remap[3] / 2))
            if trace_center != (0,0):
                all_trace[i].append(trace_center)
            for j in range(1,len(all_trace[i])):
                # img_test_original = cv2.circle(img_test_original, all_trace[i][j], 2, (255,255,255), -1)
                img_test_original = cv2.line(img_test_original, all_trace[i][j], all_trace[i][j-1], colorList[i], 2, cv2.LINE_AA)

            if len(all_good[i]) > 10:
                img_test_original = draw_obj(img_test_original, all_object_test[i], colorList[i])
            else:
                contour = get_largest_contour(all_img_test_crop[i])
                if contour != None:
                    contour = remap_contour(contour, all_bbox_test[i])
                try:
                    img_test_original = cv2.drawContours(img_test_original, [contour], 0, colorList[i], 3)
                except:
                    pass


        # After drawing bboxes and , draw matches for this frame
        img_match = build_new_img(img_template_original, img_test_original)
        for i in range(len(objects)):
            img_match = draw_matches(img_match, img_template_original, all_kp1[i], all_kp2[i], all_good[i], all_matchesMask[i], colorList[i])


        t1 = time()

        duration = 0.8*duration + 0.2*(t1-t0) #duration = t1-t0
        cv2.putText(img_match, 'FPS: '+str(1/duration)[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("tracking", img_match)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break


    camera.release()
    cv2.destroyAllWindows()
