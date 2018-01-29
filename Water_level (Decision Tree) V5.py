# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:06:59 2017

@author: princ
"""

import numpy as np
from collections import Counter
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import morphology,measurements
from PIL import Image, ImageStat
from scipy.misc import imresize
import pandas as pd
from sklearn import tree


def brightness(img):
    img_b = Image.fromarray(img)
    stat = ImageStat.Stat(img_b)
    bright_lvl = stat.mean[0]
    return bright_lvl

def select_biggest_object(img):
    labels, _ = measurements.label(img)
    labels = np.array(labels)
    labs= labels.flatten()
    labs_count = Counter(labs)
    del labs_count[0]
    obj_label = labs_count.most_common(1)[0][0]
    obj_size = labs_count.most_common(1)[0][1]
    obj_img = 1*(labels==obj_label)
    return obj_img, obj_size

def seperate_object(img, size = (9,5), iterations = 4):
    img_sep = morphology.binary_opening(img,np.ones(size),iterations)
    return img_sep

def merge_object(img, size = (9,5), iterations = 4):
    img_merge = morphology.binary_opening(abs(1-img),np.ones(size),iterations)
    img_merge = abs(1-img_merge)
    return img_merge

def crop_obj(box, source_img):
#    #Finding the initial obj Contours
#    obj_img = np.uint8(obj_img)
#    _,obj_contours,_ = cv2.findContours(img,1,2)
#
#    #Find contour of the tilted rectangle containing the broad board
#    for cnt in obj_contours:
#        rect = cv2.minAreaRect(cnt)
#        box = cv2.boxPoints(rect)
#        box = np.int0(box)
#        #cv2.drawContours(img,[box],0,(255,255,255),2)
#        #plt.figure()
#        #plt.imshow(img)

    #crop out the board contour
    mask = np.zeros_like(source_img)
    cv2.drawContours(mask,box,-1, 255, -1)
    crop = np.zeros_like(source_img)
    crop[mask == 255] = source_img[mask == 255]
    return crop

def select_special_object(img, img_op, thresh, thresh_op):
       #img_op = 1 -> keep the special object
       #img_op = 0 -> delete the special object
       #thresh_op = 1 -> labs inferior to thresh
       #thresh_op = 0 -> labs superior to thresh'''
    labels, _ = measurements.label(img)
    labels = np.array(labels)
    labs = labels.flatten()
    labs_count = Counter(labs)
    del labs_count[0]
#    labs_count.pop('0', None)
    special_objs_labs = [x for x in labs_count.keys() if (1-thresh_op)*thresh <= labs_count[x] <= thresh/(thresh_op+0.000001)]
#    if thresh_op == 1 and reverse == 1:
#        special_objs_labs = special_objs_labs[1:]
    special_obj = np.zeros(labels.shape)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j] in special_objs_labs:
                special_obj[i, j] = img_op 
            else:
                special_obj[i, j] = 1-img_op
    #plt.figure()
    #plt.imshow(normal_grad_open)
    return special_obj

def draw_result(grad_img_clr, OpenCV_grad):
    #draw the exact board rectangle
    out_bin = 1*(out!=0)
    exact_rect_board = np.uint8(out_bin)
    _,exact_board_contours,_ = cv2.findContours(exact_rect_board,1,2)
    #find the biggest area
    cnt = max(exact_board_contours, key = cv2.contourArea)
    #for cnt in exact_board_contours:
    rect_board = cv2.minAreaRect(cnt)
    box_board = cv2.boxPoints(rect_board)
    box_board = np.int0(box_board)
    cv2.drawContours(img,[box_board],0,(0,255,0),4)
#    plt.figure()
#    plt.imshow(img)
    
    #draw the exact contours of each graduation
    grad_big_clr = imresize(grad_img_clr,600,interp='bilinear')
    cv2.drawContours(grad_big_clr, seperate_grad_contours, -1, (0,0,255), 3)
    shape = grad_img_clr.shape
    grad_img_clr = imresize(grad_big_clr,shape,interp='bilinear')
    img[y:y+h, x:x+w] = grad_img_clr
    #cv2.imshow('img', grad_img_clr)
    #plt.figure()
    #plt.imshow(grad_img_clr)
    
    #draw the exact rectangle containing the graduation
    for cnt in grad_contours:
        rect_grad = cv2.minAreaRect(cnt)
        box_grad = cv2.boxPoints(rect_grad)
        box_grad = np.int0(box_grad)
        cv2.drawContours(img,[box_grad],0,(255,0,0),2)
    
    #cv2.drawContours(img, grad_contours, -1, (255,0,0), 2)
    #grad_display = 1*(np.logical_and(grad_crop<128, grad_crop>3))
    #rect_temp3 = np.uint8(grad_display)
    #_,ex_contours1,_ = cv2.findContours(rect_temp3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, ex_contours1, -1, (0,0,255), 1)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    #print the number of graduations
    txt = str(OpenCV_grad)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,txt,(x,y), font, 0.7,(0,0,0),2,cv2.LINE_AA)
    
    #show final result
    cv2.imshow('img',img)
    cv2.imwrite('IMG_(2017-10-27)_(00am07)_Result.jpg',img)
    plt.figure()
    plt.imshow(img)


# Read the picture in gray scale
img = cv2.imread('IMG_(2017-10-27)_(00am07).jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)

# Increasing contrast + denoising
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
dns = cv2.fastNlMeansDenoising(cl1, None, 65, 5, 21)
#cv2.imshow('cla', cl1)
#cv2.imshow('dns', dns)

# Selecting the appropriate binarizatioin threshold
'''the 'whiter' the wanted items to be isolated, the 'higher' the threshold should be (case of flowers)
   the 'darker' the wanted items to be isolated, the 'lower' the threshold should be (case of houses)'''
   
if 125 <= brightness(cl1) < 135:
    t1 = 0.55
    bin_input = dns
elif 115 <= brightness(cl1) < 125:
    t1 = 0.5
    bin_input = dns
elif 105 <= brightness(cl1) < 115:
    t1 = 0.45
    bin_input = cl1
elif 95 <= brightness(cl1) < 105:
    t1 = 0.4
    bin_input = cl1
elif 85 <= brightness(cl1) < 95:
    t1 = 0.35
    bin_input = cl1
elif 75 <= brightness(cl1) < 85:
    t1 = 0.3
    bin_input = cl1
elif 65 <= brightness(cl1) < 75:
    t1 = 0.25
    bin_input = cl1
    
# Binarization of the original image
bi = 1*(bin_input < t1*bin_input.max())
bi = abs(1 - bi)
#plt.figure()
#plt.imshow(bi)

# Refine the result by seperating and selecting the biggest object (board)
step = 4
bi_open = seperate_object(bi, size = (9,5), iterations = step)

while True:
    bi_open = seperate_object(bi, size = (9,5), iterations = step)
    bi2, bi2_size = select_biggest_object(bi_open)
    if bi2_size < 100 * gray.shape[0]:
        step -= 1
        print(step)
    else:
        break
   
#plt.figure()
#plt.imshow(bi_open)
plt.figure
plt.imshow(bi2)

# Refine the result by enlarging the object
bi2_open = merge_object(bi2, size = (9,5), iterations = 4)
#plt.figure()
#plt.imshow(bi2_open)

#Finding the initial board Contours
broad_rect_board = np.uint8(bi2_open)
_,broad_board_contours,_ = cv2.findContours(broad_rect_board,1,2)

#Find contour of the tilted rectangle containing the broad board
for cnt in broad_board_contours:
    rect_broad = cv2.minAreaRect(cnt)
    box_broad = cv2.boxPoints(rect_broad)
    box_broad = [np.int0(box_broad)]
#    cv2.drawContours(img,[box],0,(255,255,255),2)
#plt.figure()
#plt.imshow(img)
    
# Crop the board
board_crop = crop_obj(box_broad, gray)

# Increase the contrast of the board
cl1_crop = clahe.apply(board_crop)
#cv2.imshow('cla', cl1_crop)

# Selecting the appropriate binarizatioin threshold using Decision Tree
df = pd.read_csv('Brightness data.csv')
features = list(df.columns[1:5])

y = df['t2']
X = df[features]

cls = tree.DecisionTreeRegressor()
cls.fit(X,y)

t2 = cls.predict([[brightness(gray),brightness(cl1),brightness(board_crop),brightness(cl1_crop)]])

#if brightness(gray) < 95:
#    if brightness(cl1_crop) < 30:
#        t2 = 0.2
#    else:
#        t2 = 0.3
#else:
#    t2 = 0.45

# Binarization of the board crop 
bi_crop = 1*(board_crop < t2*board_crop.max())
bi_crop = abs(1 - bi_crop)
plt.figure()
plt.imshow(bi_crop)

# Refine the result by seperating the object
bi_crop_open = seperate_object(bi_crop, size = (9,5), iterations = 2)
#plt.figure()
#plt.imshow(bi_crop_open)

# Select the biggest object (exact board)
board, _ = select_biggest_object(bi_crop_open)
#plt.figure()
#plt.imshow(board)

# Refine the result by enlarging the object
board_open = merge_object(board, size = (9,5), iterations = 4)
#plt.figure()
#plt.imshow(board_open)

# Reverse the board image
board_open_reverse = abs(1-board_open)
#plt.figure()
#plt.imshow(board_open_reverse)

# Refine the result by enlarging the object
board_open2 = merge_object(board_open, size = (9,5), iterations = 25)
#plt.figure()
#plt.imshow(board_open2)

## Select the graduations image
#test = board_open_reverse * bi_crop
out = board_crop * board_open2 #this is theoretically, the exact image of the board
out1 = out * board_open_reverse
#plt.figure()
#plt.imshow(out)
#plt.figure()
#plt.imshow(out1)

# Select the biggest object (broad graduations)
out2, _ = select_biggest_object(out1)
#test2= out2 * bi_crop
#plt.figure()
#plt.imshow(test2)
#plt.figure()
#plt.imshow(out2)

# Take away any possible noise due to flash light or shadows
''' revize the funtion when we use abs value of an img'''
#out2 = select_special_object(out2, img_op = 0, thresh = 70, thresh_op = 0)
out2_1 = select_special_object(abs(1-out2), img_op = 0, thresh = 70, thresh_op = 0)
#plt.figure()
#plt.imshow(out2_1)

# Refine the result by seperating the object
out2_open = seperate_object(out2_1, size = (9,5), iterations = 4)
#plt.figure()
#plt.imshow(out2_open)

# Refine the result by enlarging the object
out2_open2 = merge_object(out2_open, size = (9,5), iterations = 8)
#plt.figure()
#plt.imshow(out2_open2)

# Select the biggest object (exact graduations)
out3, _ = select_biggest_object(out2_open2)

# Finding the exact graduation Contours
grad_cnt = np.uint8(out3)
_,grad_contours,_ = cv2.findContours(grad_cnt,1,2)

# Find contour of the straight rectangle containing the graduation
'''was working with retr_ccomp'''
_,rect_contours,_ = cv2.findContours(grad_cnt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Crop out the graduation contour
grad_crop2 = crop_obj(grad_contours, bi_crop)
grad_crop = crop_obj(grad_contours, cl1)

# Isolate the rectangle containing the graduations
'''WE CAN MAKE THIS A 'BIRD EYE VIEW' IMAGE'''
for cnt in rect_contours:
    x,y,w,h = cv2.boundingRect(cnt)
if h > 550:
    h = 520
    if w > 200:
        x += int(w * 0.15)
        w -= int(w * 0.3) 
    elif  150 < w < 200:
        x += int(w * 0.1)
        w -= int(w * 0.2)
        
grad_img = grad_crop[y:y+h, x:x+w]
grad_img1 = grad_crop2[y:y+h, x:x+w]
grad_img_clr = img[y:y+h, x:x+w]
#plt.figure()
#plt.axis('off')
#plt.imshow(grad_img)
#plt.figure()
#plt.imshow(grad_img1)

#increasing contrast
cl1_grad = clahe.apply(grad_img)
#plt.figure()
#plt.imshow(cl1_grad)

if brightness(cl1_grad) < 50:
    bin_thresh = 90
elif 50 <= brightness(cl1_grad) <= 100:
    bin_thresh = 128
else:
    bin_thresh = 135
'''add a condition on cl1_grad brightness to select the binarization threshold'''

# Binarization 
grad_img_bin = 1*(np.logical_and(cl1_grad<bin_thresh, cl1_grad>3)) 
#plt.figure()
#plt.gray()
#plt.imshow(grad_img_bin)

#Deleting white noise
grad_img_bin_open = select_special_object(grad_img_bin, img_op = 1, thresh = 90, thresh_op = 0)
#plt.figure()
#plt.imshow(grad_img_bin_open)

#Deleting black noise
grad_img_bin_open_1 = select_special_object(abs(1-grad_img_bin_open), img_op = 0, thresh = 70, thresh_op = 0)
#plt.figure()
#plt.imshow(grad_img_bin_open_1)

'''add a condition about abnormal graduations'''

# check for abnormal graduations
labs, _ = measurements.label(grad_img_bin_open_1)
labs = np.array(labs)
lab= labs.flatten()
count_grad = Counter(lab)
del count_grad[0]
if bin_thresh == 100:
    grad_sizes = [x for x in count_grad.values() if 120 < x < 500]
else:
    grad_sizes = [x for x in count_grad.values() if 120 < x < 700]
grad_size = np.mean(grad_sizes) + 0.75*(np.max(grad_sizes) - np.min(grad_sizes)) 
for _, size in count_grad.items():
    if size > grad_size:
        abnormal = 1
        break
    else: 
        abnormal = 0
#if np.mean(grad_sizes) > 400:
#    grad_img_bin_open1 = np.uint8(grad_img_bin_open_1)
#    grad_img_bin_open_1 = cv2.erode(grad_img_bin_open1,None)
#
#plt.figure()
#plt.imshow(grad_img_bin_open_1)

if abnormal == 1:
    
    #grad_img1 = np.uint8(grad_img1)
#    dilate = cv2.dilate(grad_img1,None)
#    plt.figure()
#    plt.imshow(dilate)

#    for i in [1, 2, 3, 4]:
#        grad_img2 = merge_object(grad_dilate, size = (9,5), iterations = i)
#        if np.all(grad_img2 == 0):
#            grad_img2 = merge_object(grad_dilate, size = (9,5), iterations = i-1)
#            print(i)
#        break
    grad_img2 = merge_object(grad_img1, size = (9,5), iterations = 4)
    plt.figure()
    plt.gray()
    plt.imshow(grad_img2)

    # Graduations corrected
    grad = grad_img_bin_open_1 * grad_img2
    plt.figure()
    plt.imshow(grad)
#    
    #grad_1 = select_special_object(abs(1-grad), img_op = 0, thresh = 70, thresh_op = 0)
#    plt.figure()
#    plt.imshow(grad_1)
    
#    grad = np.uint8(grad)
#    grad_dilate = cv2.dilate(grad,None)
#    plt.figure()
#    plt.imshow(grad_dilate)
    
#    for i in [1, 2, 3, 4]:
#        grad2 = merge_object(dilate, size = (9,5), iterations = i)
#        if np.all(grad2 == 0):
#            grad2 = merge_object(dilate, size = (9,5), iterations = i-1)
#        break
    grad2 = merge_object(grad, size = (9,5), iterations = 4)
#    plt.figure()
#    plt.imshow(grad2)
    
    grad3, _ = select_biggest_object(grad2)
#    plt.figure()
#    plt.imshow(grad3)
    
    grad3_1 = seperate_object(grad3, size = (7,5), iterations = 1)
#    plt.figure()
#    plt.imshow(grad3_1)
    
    grad4 = grad3_1 * grad
    grad4_reverse = abs(1-grad4)
#    plt.figure()
#    plt.imshow(grad4)
    
    grad_final = select_special_object(grad4, img_op = 1, thresh = 20, thresh_op = 0)
    plt.figure()
    plt.gray()
    plt.imshow(grad_final)
    
    # Select normal graduations
    labs, _ = measurements.label(grad_final)
    labs = np.array(labs)
    lab= labs.flatten()
    count_grad = Counter(lab)
    del count_grad[0]
    upper_grads = list(count_grad.items())[:-4]
    lower_grads = list(count_grad.items())[-4:]
    lower_grad_size = np.mean(list(count_grad.values())) - np.std(list(count_grad.values())) #[-4:])
    obj_list = [np.zeros(labs.shape) for i in range(4)] 
    #obj_img = 1*(labs==lower_grads[2][0])
    #plt.figure()
    #plt.imshow(obj_img)
    #r = merge_object(obj_img, size = (2,1), iterations = 3)
    #plt.figure()
    #plt.imshow(r)
    for i in range(4):
        obj_img = 1*(labs==lower_grads[i][0])
        if lower_grads[i][1] > lower_grad_size:
            obj_list[i] = seperate_object(obj_img, size = (2,1), iterations = 3)
        else:
            obj_list[i] = merge_object(obj_img, size = (2,1), iterations = 3)
    upper_grads_img = np.zeros(labs.shape)
    for i in range(len(upper_grads)):
        upper_grads_img += 1*(labs==upper_grads[i][0])
    #plt.figure()
    #plt.imshow(upper_grads_img)   
    lower_grads_img = sum(obj_list)
    #plt.figure()
    #plt.imshow(lower_grads_img)
    
    final_grad_img = lower_grads_img + upper_grads_img
    grad_final_open = select_special_object(final_grad_img, img_op = 1, thresh = 15, thresh_op = 0)
    plt.figure()
    plt.gray()
    plt.imshow(grad_final_open)
    
else:
    grad_final_open = grad_img_bin_open_1
    grad_final_reverse = abs(1-grad_final_open)
    plt.figure()
    plt.gray()
    plt.imshow(grad_final_open)
    
#if bin_thresh == 100:
#    grad_sizes = [x for x in count_grad.values() if 120 < x < 500]
#else:
#    grad_sizes = [x for x in count_grad.values() if 120 < x < 700]
#grad_size = np.mean(grad_sizes) #+ 0.75*(np.max(grad_sizes) - np.min(grad_sizes)) 
#normal_grad = select_special_object(grad4, img_op = 1, thresh = int(grad_size), thresh_op = 1)
#plt.figure()
#plt.imshow(normal_grad)
#
## Select Abnormal graduations
#abnormal_grad = select_special_object(grad4, img_op = 1, thresh = int(grad_size), thresh_op = 0)
##plt.figure()
##plt.imshow(abnormal_grad)
#
## Refine the abnormals by seperating the object
#abnormal_grad_open = seperate_object(abnormal_grad, size = (3,1), iterations = 2)
##plt.figure()
##plt.imshow(abnormal_grad_open)
#
## Select the most appropriate abnormal graduations
#abnormal_grad_open1 = select_special_object(abnormal_grad_open, img_op = 1, thresh = int(grad_size), thresh_op = 1)
##plt.figure()
##plt.imshow(abnormal_grad_open1)
#
## Adding abnormal and normal graduations together
#final_img_grad = abnormal_grad_open1 + normal_grad
#final_img_grad_reverse = abs(1-final_img_grad)
##plt.figure()
##plt.imshow(final_img_grad)

#Deleting black noise
#grad_img_bin_open = select_special_object(grad4_reverse, img_op = 0, thresh = 70, thresh_op = 0)
#plt.figure()
#plt.imshow(grad_img_bin_open)

#grad_img_bin_open = np.uint8(grad_img_bin_open)
#dilate = cv2.dilate(grad_img_bin_open,None)
#plt.figure()
#plt.gray()
#plt.imshow(dilate)

#if abnormal == 1:
#    min_grad = 30
#else:
#    min_grad = 70
#    
##Deleting white noise
#grad_img_bin_open1 = select_special_object(grad_img_bin_open, img_op = 1, thresh = min_grad, thresh_op = 0)
##plt.figure()
#plt.gray()
#plt.imshow(grad_img_bin_open1)

## Select normal graduations
#labs, _ = measurements.label(grad_img_bin_open1)
#labs = np.array(labs)
#lab= labs.flatten()
#count_grad = Counter(lab)
#if bin_thresh == 100:
#    grad_sizes = [x for x in count_grad.values() if 120 < x < 500]
#else:
#    grad_sizes = [x for x in count_grad.values() if 120 < x < 700]
#grad_size = np.mean(grad_sizes)  
#
#if grad_size > 400:
#    final = cv2.erode(grad_img_bin_open1,None)
#    #final = seperate_object(grad_img_bin_open1, size = (2,1), iterations = 2)
#    final = select_special_object(final, img_op = 1, thresh = 30, thresh_op = 0)
#else:
#    final = grad_img_bin_open1

#plt.figure()
#plt.gray()
#plt.imshow(final)

# Count graduations using Scipy
_, Scipy_grad = measurements.label(grad_final_open)

# Count graduations using OpenCV
grad_big = imresize(grad_final_open,600,interp='bilinear')

graduations = np.uint8(grad_big)
_,seperate_grad_contours,_ = cv2.findContours(graduations,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

grad_big_clr = imresize(grad_img_clr,600,interp='bilinear')
cv2.drawContours(grad_big_clr, seperate_grad_contours, -1, (0,0,255), 3)
#plt.figure()
#plt.imshow(grad_big_clr)

OpenCV_grad = len(seperate_grad_contours)

# Print the results on the original image
if OpenCV_grad == Scipy_grad:
    draw_result(grad_img_clr, OpenCV_grad)
#    if 1 in abnormal_grad:
#        print('Abnormal Graduations Detected')
#    else:
#        print('No Abnormal Graduations Detected')
else:
    raise ValueError('There was a problem while counting the graduations')
#cv2.imwrite('Tilted_final.png',img)

#Add the succesfull case to the input of the Decision Tree
#if df.shape[0] < 1000:
#    df.loc[df.shape[0]+1] = ["###"+str(df.shape[0]+1),str(brightness(gray)),str(brightness(cl1)),str(brightness(board_crop)),str(brightness(cl1_crop)),str(t2[0])]
#    df.to_csv("Brightness data.csv", encoding='utf-8', index=False)