import cv2
import glob
import os
import shutil
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

# provide the directory names here

#this directory structure is as is in the provided dataset
video_file_directory_name = 'provide/the/directory/name/here'
processed_rgb_database_name = 'provide/the/directory/name/here'

frameRate = 0.5 #//it will capture image in every 0.5 second

if os.path.isdir(processed_rgb_database_name):
    
    shutil.rmtree(processed_rgb_database_name)

os.mkdir(processed_rgb_database_name)

all_directory_list = glob.glob(video_file_directory_name+'/HZ*')

body_direction_list = ['Back', 'Front', 'Left', 'Right']

hasFrames = True

sec = 0

count=0

patient_video_directory_list = []

for directory_name in all_directory_list:
    
    if directory_name.split('/')[-1].strip('HZ').isdecimal():
        
        patient_video_directory_list.append(directory_name)

for patient_video_directory_name in patient_video_directory_list:
    
    patient_id = patient_video_directory_name.split('/')[-1]
    save_rgb_location = processed_rgb_database_name+'/'+patient_id
    
    if os.path.isdir(save_rgb_location):
    
        shutil.rmtree(save_rgb_location)
    
    os.mkdir(save_rgb_location)
    
    for body_direction_name in body_direction_list:
        
        video_file_name = patient_video_directory_name+'/'+body_direction_name+'/'+body_direction_name+'.mp4'
        
        #print (video_file_name)
        
        vidcap = cv2.VideoCapture(video_file_name)

        hasFrames = True

        sec = 0

        count=0

        while hasFrames:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
    
            if hasFrames:
                
                image = image[0:750, 0:985]
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                cv2.imwrite(save_rgb_location+'/'+patient_id+'_'+body_direction_name+'_RGB_'+str(count)+'.jpg', image)
                
