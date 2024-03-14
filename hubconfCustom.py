'''
A modified version of hubconf.py  

Modifications:
1. Added a function to detect PPE violation in a video file or video stream
2. Added a function to send email alert with attached image

Modifications made by Anubhav Patrick
Date: 04/02/2023
'''
import cv2
import time
import torch
import numpy as np
import pickle
import face_recognition

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, letterbox
from utils.plots import plot_one_box
from utils.torch_utils import select_device

from send_mail import prepare_and_send_email
from database_pandas import store_inferred_face_in_dataframe
from threading import Thread
 

from parameters import ( 
    FRAME_DOWNSAMPLE,
    NUMBER_OF_TIMES_TO_UPSAMPLE,
    FACE_RECOGNITION_MODEL,
    FACE_MATCHING_TOLERANCE,
    DLIB_FACE_ENCODING_PATH,
    EMAIL_SENDER,
    EMAIL_RECIPIENT,
    REPORT_PATH,
    MAIL_DELAY
    )


#Global Variabless
is_email_allowed = False #when user checks the email checkbox, this variable will be set to True
send_next_email = True #We have to wait for 10 minutes before sending another email
# NEXT TWO STATEMENTS NEED TO BE CHANGED TO MATCH YOUR SETUP!!!
#set the default email sender and recipient 
email_sender = EMAIL_SENDER
email_recipient = EMAIL_RECIPIENT
classes_to_filter = None  #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]


# ----------------VERY IMPORTANT - CONFIGURATION PARAMETERS----------------
# a dictionary to store options for inference
opt  = {
    "weights": "best.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/custom_data.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None
}


# # Violation Alert Generator Module


def violation_alert_generator(file_path, subject='Uniform Violation Detected', message_text='No-Tie, Non-Uniformed students detected List is attached below..'):
    '''This function will send an email with attached alert image and then wait for 10 minutes before sending another email
    
    Parameters:
      im0 (numpy.ndarray): The image to be attached in the email
      subject (str): The subject of the email
      message_text (str): The message text of the email

    Returns:
      None
    '''
    global send_next_email, email_recipient
    send_next_email = False #set flag to False so that another email is not sent
    print('Sending email alert to ', email_recipient )
    prepare_and_send_email(email_sender, email_recipient , subject, message_text, file_path)
    # wait for time given in MAIL_DELAY before sending another email
    time.sleep(MAIL_DELAY)
    send_next_email = True


# # Face Recognition Module
    
# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(DLIB_FACE_ENCODING_PATH,"rb").read())

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = data["encodings"]
known_face_names = data["names"]

#initialize the array variable to hold all face locations, encodings and names 
all_face_locations = []
all_face_encodings = []
all_face_names = []
all_processed_frames = []

is_email_allowed = False #when user checks the email checkbox, this variable will be set to True
send_next_email = True #We have to wait for 10 minutes before sending  = False


def single_frame_face_recognition(store_name, label, frame, frame_downsample, number_of_times_to_upsample, model, face_matching_tolerance):
    '''Single frame face recognition function
    
    Arguments:
        frame {numpy array} -- frame to be processed
        frame_downsample {bool} -- whether to downsample the frame or not
        number_of_times_to_upsample
        model -- face detection model
        face_matching_tolerance -- tolerance for face matching
    
    Returns:
        a processed frame
    '''

    if frame_downsample:
        #resize the current frame to 1/4 size to proces faster
        #current_frame_small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        
        #resize the frame to 1024x576 to display the video if frame is too big
        # frame.shape[1] is width and frame.shape[0] is height
        if frame.shape[1] > 1024 or frame.shape[0] > 576:
            current_frame_small = cv2.resize(frame,(1024,576))
        else:
            current_frame_small = frame
    else:
        #consider the frame as it is
        current_frame_small = frame

    #detect all faces in the image
    #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample,model)
        
    #* detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

    #looping through the face locations and the face embeddings
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        if frame_downsample:
            #change the position magnitude to fit the actual size video frame
            '''top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4'''
            pass
        
        #* find all the matches and get the list of matches
        all_matches = face_recognition.face_distance(known_face_encodings, current_face_encoding)
        # Find the best match (smallest distance to a known face)
        best_match_index = np.argmin(all_matches)
        # If the best match is within tolerance, use the name of the known face
        if all_matches[best_match_index] <= face_matching_tolerance:
            name_of_person = known_face_names[best_match_index]
            #save the name of the person in the dataframe
            if store_name == True:
             store_inferred_face_in_dataframe(name_of_person, all_matches[best_match_index],label )
        else:
            name_of_person = 'Unknown face'


        # For known face use green color and for unknown face use red color
        if name_of_person == 'Unknown face':
            color = (0,0,255) #Red
        else:
            color = (0,255,0) #Green
        
        #draw rectangle around the face    
        cv2.rectangle(current_frame_small,(left_pos,top_pos),(right_pos,bottom_pos),color,5)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame_small, name_of_person, (left_pos,bottom_pos), font, 1, (255,255,255),2)

    # * create a thread for sending email
    if is_email_allowed == True and send_next_email == True:
        t = Thread(target=violation_alert_generator, args=(REPORT_PATH, 'Uniform Violation Detected','No-Tie, Non-Uniformed students detected' ))
        t.start()  
    
    yield current_frame_small


# # Detection Module


def video_detection(conf_=0.25, frames_buffer=[]):
  '''This function will detect violations in a video file or a live stream 

  Parameters:
    conf_ (float): Confidence threshold for inference
    frames_buffer (list): A list of frames to be processed

  Returns:
    None
  '''    
  # Declare global variables to be used in this function
  global send_next_email
  global is_email_allowed
  store_name = False
  label = ''
  

  violation_frames = 0 # Number of frames with violation
 
  #pop first frame from frames_buffer to get the first frame
  # We encountered a bug in which the first frame was not getting properly processed so we are popping it out
  while True:
    if len(frames_buffer) > 0:
      _ = frames_buffer.pop(0)
      break

  # empty the GPU cache to free up memory for inference
  torch.cuda.empty_cache()
  # Initializing model and setting it for inference
  # no_grad() is used to speed up inference by disabling gradient calculation
  with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']    
    device = select_device(opt['device'])
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    # if device is not cpu i.e, it is gpu, convert model to half precision
    half = device.type != 'cpu'
    if half:
      model.half() # convert model to FP16

    # find names of classes in the model
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # Run inference one time to initialize the model on a tensor of zeros
    if device.type != 'cpu':
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # classes to filter out from the detections
    # We will not be filtering out any class and use all four , no-helmet and no-jacket
    classes = None
    if opt['classes']:
      classes = []
      for class_name in opt['classes']:
        classes.append(opt['classes'].index(class_name))

    try:
      # Continuously run inference on the frames in the buffer
      while True:
        # if the frames_buffer is not empty, pop the first frame from the buffer
        if len(frames_buffer) > 0:
          #pop first frame from frames_buffer 
          img0 = frames_buffer.pop(0)
          # if the popped frame is None, continue to the next iteration
          if img0 is None:
            continue
          #clear the buffer if it has more than 10 frames to avoid memory overflow
          if len(frames_buffer) >= 10:
            frames_buffer.clear() 
        else:
          # buffer is empty, nothing to do
          continue

        img = letterbox(img0, imgsz, stride=stride)[0] # resize and pad image
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img) # convert to contiguous array
        img = torch.from_numpy(img).to(device) # place the image on the device (cpu or gpu)
        img = img.half() if half else img.float()  # convert to FP16 if device is gpu
        img /= 255.0  # Normalize to [0, 1] range
        
        # add a dimension to the image if it is a single image
        if img.ndimension() == 3:
          img = img.unsqueeze(0)

        # Do the Inference (Prediction)
        pred = model(img, augment= False)[0]

        # Do the non-maximum suppression to remove the redundant bounding boxes
        total_detections = 0
        pred = non_max_suppression(pred, conf_, opt['iou-thres'], classes= classes, agnostic= False)
        
        # Process all the predictions and draw the bounding boxes
        for _, det in enumerate(pred):
          # classwise_summary = ''

          # if there is a detection
          if len(det):
            # Rescale boxes from img_size (predicted image) to im0 (actual image) size and round the coordinates
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Consider that the frame is safe
            unsafe = False # Flag to indicate if the frame is unsafe

            # Write the detections to the frame
            for c in det[:, -1].unique():
              n = (det[:, -1] == c).sum()  # detections per class
              total_detections += int(n)
              c = int(c)

              #we need to make sure at there is violation in atleast 5 continous frames
              # Firstly, Check if the frame is unsafe
              # c == 0 means No-Tie, c == 1 means Non-Unifromed
              # c == 2 means Unifromed
              if unsafe == False and (c == 0 or c == 1) and n > 0:
                unsafe = True
              # * Extra Feature             
              # # add the number of detections per class to the string
              # classwise_summary += f"{n} {names[c]}{'s' * (n > 1)}, "  # add to string

#!! chkpnt1...
              
            #code to send email on five continous violations
            if unsafe == True:
              violation_frames += 1
              if violation_frames >= 5:
                # reset the violation_frames since violation is detected
                violation_frames = 0
                # set the flag to store the mail
                store_name = True
                  
            elif unsafe == False:
              store_name = False
              # reset the number of violation_frames if current frame is safe
              violation_frames = 0

            # # Store the detections summary in a string
            # #get current time in hh:mm:ss format
            # current_time = time.strftime("%H:%M:%S", time.localtime())
            # detections_summary += f"\n {current_time}\n Total Detections: {total_detections}\n Detections per class: {classwise_summary}\n###########\n"
            
            # Plot the bounding boxes on the frame
            for *xyxy, conf, cls in det:
              label = f'{names[int(cls)]} {conf:.2f}'
              if label.startswith('safe'):
                color = (0,255,0) #Green in BGR
              else:
                color = (0,0,255) #Red in BGR

              plot_one_box(xyxy, img0, label=label, color=color, line_thickness=3)   

        yield from single_frame_face_recognition(store_name, label, frame= img0,frame_downsample=FRAME_DOWNSAMPLE, number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE, model=FACE_RECOGNITION_MODEL, face_matching_tolerance=FACE_MATCHING_TOLERANCE )

    except Exception as e:
      print(e)
    
