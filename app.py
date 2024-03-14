'''A Flask application to run the YOLOv7 Uniform Detection model and face recognition on ip cam stream

Authors: Anubhav Patrick and Hamza Aziz
Edited By : Aparnesh Shukla
Date: 11/02/2024
'''
    
import validators
from flask import Flask, render_template, request, Response
import cv2


# * imported
import hubconfCustom 
from hubconfCustom import video_detection

from database_pandas import store_dataframe_in_csv

from parameters import REPORT_PATH, CONF_
    

# Initialize the Flask application
app = Flask(__name__, static_folder = 'static')

#secret key for the session
app.config['SECRET_KEY'] = 'ppe_violation_detection'

#global variables
vid_path = ''   #path for the live stream video
frames_buffer = [] #buffer to store frames from a stream
video_frames = cv2.VideoCapture(vid_path) #video capture object


def generate_raw_frames():
    '''A function to yield unprocessed frames from stored video file or ip cam stream
    
    Args:
        None
    
    Yields:
        bytes: a frame from the video file or ip cam stream
    '''
    global video_frames

    while True:            
        # Keep reading the frames from the video file or ip cam stream
        success, frame = video_frames.read()

        if success:
            # create a copy of the frame to store in the buffer
            frame_copy = frame.copy()

            #store the frame in the buffer for violation detection
            frames_buffer.append(frame_copy) 
            
            #compress the frame and store it in the memory buffer
            _, buffer = cv2.imencode('.jpg', frame) 
            #convert the buffer to bytes
            frame = buffer.tobytes() 
            #yield the frame to the browser
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n') 


def generate_processed_frames():
    '''A function to yield processed frames from stored video file or ip cam stream after violation detection
    
    Args:
        conf_ (float, optional): confidence threshold for the detection. Defaults to 0.25.
    
    Yields:
        bytes: a processed frame from the video file or ip cam stream
    '''
    #call the video_detection for violation detection which yields a list of processed frames

    fr_output = video_detection(CONF_, frames_buffer)
    #iterate through the list of processed frames
    for detection_ in fr_output:
        #The function imencode compresses the image and stores it in the memory buffer 
        _,buffer=cv2.imencode('.jpg',detection_)
        #convert the buffer to bytes
        frame=buffer.tobytes()
        #yield the processed frame to the browser
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/video_raw')
def video_raw():
    '''A function to handle the requests for the raw video stream
    
    Args:
        None

    Returns:
        Response: a response object containing the raw video stream
    '''

    return Response(generate_raw_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_processed')
def video_processed():
    '''A function to handle the requests for the processed video stream after violation detection

    Args:
        None
    
    Returns:
        Response: a response object containing the processed video stream
    '''
    return Response(generate_processed_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET", "POST"])
def index():
    '''A function to handle the requests from the web page

    Args:
        None
    
    Returns:
        render_template: the index.html page (home page)
    '''
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit_form():
    '''A function to handle the requests from the HTML form on the web page

    Args:
        None
    
    Returns:
        str: a string containing the response message
    '''
    # global variables
    global vid_path, video_frames, frames_buffer

    #if the request is a POST request made by user interaction with the HTML form
    if request.method == "POST":   
        
        # handle download request for the detections summary report                
        if "download_button" in request.form:
            print('Download Button Clicked')

            # store the detection and faces recognition summary report in a csv file
            store_dataframe_in_csv()

            # To do - add a check to see if the file exists
            return Response(
                open(REPORT_PATH, "rb").read(),
                mimetype="text/plain",
                headers={
                    "Content-Disposition": "attachment;filename=inferred_faces.csv"
                },
            )
        
        # handle alert email request
        elif 'alert_email_checkbox' in request.form:
            email_checkbox_value = request.form['alert_email_checkbox']
            if email_checkbox_value == 'false':
                hubconfCustom.is_email_allowed = False
                return "Alert email is disabled"  

            else: 
                # set flag that sending email alert is allowed
                hubconfCustom.is_email_allowed = True
                # set flag that next email can be sent when a violation is detected
                hubconfCustom.send_next_email = True
                hubconfCustom.email_recipient = request.form['alert_email_textbox']
                print(f'Alert email is enabled at {hubconfCustom.email_recipient}. Violation alert(s) with a gap of 10 minutes will be sent')
                return f"Alert email is enabled at {hubconfCustom.email_recipient}. Violation alert(s) with a gap of 10 minutes will be sent"
        
        # handle inference request for a live stream via IP camera
        elif 'live_inference_button' in request.form:
            #read ip cam url from the text box
            vid_ip_path = request.form['live_inference_textbox']
            #check if vid_ip_path is a valid url
            if validators.url(vid_ip_path):
                vid_path = vid_ip_path.strip()
                video_frames = cv2.VideoCapture(vid_path)
                #check connection to the ip cam stream
                if not video_frames.isOpened():
                    # display a flash alert message on the web page
                    return 'Error: Cannot connect to live stream',500
                
                else:
                    frames_buffer.clear()
                    print('live inference started...')
                    return 'success'
            else:
                # the url is not valid
                return 'Error: Entered URL is invalid',500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
