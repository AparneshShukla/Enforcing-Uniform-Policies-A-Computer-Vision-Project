'''
This module contain Parameters used by the app 
'''

#The default sender of emails
EMAIL_SENDER = 'caiews20002@glbitm.ac.in'

# The default recipient for email
EMAIL_RECIPIENT = 'caiews20002@glbitm.ac.in'

# Mail Delay time in seconds
MAIL_DELAY = 600

# Path where client_secrets.json is stored
CLIENT_SECRETS_PATH = 'client_secrets.json'

# Path where token.json is stored

TOKEN_PATH = 'token.json'

# Path where dataset is stored
DATASET_PATH = 'dataset/train/'

# The path where dlib face encodings are stored
DLIB_FACE_ENCODING_PATH = 'dataset/dlib_face_encoding.pkl'     #'dataset/dlib_face_encoding.pkl'

# The path where face recognition report will be stored
REPORT_PATH = 'static/reports/violation-list.csv'


# face matching tolerance (distance -> less the distance, more the similarity)
FACE_MATCHING_TOLERANCE = 0.4


# face recognition model
FACE_RECOGNITION_MODEL = 'hog' #hog -> for CPU or cnn -> for GPU

# frame downsample in single_face_recognition
FRAME_DOWNSAMPLE = True

# Number of times to upsample the image looking for faces
NUMBER_OF_TIMES_TO_UPSAMPLE = 1 # for realtime keep it to 1

# default confidence threshold
CONF_ = 0.75 

# The location of the Room or Room Number.
CURRENT_LOCATION = "Room Number XYZ"




