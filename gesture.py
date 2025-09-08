import cv2
import mediapipe as mp
import math
from IPython.display import clear_output

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Connect to smartphone camera (using IP Webcam)
# Replace with your phone's IP address
url = "http://192.168.1.100:8080/video"
cap = cv2.VideoCapture(url)

# Landmark indices (see MediaPipe documentation)
SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
HIP = mp_pose.PoseLandmark.LEFT_HIP.value

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Recolor image to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    
    # Recolor back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates of needed landmarks
        shoulder = [landmarks[SHOULDER].x, landmarks[SHOULDER].y]
        elbow = [landmarks[ELBOW].x, landmarks[ELBOW].y]
        wrist = [landmarks[WRIST].x, landmarks[WRIST].y]
        
        # Calculate elbow angle for flexion/extension
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # Display the angle on the screen
        cv2.putText(image, f"Elbow Angle: {int(angle)}", 
                    (100, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check if wrist is near shoulder (for pointing to origin)
        # This is a simple distance check
        distance = math.sqrt((shoulder[0]-wrist[0])**2 + (shoulder[1]-wrist[1])**2)
        if distance < 0.1: # Threshold value
            cv2.putText(image, "GOOD! Pointing to Origin", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    except:
        pass
        
    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Anatomy Motion Coach', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
