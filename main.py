import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)

# Initialize hand detection
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize variables
index_y = 0
previous_landmarks = None

while True:
    # Read frame from camera
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    
    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (320, 240))

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw hand landmarks and connections
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                                         landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 0, 0),
                                                                                         thickness=2,
                                                                                         circle_radius=4),
                                         connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 0, 0),
                                                                                            thickness=2))
            landmarks = hand.landmark
            
            # Check for significant change in hand landmarks
            if previous_landmarks is not None:
                landmark_changes = sum([abs(landmark.x - prev_landmark.x) + abs(landmark.y - prev_landmark.y)
                                        for landmark, prev_landmark in zip(landmarks, previous_landmarks)])
                if landmark_changes < 0.1:
                    continue
            previous_landmarks = landmarks
            
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    pyautogui.moveTo(index_x, index_y)
                    
                if id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    if abs(index_y - thumb_y) < 60:
                        pyautogui.click()
                        pyautogui.sleep(1)
                        
    # Display frame
    cv2.imshow('Virtual Mouse', frame)
    
    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()