import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip image (mirror view)
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark coordinates
            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8]   # Index fingertip
            thumb = landmarks[4]         # Thumb tip

            # Convert coordinates to screen scale
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            screen_x = screen_w / w * x
            screen_y = screen_h / h * y

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Check click (thumb and index close)
            dist = abs(index_finger.x - thumb.x) + abs(index_finger.y - thumb.y)
            if dist < 0.05:  # Threshold
                pyautogui.click()
                pyautogui.sleep(0.2)

    cv2.imshow("AI Virtual Mouse", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

