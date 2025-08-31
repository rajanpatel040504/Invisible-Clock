import cv2        
import time       
import numpy as np 

# Start video capture from the default webcam (0 = built-in or primary camera)
cap = cv2.VideoCapture(0)


time.sleep(2)

# Capture the background by reading 30 frames
for i in range(30):
    ret, background = cap.read()

# Flip the background horizontally (mirror image)
background = np.flip(background, axis=1)

# Start reading video frames in a loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the current frame horizontally (mirror image)
    frame = np.flip(frame, axis=1)

    # Convert the frame from BGR (OpenCV default) to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper range for detecting red color (1st range)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # Define the lower and upper range for detecting red color (2nd range)
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to detect red color in both defined HSV ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine both red masks to cover all red shades
    mask = mask1 + mask2

    # Create the inverse of the red mask to get everything else
    inverse_mask = cv2.bitwise_not(mask)

    # Use the red mask to extract the background where the red cloak is
    cloak_area = cv2.bitwise_and(background, background, mask=mask)

    # Use the inverse mask to keep the parts of the current frame without the red cloak
    current_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine the two results: background where the cloak is, and current frame elsewhere
    combined = cv2.add(cloak_area, current_area)

    # Display the final output
    cv2.imshow("Invisible Cloak", combined)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
