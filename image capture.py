import cv2
import datetime

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Load Haar cascade for face detection (built-in OpenCV model)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Press 'c' to capture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam - Face Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Capture and save image with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
    elif key == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
