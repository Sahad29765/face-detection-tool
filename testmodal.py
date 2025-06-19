import cv2
import pyttsx3
import threading
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speed of speech if needed

# Set up video capture and face recognizer
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# List of names corresponding to label numbers
name_list = ["", "Saad"]

# Variables to prevent repeated announcements and store previous faces
last_spoken_name = ""
last_spoken_time = 0
speak_interval = 5  # Minimum seconds between announcements for the same person
last_faces = []  # Store last detected face positions and labels

# Function to speak the detected face message asynchronously
def speak(message):
    def speak_thread():
        engine.say(message)
        engine.runAndWait()
    
    threading.Thread(target=speak_thread).start()

# Main loop
frame_count = 0
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Process every 5th frame to reduce computation
    if frame_count % 5 == 0:
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        last_faces = []  # Clear last faces list for fresh detection
        for (x, y, w, h) in faces:
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 50:
                name = name_list[serial]
                color = (0, 128, 0)
                label = name
                
                # Announce only recognized faces if enough time has passed
                if name != last_spoken_name or (time.time() - last_spoken_time) > speak_interval:
                    speak(f"{name} face detected")
                    last_spoken_name = name
                    last_spoken_time = time.time()
            else:
                name = "Unknown"
                color = (50, 50, 255)
                label = "Unknown"
                
            # Store the position and label for drawing in skipped frames
            last_faces.append((x, y, w, h, color, label))
    
    # Draw rectangles for detected faces (even on skipped frames)
    for (x, y, w, h, color, label) in last_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

# Release resources
video.release()
cv2.destroyAllWindows()
