Real-Time Face Recognition with Voice Alerts using OpenCV and LBPH
This project is a complete face recognition system built with Python and OpenCV, capable of:

📸 Collecting face images from a webcam and saving them as a dataset.

🏷️ Training a model using the LBPH (Local Binary Pattern Histogram) algorithm.

🧑‍💻 Real-time face detection & recognition with audio alerts using Text-to-Speech (TTS).

🔊 Speaking out the names of recognized individuals using pyttsx3.

🔍 Project Structure
1. Dataset Collection
Captures 500+ grayscale face images of a user from the webcam and saves them in the datasets/ folder with naming convention:

php-template
Copy
Edit
Users.<ID>.<ImageNumber>.jpg
2. Model Training
Reads images from the datasets/ folder

Extracts face IDs and trains them using OpenCV’s LBPH face recognizer

Saves the model in Trainer.yml

3. Real-Time Recognition with Voice
Detects and recognizes faces in real-time from webcam feed

Announces the name of the recognized person using pyttsx3

Updates recognition every few seconds to avoid repeated announcements

🚀 Features
✅ Face detection using Haar Cascades
✅ LBPH face recognition
✅ Real-time webcam capture
✅ Voice alert on face recognition
✅ Efficient and modular code
✅ Easy training with just one script

📁 Project Files
File	Description
collect_dataset.py	Script to capture face images from webcam
train_model.py	Trains the LBPH face recognizer using collected data
recognize_face.py	Real-time face recognition with TTS
Trainer.yml	Saved trained model
datasets/	Directory for captured face images
haarcascade_frontalface_default.xml	Haar Cascade for face detection

🛠️ Requirements
Install dependencies:

bash
Copy
Edit
pip install opencv-python opencv-contrib-python numpy Pillow pyttsx3
🧪 How to Use
Step 1: Collect Dataset
bash
Copy
Edit
python collect_dataset.py
Enter a unique numeric ID when prompted.

Look into the camera until 500 face samples are captured.

Step 2: Train the Model
bash
Copy
Edit
python train_model.py
This reads images from the datasets/ folder and creates Trainer.yml.

Step 3: Start Real-Time Recognition
bash
Copy
Edit
python recognize_face.py
The webcam opens and recognizes known faces.

The system announces the name aloud using voice alerts.
