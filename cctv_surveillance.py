import face_recognition
import cv2
import numpy as np
import os
import datetime
import time
from twilio.rest import Client

# Twilio setup
TWILIO_ACCOUNT_SID = #fill with your Twilio Account SID
TWILIO_AUTH_TOKEN = #fill with your Twilio Auth Token
TWILIO_PHONE = # fill with your Twilio phone number
ADMIN_PHONE = # fill with the admin's phone number

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms_to_admin(image_path):
    message = client.messages.create(
        body=f"⚠️ Unknown person detected. Check saved image: {image_path}",
        from_=TWILIO_PHONE,
        to=ADMIN_PHONE
    )
    print(f"[INFO] SMS sent to admin: SID {message.sid}")

def save_unknown_face(frame, top, right, bottom, left):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    face_image = frame[top:bottom, left:right]
    filename = os.path.join(unknown_folder, f"unknown_{now}.jpg")
    cv2.imwrite(filename, face_image)
    print(f"[INFO] Unknown face saved at: {filename}")
    return filename

# Load known faces
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])

# Prepare unknown folder
unknown_folder = "unknown_faces"
os.makedirs(unknown_folder, exist_ok=True)

# Open camera
video_capture = cv2.VideoCapture(0)

# Save face only once every 10 minutes
last_unknown_saved = time.time() - 600  # so it triggers at start

print("[INFO] Surveillance started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Scale face coordinates back to original frame size
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        if name == "Unknown" and (time.time() - last_unknown_saved) >= 600:
            image_path = save_unknown_face(frame, top, right, bottom, left)
            send_sms_to_admin(image_path)
            last_unknown_saved = time.time()

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255) if name == "Unknown" else (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Display output
    cv2.imshow('CCTV Surveillance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
