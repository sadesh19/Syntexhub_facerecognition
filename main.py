import cv2
import face_recognition
import os
import numpy as np

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    path = "known_faces"

    if not os.path.exists(path):
        os.makedirs(path)
        return known_face_encodings, known_face_names

    for file in os.listdir(path):
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(path, file)
            image = face_recognition.load_image_file(image_path)
            
            # Get encodings (assuming one face per image)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file)[0])
                print(f"Loaded: {file}")

    return known_face_encodings, known_face_names

def run_recognition():
    known_encodings, known_names = load_known_faces()

    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")

    video_capture = cv2.VideoCapture(0)

    # Load OpenCV's pre-trained DNN face detector
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print("DNN Model files not found. Make sure 'deploy.prototxt' and 'res10_300x300...caffemodel' exist.")
        return

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    print("Starting Recognition with OpenCV DNN.")
    print("Press 'r' to register an Unknown face on the fly!")
    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Only process every other frame of video to save time
        if process_this_frame:
            # 1. Detect faces using OpenCV DNN
            (h, w) = small_frame.shape[:2]
            
            # The DNN model expects a specific blob format
            blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            face_locations = []
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections (less than 50% confidence)
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure bounding boxes are within frame dimensions
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # Ignore invalid boxes
                    if startX >= endX or startY >= endY:
                        continue

                    # Convert to face_recognition format (top, right, bottom, left)
                    face_locations.append((startY, endX, endY, startX))

            # 2. Extract encodings for those specific locations
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                name = "Unknown"

                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    
                    # Use the known face with the smallest distance to the new face
                    if True in matches:
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # On-screen instructions
        cv2.putText(frame, "Press 'r' to register Unknown face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the resulting image
        cv2.imshow('Face Recognition (OpenCV DNN)', frame)

        key = cv2.waitKey(1) & 0xFF
        # Hit 'q' on the keyboard to quit!
        if key == ord('q'):
            break
        # Hit 'r' to register an Unknown face on the fly
        elif key == ord('r'):
            if "Unknown" in face_names:
                # Find the index of the first Unknown face
                unknown_index = face_names.index("Unknown")
                unknown_encoding = face_encodings[unknown_index]
                
                # Pause the video feed by prompting in the terminal
                print("\n*** Unknown face detected! ***")
                new_name = input("Enter the name for this person (or press Enter to cancel): ").strip()
                
                if new_name:
                    # Save the image
                    file_path = os.path.join("known_faces", f"{new_name}.jpg")
                    cv2.imwrite(file_path, frame)
                    print(f"[{new_name}] saved to database at {file_path}!")
                    
                    # Update the live running database
                    known_encodings.append(unknown_encoding)
                    known_names.append(new_name)
            else:
                print("No 'Unknown' face detected on the screen to register.")

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
