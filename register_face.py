import cv2
import os

def register_face():
    # Create directory for known faces if it doesn't exist
    save_path = "known_faces"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    name = input("Enter the name of the person to register: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Look at the camera. Press 's' to save or 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Display instruction on frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Registering: {name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to Save", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Register Face", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save the original frame
            file_path = os.path.join(save_path, f"{name}.jpg")
            cv2.imwrite(file_path, frame)
            print(f"Face saved for {name} at {file_path}")
            break
        elif key == ord('q'):
            print("Registration cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_face()
