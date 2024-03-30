import cv2
import face_recognition
import numpy as np  # Import NumPy

# Load known faces
elons_image = face_recognition.load_image_file("C:\\Users\\kunal chaudhai\\Desktop\\Face recognition trial\\images\\messi\\Elon Musk.jpg")
elons_encoding = face_recognition.face_encodings(elons_image)[0]

kunal_image = face_recognition.load_image_file("C:\\Users\\kunal chaudhai\\Desktop\\Face recognition trial\\images\\kunal\\kunal.jpg")
kunal_encoding = face_recognition.face_encodings(kunal_image)[0]

shreya_image = face_recognition.load_image_file("C:\\Users\\kunal chaudhai\\Desktop\\Face recognition trial\\images\\kunal\\Snapchat-1494126976.jpg")
shreya_encoding = face_recognition.face_encodings(shreya_image)[0]

kashish_image = face_recognition.load_image_file("C:\\Users\\kunal chaudhai\\Desktop\\Face recognition trial\\images\\kunal\\IMG_20240318_105925.jpg")  # Replace "path_to_kashish_image.jpg" with the actual path to Kashish's image
kashish_encoding = face_recognition.face_encodings(kashish_image)[0]

# Create arrays of known face encodings and corresponding names
known_face_encodings = [
    elons_encoding,
    kunal_encoding,
    shreya_encoding,
    kashish_encoding
]
known_face_names = [
    "Elon Musk",
    "Kunal",
    "Shreya",
    "Kashish"
]

# Load the custom background image
background_img = cv2.imread("C:\\Users\\kunal chaudhai\\Desktop\\Face recognition trial\\images\\messi\\baground image.jpg")
background_img = cv2.resize(background_img, (640, 480))  # Resize the image to match the video feed dimensions

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Open a video capture
video_capture = cv2.VideoCapture(0)

# Set camera properties for smoother capture
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR color (OpenCV uses) to RGB color (face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame to improve performance
    if process_this_frame:
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Apply the custom background image
    frame_with_background = cv2.addWeighted(frame, 0.8, background_img, 0.2, 0)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame was resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame_with_background, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame_with_background, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_with_background, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame_with_background)

    # Exit the program when 'esc' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

