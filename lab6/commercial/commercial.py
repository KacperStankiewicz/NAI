import cv2
import dlib

def get_eye_aspect_ratio(eye):
    # Calculate the eye aspect ratio
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])

    ear = (a + b) / (2.0 * c)
    return ear

def distance(p1, p2):
    # Calculate the Euclidean distance between two points
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def play_video_with_face_and_eyes():
    # Load the dlib facial landmark predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"  # Replace with the path to the predictor file
    predictor = dlib.shape_predictor(predictor_path)

    # Load the dlib face detector
    face_detector = dlib.get_frontal_face_detector()

    # Open the webcam
    webcam = cv2.VideoCapture(0)

    # Check if the webcam was successfully opened
    if not webcam.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    # Open the video file
    video_path = "commercial.mp4"
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a window to display the video
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)

    # Flag to indicate whether video playback should be paused
    pause_video = False

    while True:
        # Read a frame from the webcam
        ret, frame = webcam.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector(gray)

        # Initialize eyes variable
        eyes = []

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Extract the coordinates of the eyes
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]

            # Draw rectangles around the eyes
            cv2.rectangle(frame, left_eye[0], left_eye[3], (0, 255, 0), 2)
            cv2.rectangle(frame, right_eye[0], right_eye[3], (0, 255, 0), 2)

            eyes.append(left_eye)
            eyes.append(right_eye)

        # Check if eyes are closed
        if all(get_eye_aspect_ratio(eye) < 0.3 for eye in eyes):
            print("Eyes closed. Pausing the video.")
            pause_video = True
        else:
            pause_video = False

        # Read a frame from the video if video playback is not paused
        if not pause_video:
            ret, video_frame = cap.read()

            # If the video has ended, break out of the loop
            if not ret:
                break

            # Resize the webcam frame to match the video frame dimensions
            webcam_frame_resized = cv2.resize(frame, (video_frame.shape[1], video_frame.shape[0]))

            # Display the frames side by side
            combined_frame = cv2.hconcat([webcam_frame_resized, video_frame])
            cv2.imshow("Video Player", combined_frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # Release the video capture objects and close the windows
    cap.release()
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video_with_face_and_eyes()
