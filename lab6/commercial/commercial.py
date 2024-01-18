"""
Projekt NAI 6: platforma do wyświetlania reklam

Autorzy:
Rutkowski, Marcin       s12497
Stakniewicz, Kacper     s22619

Przygotowanie środowiska:

wypakować shape_predictor_68_face_landmarks.dat.bz2

pip install opencv-python
python -m pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
"""

import cv2
import dlib

"""
    Obliczanie EAR
"""


def get_eye_aspect_ratio(eye):
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])

    ear = (a + b) / (2.0 * c)
    return ear


"""
    Oblicznie odległości euklidesowej pomiędzy dwoma punktami
"""


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def play_video_with_face_and_eyes():
    """ wczytanie predykatora z biblioteki dlib """
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    """ wczytanie detektora twarzy """
    face_detector = dlib.get_frontal_face_detector()

    """ inicjalizacja przechwytywania obrazu z kamery """
    webcam = cv2.VideoCapture(0)

    """ sprawdzenie czy inicjalizacja się powiodła """
    if not webcam.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    """ otwarcie pliku z reklamą i sprawdzenie czy zakończyło się to sukcesem """
    video_path = "commercial.mp4"
    commercial_player = cv2.VideoCapture(video_path)
    if not commercial_player.isOpened():
        print("Error: Couldn't open the video file.")
        return

    """ pobranie klatkarzu reklamy """
    fps = commercial_player.get(cv2.CAP_PROP_FPS)

    """ inicjalizcja okienka do wyświetlenia reklamy"""
    cv2.namedWindow("Commercial Player", cv2.WINDOW_NORMAL)

    """ flaga wskazująca czy reklama powinna być odtwarzana czy nie """
    pause_video = False

    while True:
        """ pobranie klatki z kamerki """
        ret, frame = webcam.read()

        """ konwersja klatki na odcienie szarości """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        """ detekcja twarzy """
        faces = face_detector(gray)

        eyes = []

        for face in faces:
            """ pobranie znaczników dla twarzy """
            landmarks = predictor(gray, face)

            """ pobranie koordynatów dla oczu """
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]

            """ Rysowanie prostokątów dookoła oczu """
            cv2.rectangle(frame, left_eye[0], left_eye[3], (0, 255, 0), 2)
            cv2.rectangle(frame, right_eye[0], right_eye[3], (0, 255, 0), 2)

            eyes.append(left_eye)
            eyes.append(right_eye)

        """ sprawdzenie czy oczy są zamknięte """
        if all(get_eye_aspect_ratio(eye) < 0.3 for eye in eyes):
            print("Eyes closed. Pausing the video.")
            pause_video = True
        else:
            pause_video = False

        if not pause_video:
            """ pobranie klatki reklamy"""
            ret, video_frame = commercial_player.read()

            """ Jeśli reklama się skończy, wyjdź z pętli """
            if not ret:
                break

            """ zmiana wielkości obrazu z kamerki tak, by pasował do rozmiaru okienka z reklamą """
            webcam_frame_resized = cv2.resize(frame, (video_frame.shape[1], video_frame.shape[0]))

            """ złączenie i wyświetlenie obrazu z kamerki i reklamy obok siebie w jednym oknie """
            combined_frame = cv2.hconcat([webcam_frame_resized, video_frame])
            cv2.imshow("Video Player", combined_frame)

        """ jeśli użytkownik wciśnie 'q' zakończ program """
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    commercial_player.release()
    webcam.release()
    cv2.destroyAllWindows()


play_video_with_face_and_eyes()
