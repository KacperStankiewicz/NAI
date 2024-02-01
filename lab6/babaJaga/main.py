"""
Projekt NAI 6: Baba Jaga Patrzy

Autorzy:
Rutkowski, Marcin       s12497
Stakniewicz, Kacper     s22619

Przygotowanie środowiska:

pip install opencv-python

Instrukcja:
Program przechwytuje obraz z kamery i, korzystając z dostępnego w bibliotece OpenCV klasyfikatora HaarCascade,
wykrywa twarze widoczne na ekranie.
Ponadto program wykrywa pozycje twarzy i porównując do poprzedzających współrzędnych określa
czy twarz zmieniła położenie (wykrywanie ruchu).
Każda twarz oznaczona jest celownikiem - zielonym, jeśli ruch nie został wykryty lub czerwonym, jeśli twarz
poruszyła się
"""

import cv2
import copy

""" Wymiary znaczników """
CIRCLE_DIAMETER_DIVIDER = 2
CIRCLE_THICKNESS = 2
VIEWFINDER_LENGTH = 20
VIEWFINDER_THICKNESS = 2

""" 
Strojenie Face Cascade

SCALE FACTOR - parametr określający, jak bardzo obraz jest redukowany na każdym etapie piramidy obrazu. 
Wartość większa niż 1 oznacza, że algorytm używa mniejszej skali do wykrywania obiektów

MIN_NEIGHBOURS - parametr określający, ile sąsiadów musi mieć każdy prostokąt kandydujący, aby go zachować. 
Jest to sposób na filtrację fałszywych pozytywów. 
Jeśli parametr jest ustawiony na wyższą wartość, algorytm będzie wymagał większej liczby potwierdzeń, 
co może pomóc w eliminacji fałszywych wykryć, ale może także pominąć niektóre prawidłowe wykrycia.
"""
SCALE_FACTOR = 1.5
MIN_NEIGHBOURS = 6

""" Strojenie wykrywania ruchu"""
PIXEL_THRESHOLD = 10

""" Klasyfikatory haarcascade_frontalface """
HAARCASCADE_DEFAULT_CLASIFIER = 'haarcascade_frontalface_default.xml'
HAARCASCADE_ALT_CLASIFIER = 'haarcascade_frontalface_alt.xml'
HAARCASCADE_ALT2_CLASIFIER = 'haarcascade_frontalface_alt2.xml'

"""Inicjalizacja przechwytywania wideo z pierwszego urządzenia kamery"""
cap = cv2.VideoCapture(0)

"""Klasyfikator Haar do wykrywania twarzy"""
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAARCASCADE_DEFAULT_CLASIFIER)

prev_faces = []


def is_face_moved(current_faces, prev_faces, PIXEL_THRESHOLD):
    """
    :param current_faces:
    Lista krotek (x, y, w, h) reprezentujących wykryte twarze w bieżącej klatce.
    Każda krotka zawiera współrzędne (x, y) lewego górnego rogu prostokąta twarzy
    oraz jego szerokość (w) i wysokość (h).

    :param prev_faces:
    Lista krotek (x, y, w, h) reprezentujących wykryte twarze w poprzedniej klatce.
    Format krotek jest taki sam jak w current_faces

    :param threshold:
    (opcjonalnie): Liczbowy próg (domyślnie 10), który określa maksymalną akceptowalną
    różnicę w pikselach dla współrzędnych twarzy między klatkami, aby uznać, że twarz nie poruszyła się.
    Jeśli różnica jest większa niż ten próg, uznaje się, że twarz się poruszyła.

    :return: True / False - True jeśli wykryto ruch, False jeśli nie wykryto
    """
    if len(prev_faces) != len(current_faces):
        return True  # Liczba twarzy się zmieniła

    for prev_face in prev_faces:
        px, py, pw, ph = prev_face
        moved = True  # Zakładamy, że twarz się poruszyła
        for curr_face in current_faces:
            x, y, w, h = curr_face
            if abs(x - px) < PIXEL_THRESHOLD and abs(y - py) < PIXEL_THRESHOLD:
                moved = False  # Znaleziono twarz w podobnym położeniu
                break
        if moved:
            return True  # Jeśli któraś twarz się poruszyła, zwróć True

    return False  # Żadna twarz się nie poruszyła



while True:
    ret, frame = cap.read()
    if not ret:
        break

    """Konwersja obrazu do skali szarości"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """Wykrywanie twarzy"""
    faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBOURS)

    if is_face_moved(faces, prev_faces, 5):
        color = (0, 0, 255)  # Czerwony dla ruchomej twarzy
    else:
        color = (0, 255, 0)  # Zielony dla nieruchomej twarzy

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(frame, center, w // CIRCLE_DIAMETER_DIVIDER, color, CIRCLE_THICKNESS) # Rysowanie okręgu
        cv2.line(frame, (center_x, center_y - VIEWFINDER_LENGTH), (center_x, center_y + VIEWFINDER_LENGTH), color, VIEWFINDER_THICKNESS)  # Pionowa linia celownika
        cv2.line(frame, (center_x - VIEWFINDER_LENGTH, center_y), (center_x + VIEWFINDER_LENGTH, center_y), color, VIEWFINDER_THICKNESS)  # Pozioma linia celownika

    cv2.imshow('Rozpoznawanie twarzy i ruchu', frame)

    prev_faces = copy.deepcopy(faces)

    """ Wyłączenie programu przyciskiem 'q' """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
