import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer el frame. ¿Cámara desconectada?")
        break  # Rompe el bucle si la cámara falla

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Procesar solo la PRIMERA mano detectada (índice [0])
        hand_landmarks = results.multi_hand_landmarks[0]

        # --- 1. Dibujar esta mano ---
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # --- 2. Extraer coordenadas primero ---
        # Primero llenamos las listas x_ y y_ para poder encontrar el min/max
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # --- 3. Normalizar datos después ---
        # Ahora que tenemos min(x_) y min(y_), creamos data_aux
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        # --- 4. Dibujar el cuadro y la predicción ---
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10  # (Corregí esto a +10, -10 hacía el cuadro más pequeño)
        y2 = int(max(y_) * H) + 10  # (Corregí esto a +10)

        # Asegurarnos de que el modelo reciba el formato correcto
        prediction = model.predict([np.asarray(data_aux)])
        
        # Obtener el caracter predicho
        predicted_character = labels_dict[int(prediction[0])]

        # Dibujar todo
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)#

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()