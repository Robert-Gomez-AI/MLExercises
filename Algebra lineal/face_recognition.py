import cv2
import numpy as np

# Inicialización de la captura de video
capture = cv2.VideoCapture(0)

# Carga del clasificador para detección de rostros
classifier = cv2.CascadeClassifier('Algebra lineal/weights/haarcascade_frontalface_default.xml')

# Definición del color verde para el contorno del rostro
green = (0, 255, 0)

while True:
    # Lectura del fotograma de la cámara
    ret, frame = capture.read()
    
    # Conversión del fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros en la imagen en escala de grises
    faces = classifier.detectMultiScale(gray, 1.5, 5)
    #Él metodo anterior recorre la imagen mediante ventanas y usa un clasificador para detectar el rostro
    # A partir de ahí retorna los valores (x,y) de la esquina superior de la detección y w,h dando las dimensiones del rectangulo.

    # Dibujar un rectángulo verde alrededor de cada rostro detectado
    for (x, y, h, w) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)

    # Mostrar la imagen con los rostros detectados
    cv2.imshow('Imagen de prueba', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
capture.release()
cv2.destroyAllWindows()