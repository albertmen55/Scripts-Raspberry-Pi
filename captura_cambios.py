import time
import picamera
import cv2
import numpy as np
import os

camera = picamera.PiCamera()

#camera.resolution = (1280, 720)
camera.resolution = (1600, 1200)

captura_anterior = None

# Inicializar el sustractor de fondo
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# Directorio para guardar imágenes con cambios
directorio_cambios = "./cambios"
os.makedirs(directorio_cambios, exist_ok=True)

try:
    while True:
        timestamp = time.strftime("%Y_%m_%d_%H:%M:%S")
        filename = f"foto_{timestamp}.jpg"
        ruta_completa = os.path.join(directorio_cambios, filename)
        ruta_escritorio = os.path.join("./", filename)

        camera.capture(ruta_escritorio)

        captura_actual = cv2.imread(ruta_escritorio)

        if captura_anterior is not None:
            # Convertir a escala de grises
            gris_anterior = cv2.cvtColor(captura_anterior, cv2.COLOR_BGR2GRAY)
            gris_actual = cv2.cvtColor(captura_actual, cv2.COLOR_BGR2GRAY)

            # Aplicar desenfoque gaussiano
            gris_anterior = cv2.GaussianBlur(gris_anterior, (21, 21), 0)
            gris_actual = cv2.GaussianBlur(gris_actual, (21, 21), 0)

            # Usar el background subtractor
            fg_mask = bg_subtractor.apply(gris_actual)

            # Aplicar un umbral a la máscara del primer plano
            _, umbral = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

            # Dilatar el umbral para cubrir agujeros en la máscara
            umbral = cv2.dilate(umbral, None, iterations=2)

            # Encontrar contornos
            contornos, _ = cv2.findContours(umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Comprobar si hay cambios significativos
            hay_cambio = False
            for contorno in contornos:
                if cv2.contourArea(contorno) > 750:  # Valor comprobado mediante pequenhas pruebas
                    hay_cambio = True
                    break

            if hay_cambio:
                print("Fotografía tomada con cambios detectados")
                cv2.imwrite(ruta_completa, captura_actual)
                os.remove(ruta_escritorio)  # Borrar del directorio raiz después de moverla a ./cambios
            else:
                print("No hay cambio")
                os.remove(ruta_escritorio)  # Borrar del directorio raiz si no hay cambio

        captura_anterior = captura_actual.copy() #Actualizamos captura actual que sirvirá para el siguiente cambio

except KeyboardInterrupt:
    pass
