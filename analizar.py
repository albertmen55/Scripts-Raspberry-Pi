import cv2
import numpy as np
import os
import time
import shutil
import json
from kafka import KafkaProducer
import base64

#Funcion para calcular la distancia de un punto a una recta
def distancia_punto_a_recta(punto, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distancia = np.abs((y2 - y1) * punto[0] - (x2 - x1) * punto[1] + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return distancia

#Funcion para clasificar un conjunto de puntos en ArUcos Verticales u Horizontales 
def clasificar_puntos(aux, aruco_vertical, aruco_horizontal):
    for punto in aux:
        distancia_vertical = distancia_punto_a_recta(punto, aruco_vertical[0], aruco_vertical[1])
        distancia_horizontal = distancia_punto_a_recta(punto, aruco_horizontal[0], aruco_horizontal[1])
        if distancia_vertical < distancia_horizontal:
            aruco_vertical.append(punto)
        else:
            aruco_horizontal.append(punto)

#Funcion principal
def detectar_ArUco(ruta, producer, topic):

    imagen = cv2.imread(ruta)
    if imagen is None:
        print(f"Error al leer la imagen: {ruta}")
        return

    # Extraer timestamp del nombre del archivo original
    timestamp_original = '_'.join(os.path.splitext(os.path.basename(ruta))[0].split('_')[1:])

    #Definimos parametros necesarios para la deteccion de ArUcos y se activa el detector
    diccionario = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parametros = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(imagen, diccionario, parameters=parametros)

    if ids is not None:

        for i in range(len(ids)):
            print(f"Marcador identificado de ID: {ids[i]}")

        #En caso de que se detecten 3 o mas marcadores (Minimo para delimitar una seccion)
        if len(ids) >= 3:
            centros = []
            
            #Calculo de centro de cada marcador en base a las coordenadas de sus esquinas
            for i in range(len(ids)):
                esquinas = corners[i][0]
                centro_x = int(np.mean(esquinas[:, 0]))
                centro_y = int(np.mean(esquinas[:, 1]))
                centros.append((centro_x, centro_y))

            #Calculamos el punto minimo, el maximo y el que delimita la parte inferior izquierda del cajon
            punto_min = min(centros)
            punto_max = max(centros)
            punto_abajo = list((punto_max[0] - punto_min[0], punto_max[1]))

            #Anhadimos el punto minimo a los ArUcos del eje X y el maximo a los del eje Y
            aruco_vertical = [punto_min]
            aruco_horizontal = [punto_max]

            vectores = [(centro[0] - punto_abajo[0], centro[1] - punto_abajo[1]) for centro in aux]  # Calculamos los vectores desde el punto de abajo
            magnitudes = np.linalg.norm(vectores, axis=1)  # Calculamos las magnitudes de los vectores
            indice = np.argmax(magnitudes)  # Encontramos el índice del vector con la mayor magnitud
            aruco_identificador = aux[indice]  # Obtenemos el punto con la mayor magnitud
            aruco_vertical.append(aruco_identificador)  # Añadimos el identificador a los ArUcos verticales
            aruco_horizontal.append(aruco_identificador)  # Añadimos el identificador a los ArUcos horizontales
            clasificar_puntos(aux, aruco_vertical, aruco_horizontal)  # Clasificamos los puntos restantes

            #Ordenamos los puntos para que los recortes para que se hagan en orden
            recortes_verticales = sorted(list(set(aruco_vertical)), key=lambda punto: punto[0])
            recortes_horizontales = sorted(list(set(aruco_horizontal)), key=lambda punto: punto[1])

            imagenes_horizontales = []

            #Primero recortamos la imagen horizontalmente
            for i in range(len(recortes_horizontales) - 1):
                x1, y1 = (punto_min[0], recortes_horizontales[i][1])
                x2, y2 = recortes_horizontales[i]
                x3, y3 = (punto_min[0], recortes_horizontales[i + 1][1])
                x4, y4 = recortes_horizontales[i + 1]
                imagenes_horizontales.append(imagen[y1:y3, x1:x2])

            imagenes_definitivas = []

            #Recortamos las imagenes horizontales anteriores verticalmente
            for recorte in imagenes_horizontales:
                for i in range(len(recortes_verticales) - 1):
                    x1, y1 = (recortes_verticales[i][0] - punto_min[0], recortes_verticales[i][1] - punto_min[1])
                    x2, y2 = (recortes_verticales[i + 1][0] - punto_min[0], recortes_verticales[i][1] - punto_min[1])
                    x3, y3 = (recortes_verticales[i][0] - punto_min[0], recorte.shape[1])
                    x4, y4 = (recortes_verticales[i + 1][0] - punto_min[0], recorte.shape[1])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = max(0, x2)
                    y2 = max(0, y2)
                    x3 = max(0, x3)
                    y3 = max(0, y3)
                    x4 = max(0, x4)
                    y4 = max(0, y4)
                    recorte_img = recorte[y1:y3, x1:x2]
                    if recorte_img.size > 0:  # Verificar si la imagen no esta vacia por errores de detección de ArUco
                        imagenes_definitivas.append(recorte_img)

            #Creamos o definimos la carpeta donde se almacenaran estos recortes
            nombre_carpeta = '-'.join(map(str, ids)) + '_' + timestamp_original
            carpeta_destino = os.path.join("./detectadas", nombre_carpeta)
            os.makedirs(carpeta_destino, exist_ok=True)

            #Guardamos los recortes enumerandolos en orden de recorte
            for i, definitiva in enumerate(imagenes_definitivas):
                nombre_archivo = f"recortada_{i}_{timestamp_original}.jpg"
                ruta_img_recortada = os.path.join(carpeta_destino, nombre_archivo)
                cv2.imwrite(ruta_img_recortada, definitiva)

            #Guardamos tambien la imagen original en la carpeta donde se detectan los ArUco
            imagen_res = cv2.aruco.drawDetectedMarkers(imagen.copy(), corners, ids)
            nombre_arch = f"{os.path.splitext(os.path.basename(ruta))[0]}.jpg"
            ruta_imagen_pr = os.path.join(carpeta_destino, nombre_arch)
            cv2.imwrite(ruta_imagen_pr, imagen_res)
            
            # Creamos o definimos la carpeta a donde moveremos la imagen ya procesada
            processed_dir = "./processed"
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir, exist_ok=True)
            
            # Movemos la imagen ya analizada al directorio "processed" como copia de respaldo
            try:
                processed_file_path = os.path.join(processed_dir, os.path.basename(ruta))
                shutil.move(ruta, processed_file_path)
                print(f"Archivo movido a: {processed_file_path}")
            except Exception as e:
                print(f"Error al mover el archivo {ruta} a {processed_dir}: {e}")

            #Definimos el directorio para producir con Kafka el mismo
            directory_data = {
                'directory': carpeta_destino,
                'files': {}
            }

            #Para cada archivo del directorio lo anhadimos a ese directorio en base 64 
            for file_name in os.listdir(carpeta_destino):
                file_path = os.path.join(carpeta_destino, file_name)
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    file_data_base64 = base64.b64encode(file_data).decode('utf-8')
                    directory_data['files'][file_name] = file_data_base64

            producer.send(topic, directory_data) #Publicamos en el topico
            print(f"Directorio publicado: {carpeta_destino}")
        else:
            carpeta_destino = None  # Asegurarse de que carpeta_destino está definido
            print("No se identificaron suficientes marcadores ArUco.")
            try:
                os.remove(ruta)
                print(f"Archivo eliminado: {ruta}")
            except Exception as e:
                print(f"Error al eliminar el archivo {ruta}: {e}")
    else:
        carpeta_destino = None  # Asegurarse de que carpeta_destino está definido
        print("No se detectaron marcadores ArUco.")
        try:
            os.remove(ruta)
            print(f"Archivo eliminado: {ruta}")
        except Exception as e:
            print(f"Error al eliminar el archivo {ruta}: {e}")

def main():
    #Definicion de parametros necesarios
    kafka_server = '169.254.52.160:9092'
    topic = 'topic_oficial_tfg'
    base_directory = './cambios'

    #Inicialización de productor
    producer = KafkaProducer(
        bootstrap_servers=[kafka_server],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        max_request_size=5000000,  # 5 MB
        buffer_memory=33554432  # 32 MB
    )

    try:
        while True:
            #Abrimos el directorio donde estan las imagenes a analizar y cogemos los archivos por la hora
            esc = os.listdir(base_directory)
            archivos = sorted(
                (filename for filename in esc if filename.lower().endswith(".jpg")),
                key=lambda x: os.path.getmtime(os.path.join(base_directory, x))
            )
            
            # Mientras no haya archivos .jpg esperamos y comprobamos cada 3 segundos
            while not archivos:
                time.sleep(3)
                esc = os.listdir(base_directory)
                archivos = sorted(
                    (filename for filename in esc if filename.lower().endswith(".jpg")),
                    key=lambda x: os.path.getmtime(os.path.join(base_directory, x))
                )

            for filename in archivos:
                ruta = os.path.join(base_directory, filename)
                #Si no ha sido ya analizado el archivo lo analizamos
                if not os.path.isfile(os.path.join("./processed", filename)):
                    detectar_ArUco(ruta, producer, topic)
                    time.sleep(2)

    except KeyboardInterrupt:
        print("Finalizando...")

    #Cerramos el productor
    producer.close()

if __name__ == "__main__":
    main()
