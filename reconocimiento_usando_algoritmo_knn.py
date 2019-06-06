"""
En este ejemplo usamos el algoritmo k-nearest-neighbors (KNN) para el reconocimiento facial

Este ejemplo es útil cuando desea reconocer un gran conjunto de personas conocidas,
y hacer una predicción para una persona desconocida en un tiempo de cálculo factible.

Descripción del algoritmo:
El algoritmo clasificador KNN se entrena primero en un conjunto de caras etiquetadas (conocidas) y luego puede predecir a la persona en otra imagen
en una imagen desconocida al encontrar las k caras más similares (imágenes con rasgos faciales de armarios a una distancia euclediana)
en su conjunto de entrenamiento, y realizar un voto mayoritario (posiblemente ponderado) en su etiqueta.

Por ejemplo, si k = 3, y las tres imágenes de rostro más cercanas a la imagen dada en el conjunto de entrenamiento son una imagen de Biden
y dos imágenes de Obama, el resultado sería 'Obama'.

* Esta implementación utiliza un voto ponderado, de manera que los votos de los vecinos más cercanos se ponderan más.
Usage:

1. Prepare un conjunto de imágenes de las personas conocidas que desea reconocer. Organiza las imágenes en un solo directorio.
   con un subdirectorio para cada persona conocida.

2. Luego, llame a la función 'train' con los parámetros apropiados. Asegúrese de pasar en 'model_save_path' si desea guardar el modelo en el disco para poder reutilizarlo sin tener que volver a entrenarlo.

3. Llame 'analizar' y pase su modelo entrenado para reconocer a las personas en una imagen desconocida.

NOTA: ¡Este ejemplo requiere que scikit-learn esté instalado! Puedes instalarlo con pip:

$ pip3 install scikit-learn

"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path='save', n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Entrena a un k-clasificador de vecinos más cercano para reconocimiento facial.

     : variable train_dir: directorio que contiene un subdirectorio para cada persona conocida, con su nombre.

      (Ver en el código fuente para ver la estructura de árbol de ejemplo train_dir)
     Estructura:
        <imagenes de entrenamiento>/
        ├── <persona1>/
        │   ├── <foto1>.jpeg
        │   ├── <foto2>.jpeg
        │   ├── ...
        ├── <persona2>/
        │   ├── <foto1>.jpeg
        │   └── <foto2>.jpeg
        └── ...

    :variable model_save_path: (optional) path to save model on disk
    :variable n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :variable knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :variable verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = [] #definimos estas variables para almacenar informacion como una lista(array) para almacenar datos si no se define asi la informacion se sobre escribe
    y = []

   # Recorra cada persona en el conjunto de entrenamiento.
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

       # Recorre cada imagen de entrenamiento para la persona actual
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # Si no hay personas (o demasiadas personas) en una imagen de entrenamiento, omita la imagen de entrenamiento.
                if verbose:
                    print("Imagen {} no apta para entrenamiento: {}".format(img_path, "No se encontro nunguna cara" if len(face_bounding_boxes) < 1 else "Se encontro mas de una cara"))
            else:
               # Agregar codificación de cara para la imagen actual al conjunto de entrenamiento
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

# Determine cuánto en comun hay para ponderar en el clasificador KNN
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Elija n_neighbors automáticamente:", n_neighbors)

    # Crea y entrena el clasificador KNN 
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Guarda la imagen generada del modelo entrenado KNN
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
   
Reconoce caras en una imagen dada usando un clasificador KNN entrenado

    : variable X_img_path: ruta a la imagen a reconocer
    : variable knn_clf: (opcional) un objeto clasificador knn. si no se especifica, se debe especificar model_save_path.
    : variable model_path: (opcional) ruta a un clasificador knn encurtido. si no se especifica, model_save_path debe ser knn_clf.
    : variable distance_threshold: (opcional) umbral de distancia para la clasificación de caras. cuanto más grande es, más posibilidades hay
           de clasificar erróneamente a una persona desconocida como conocida.
    : return: una lista de nombres y ubicaciones de caras para las caras reconocidas en la imagen: [(nombre, cuadro delimitador), ...].
        Para las caras de personas no reconocidas, se devolverá el nombre "desconocido".
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("carpeta de imagenes invalida: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

# Cargar un modelo KNN entrenado (si se pasó uno)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

  # Cargar archivo de imagen y encontrar ubicaciones de caras
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Usa el modelo KNN para buscar el que conincida con la cara de entrenamiento
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("Desconocido", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
   Muestra visualmente los resultados del reconocimiento facial.

     : variable img_path: ruta a la imagen a reconocer
     : predicciones variables: resultados de la función de predicción
     :regreso:
     """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Dibuja un cuadro usando el modulo Pillow
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # El modulo Pillow tiene un bug que hace que se cierre si  no se declara el UTF-8
        # se usara la fuente prederterminada
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Elimina de la memoria la libreria Pillow docs
    del draw

    # Muestra en pantalla el resultado de las imagenes
    pil_image.show()


if __name__ == "__main__":
# PASO 1: Entrene al clasificador de KNN y guárdelo en el disco
# Una vez que el modelo está entrenado y guardado, puede omitir este paso la próxima vez.
    print("Entrenando al modelo KNN...")
    classifier = train("imagenes de entrenamiento", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Entrenamiento completado!")

# PASO 2: Usando el clasificador entrenado, haga predicciones para imágenes desconocidas
    for image_file in os.listdir("analizar"):
        full_file_path = os.path.join("analizar", image_file)

        print("Detectando las caras en la imagen  {}".format(image_file))

        # Busca todas las personas en la imagen usando el modelo clasificador
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Imprime los resultados en consola 
        for name, (top, right, bottom, left) in predictions:
            print("- Cara encontrada {} en las coordenadas ({}, {})".format(name, left, top))

        # Mostrar resultados en una imagen
        show_prediction_labels_on_image(os.path.join("analizar", image_file), predictions)
