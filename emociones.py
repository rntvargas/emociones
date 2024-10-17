import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from deepface import DeepFace

# Diccionario para traducir emociones de inglés a español
emociones_traducidas = {
    'angry': 'Enojado',
    'disgust': 'Disgusto',
    'fear': 'Miedo',
    'happy': 'Feliz',
    'sad': 'Triste',
    'surprise': 'Sorprendido',
    'neutral': 'Neutral'
}

# Función para cargar y convertir la imagen
def cargar_imagen(imagen_subida):
    return np.array(Image.open(imagen_subida))

# Función para detectar caras en la imagen
def detectar_caras(img_array):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

# Función para detectar emociones usando DeepFace
def detectar_emocion(face_roi):
    emotion_result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    
    # Si hay varias caras, seleccionamos la primera
    if isinstance(emotion_result, list):
        emotion_result = emotion_result[0]
    
    # Obtener la emoción dominante y traducirla
    emotion = emotion_result['dominant_emotion']
    return emociones_traducidas.get(emotion, emotion)

# Función para guardar la imagen procesada
def guardar_imagen(img_array, nombre_archivo):
    img = Image.fromarray(img_array)
    img.save(nombre_archivo)
    return nombre_archivo

# Función principal que combina todo y cuenta el número de caras detectadas
def procesar_imagen(imagen_subida):
    img_array = cargar_imagen(imagen_subida)
    faces = detectar_caras(img_array)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = img_array[y:y+h, x:x+w]

        # Detectar emoción en la región de la cara
        emotion_es = detectar_emocion(roi_color)

        # Añadir la emoción detectada sobre la cara
        cv2.putText(img_array, emotion_es, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Retornar la imagen procesada y el número de caras detectadas
    return img_array, len(faces)

# Streamlit para la interfaz de usuario
st.title("Detección de Caras y Emociones")
st.subheader("Sube una imagen para detectar caras y emociones")

imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    img_procesada, num_caras = procesar_imagen(imagen_subida)
    
    # Mostrar la imagen procesada
    st.image(img_procesada, channels="RGB", use_column_width=True)
    
    # Mostrar el número de caras detectadas
    st.write(f"Número de caras detectadas: {num_caras}")
    
    # Botón para guardar la imagen procesada
    if st.button("Guardar imagen procesada"):
        nombre_archivo = "imagen_procesada.jpg"
        ruta_guardada = guardar_imagen(img_procesada, nombre_archivo)
        st.write(f"Imagen guardada como: {ruta_guardada}")