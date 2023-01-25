import os
import pickle

import cv2
import numpy as np
from PIL import Image

# Haarcascade é utilizado para identificar objetos, nesse caso utilizamos a 'haarcascade_eye.xml'
# para detectar olhos

# Mais informações em:
# https://pyimagesearch.com/2021/04/12/opencv-haar-cascades/
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# Reconhecedor de rosto baseado em LBP (Local Binary Pattern) que as imagens estão em binário
# e são reconhecidas rotulando os pixels de uma imagem limitando a vizinhança de cada pixel
# e considera o resultado como um número binário.# Mais informações em:
# https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIRECTORY = "./images/"

current_id = 0
labels_id = {}
y_labels = []
x_train = []

# os.walk(dir) faz com que seja possível um for caminhar olhando os diretórios
# e os arquivos desse diretórios internos a partir de um diretório
for root, dirs, files in os.walk(BASE_DIRECTORY):
    for file in files:
        # Garantia que o arquivo é uma imagem com extensão .png
        if file.endswith("png"):
            # path contém o caminho até esse arquivo
            # label possui o nome do diretório desse arquivo
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()

            # labels_id é um dicionário que as chaves são o nome dos diretórios
            # e o valor é um contador que soma 1 toda vez que uma chave nova
            # é adicionada
            if not label in labels_id:
                labels_id[label] = current_id
                current_id += 1

            # Recebe o valor atual do current_id
            id_ = labels_id[label]

            # Uma forma de garantir que o treino tenha uma precisão maior
            # é garantir que todas as imagens estejam do mesmo tamanho
            # e mudar para grayscale (escala cinza) de cores para facilitar
            # a identificação de cada pixel da imagem
            # conversão da imagem para grayscale
            pil_image = Image.open(path).convert('L')
            # Muda o tamanho de todas as imagens para 128x128 e
            final_image = pil_image.resize(
                (128, 128), Image.Resampling.LANCZOS)
            # passa essas imagens tratadas para um array
            image_array = np.array(final_image, "uint8")

            # eye_cascade.detectMultiScale faz a detecção baseado na cascade utilizada
            # scaleFactor é o quanto vai diminuir a imagem para, quanto mais perto de 1 mais preciso - menos performance
            # minNeighbors afeta a qualidade dos olhos detectados,
            # valores altos resulta em menos detecção, mas melhor qualidade
            # retorna um retângulo do que foi identificado
            eyes = eye_cascade.detectMultiScale(
                image_array, scaleFactor=1.3, minNeighbors=5)

            # for para caminhar dentro do que foi detectado pelo detectMultiScale
            # x: coordenada x do retângulo
            # y: coordenada y do retângulo
            # w: largura do retângulo
            # h: altura do retângulo
            for (x, y, w, h) in eyes:
                # Region of Interest (roi) é basicamente a partir de uma imagem pega somente o que é interessante para
                # o algoritmo
                roi = image_array[y:y+h, x:x+w]

                # x_train: roi: olho da pessoa
                # y_labels: id da label
                x_train.append(roi)
                y_labels.append(id_)


# Salva as labels em arquivo .pickle que serve para guardar binário
with open('labels.pickle', 'wb') as f:
    pickle.dump(labels_id, f)

# Reconhecedor vai treinar e salvar o treino
recognizer.train(x_train, np.array(y_labels))
recognizer.save('training.yml')
