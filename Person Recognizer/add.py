import os

import cv2

# Declara que vai ter uma captura de vídeo
cap = cv2.VideoCapture(0)

# Haarcascade é utilizado para identificar objetos, nesse caso utilizamos a 'haarcascade_frontalface_alt2.xml'
# para detectar faces

# Mais informações em:
# https://pyimagesearch.com/2021/04/12/opencv-haar-cascades/
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_alt2.xml')

BASE_DIRECTORY = './images'

# Contador de quantas fotos foram tiradas
img_counter = 1

# Se não existe o diretório 'images' é criado
if not os.path.exists(BASE_DIRECTORY):
    os.mkdir(BASE_DIRECTORY)

# while para adicionar uma pessoa no dataset, a partir do nome e sobrenome da pessoa
# se o nome já existir repete esse loop, se não cria um diretório com nome e sobrenomde
# separado por hífen
while True:
    print('Digite o nome da pessoa que será adicionada no dataset\n')
    name = input('Nome da Pessoa: ').lower()
    lastname = input('Sobrenome da Pessoa: ').lower()

    fullname = (name + '-' + lastname).replace(" ", "-")
    if fullname.endswith('-'):
        fullname = fullname[:-1]

    PATH = './images/{}'.format(fullname)

    if(os.path.exists(PATH)):
        print('\nPessoa já existente no banco de dados!!! \n')
    else:
        os.mkdir(PATH)
        break

# while que faz um loop entre os frames da captura de vídeo
while True:
    # lê as informações da captura de vídeo
    ret, frame = cap.read()

    # Se não tiver o ret (retorno) para o programa
    if not ret:
        break

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC = (27 em ASCII) foi pressionado para fechar o programa
        break
    # Espaço = (32 em ASCII) foi pressionado para tirar uma foto
    elif k % 256 == 32:
        # Nomeia a imagem baseado no contador de imagem (img_counter)
        img_name = "{}.png".format(img_counter)

        # Salva a imagem do rosto da pessoa no dataset
        cv2.imwrite(os.path.join(
            PATH, img_name), frame)
        img_counter += 1

        # Quando tirar 10 fotos o programa é fechado
        if img_counter == 11:
            break

    # Converte a captura de tela para grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face_cascade.detectMultiScale faz a detecção baseado na cascade utilizada
    # scaleFactor é o quanto vai diminuir a imagem para, quanto mais perto de 1 mais preciso - menos performance
    # minNeighbors afeta a qualidade dos olhos detectados,
    # valores altos resulta em menos detecção, mas melhor qualidade
    # retorna um retângulo do que foi identificado
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # for para caminhar dentro do que foi detectado pelo detectMultiScale
    # x: coordenada x do retângulo
    # y: coordenada y do retângulo
    # w: largura do retângulo
    # h: altura do retângulo
    for (x, y, w, h) in faces:
        # Region of Interest (roi) é basicamente a partir de uma imagem pega somente o que é interessante para
        # o algoritmo
        roi_color = frame[y:y+h, x:x+w]

        # estilização de fonte, cor e grossura da linha
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)  # BGR
        stroke = 2

        # Coloca na tela um texto 'Foto {img_counter}' para mostrar para o usuário em qual foto ele está
        cv2.putText(frame, 'Foto: {}'.format(img_counter), (x, y), font, 1,
                    color, stroke, cv2.LINE_AA)

        # Desenha um retângulo na tela para mostrar para o usuário que ele está sendo reconhecido pelo programa
        color = (0, 255, 0)  # BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    cv2.imshow('frame', frame)

# Libera e destroi a captura de tela para não ficar na memória
cap.release()
cv2.destroyAllWindows()
