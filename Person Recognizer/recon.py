import pickle

import cv2
import numpy as np

# Haarcascade é utilizado para identificar objetos, nesse caso utilizamos a 'haarcascade_eye.xml'
# para detectar olhos

# Mais informações em:
# https://pyimagesearch.com/2021/04/12/opencv-haar-cascades/
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# Reconhecedor de rosto baseado em LBP (Local Binary Pattern) que as imagens estão em binário
# e são reconhecidas rotulando os pixels de uma imagem limitando a vizinhança de cada pixel
# e considera o resultado como um número binário.

# Mais informações em:
# https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

labels = {}

# Abre o arquivo labels.pickle gerado em train.py
with open('labels.pickle', 'rb') as f:
    # og_labels: {dir_name: id}
    # labels = {v: k for k, v in og_labels.items()} faz a inversão de chave e valor
    # labels: {id: dir_name}
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


# kernel se refere a um método que possibilita separar o que está na dimensão x do que está y,
# ou seja, lineariza algo que não é linear
# kernel = matriz 3 x 3 preenchida apenas com 1

# Mais informações em:
# https://programmathically.com/what-is-a-kernel-in-machine-learning/#:~:text=In%20machine%20learning%2C%20a%20kernel,understand%20that%20higher%2Ddimensional%20space.
kernel = np.ones((3, 3), np.uint8)

# Declara que vai ter uma captura de vídeo
cap = cv2.VideoCapture(0)


# Função para aparecer um texto no canto superior esquerdo da tela

# Mais informações em:
# https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(255, 255, 255),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)

    return text_size


# while que faz um loop entre os frames da captura de vídeo
while True:
    ret, frame = cap.read()

    # Se não tiver o ret (retorno) para o programa
    if not ret:
        break
    if cv2.waitKey(1) % 256 == 27:  # ESC = (27 em ASCII) foi pressionado para fechar o programa
        break

    # Converte a captura de tela para grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # eye_cascade.detectMultiScale faz a detecção baseado na cascade utilizada
    # scaleFactor é o quanto vai diminuir a imagem para, quanto mais perto de 1 mais preciso - menos performance
    # minNeighbors afeta a qualidade dos olhos detectados,
    # valores altos resulta em menos detecção, mas melhor qualidade
    # retorna um retângulo do que foi identificado
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # for para caminhar dentro do que foi detectado pelo detectMultiScale
    # x: coordenada x do retângulo
    # y: coordenada y do retângulo
    # w: largura do retângulo
    # h: altura do retângulo
    for (x, y, w, h) in eyes:
        # Region of Interest (roi) é basicamente a partir de uma imagem pega somente o que é interessante para
        # o algoritmo
        # roi_gray: roi em grayscale
        roi_gray = gray[y:y+h, x:x+w]

        # cv2.threshold serve para pegar a imagem em binário a partir de uma imagem em grayscale
        ret, binary = cv2.threshold(roi_gray, 60, 255, cv2.THRESH_BINARY_INV)

        # pega a largura e altura da imagem em binário
        width, height = binary.shape

        # Cortar 30% do topo da imagem para remover a sombracelha da imagem
        binary = binary[int(0.3 * height):height, :]
        cv2.imshow('Iris', binary)

        # cv2.morphologyEx aplica transformações morfológicas em imagens binárias

        # Imagem Binária: 1 = Branco | 0 = Preto

        # Convolução é simplesmente uma operação de somatório do produto entre duas funções,
        # ao longo da região em que elas se sobrepõem, em razão do deslocamento existente entre elas.

        # Erosion: Vai erosar as bordas da Região de Interesse (ROI) da imagem atráves de uma convulução 2D feita pelo kernel e
        # no pixel que for 1 na imagem vai ser mantido e se for 0 vai ser erosado (remoção de ruído)

        # Dilate: Faz o contrário do Erosion, ou seja, vai aumentar o tamanho da Região de Interesse

        # Opening: É o processo de fazer uma Erosion seguido de uma Dilate, é utilizado para remover ruído da imagem,
        # porque Erosion diminui o tamanho da ROI e e Dilate aumenta

        # Mais informações em:
        # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        opening = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel)

        # Outro dilate para aumentar mais a ROI, mas sem o ruído em volta
        dilate = cv2.morphologyEx(
            opening, cv2.MORPH_DILATE, kernel)

        # Pega o id e a confiança (conf) da previsão do reconhecdor com roi_gray
        id_, conf = recognizer.predict(roi_gray)

        # Vai pegar o contorno dos pixels que o valor é 1 e que tenha uma forma geométrica
        # cv2.RETR_TREE retorna todos os contornos e cria uma hierarquia
        # cv.CHAIN_APPROX_NONE pega todos os pixels = 1 da imagem, tem menos performance, mas funciona melhor com círculos
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)

        # contorno será zero se não for possivel encontrar uma forma geométrica
        # confiança tem que ser maior de 50% para poder validar a previsão
        if len(contours) != 0 and conf >= 50:
            name = labels[id_]
            draw_text(frame, name, pos=(10, 10))

    cv2.imshow('Frame', frame)

# Libera e destroi a captura de tela para não ficar na memória
cap.release()
cv2.destroyAllWindows()
