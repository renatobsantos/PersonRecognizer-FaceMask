import cv2
import numpy as np
from imutils.video import VideoStream
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
from keras.utils import img_to_array


def detect_and_predict_mask(frame, faceNet, maskNet):
    # Pega a altura e largura do frame (captura de video)
    (h, w) = frame.shape[:2]
    # Cria um blob -> um objeto do tipo do arquivo com dados brutos imutáveis
    # Mais informações em:
    # https://developer.mozilla.org/pt-BR/docs/Web/API/Blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # passa o blob para o modelo de detectar faces
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # faces detectadas, posicão de cada face e as previsões
    faces = []
    locs = []
    preds = []

    # para cada face detectada analisa se está ou não com máscara
    for i in range(0, detections.shape[2]):
        # confiança de cada face detectada
        confidence = detections[0, 0, i, 2]

        # confiança tem que ser acima de 50% para a precição ser maior
        if confidence > 0.5:
            # pega o contorno (retângulo) da face detectada
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # Região de Interesse (ROI)
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Muda o tamanho da imagem para 224x224
            face = cv2.resize(face, (224, 224))
            # imagem vira um array
            face = img_to_array(face)

            # Pré processa uma matriz numpy que codifica um lote de imagens
            # Mais informações em:
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # a previção so é feita se tiver alguma face no frame
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # retorna a localização de cada face e a previsão
    return locs, preds


# Modelo para detectar rostos
# baseado no modelo caffemodel https://caffe.berkeleyvision.org/#:~:text=Caffe%20is%20a%20deep%20learning,the%20BSD%202%2DClause%20license.
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# carrega o modelo treinado
maskNet = load_model("training.model")

# Indica que vai ter uma captura de vídeo
vs = VideoStream(src=0).start()

# while que faz um loop entre os frames da captura de vídeo
while True:
    # le frame a frame a captura de video
    frame = vs.read()

    if cv2.waitKey(1) % 256 == 27:  # ESC = (27 em ASCII) foi pressionado para fechar o programa
        break
    # faz a detecção se a pessoa está utilizando máscara ou não
    # retorna a localização de cada face e a previsão
    locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

    # zip(locs, preds) -> ex:
    # locs = ['a', 'b', 'c']
    # preds = [1, 2, 3]
    # zip(locs, preds) = [('a', 1), ('b', 2), ('c', 3)]
    # uma tupla com uma correlação entre listas
    for box, pred in zip(locs, preds):
        startX, startY, endX, endY = box
        mask, withoutMask = pred

        # Rótulo e cor para mostrar pro usuário se está com máscara ou não
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # probabilidade da previsão está correta
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # mostra na tela o resultado
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)


# Libera e destroi a captura de tela para não ficar na memória
cv2.destroyAllWindows()
vs.stop()
