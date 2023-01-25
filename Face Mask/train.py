import os

import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img, to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

LEARNING_RATE = 0.0001
EPOCHS = 20
BATCH_SIZE = 32

DIRECTORY = "dataset"
CATEGORIES = ["com-mascara", "sem-mascara"]

images = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    # path = './dataset/com-mascara/' ou './dataset/sem-mascara/'

    # os.listdir(path) carrega todos os arquivos dentro do diretório
    for image in os.listdir(path):
        # img_path = './dataset/com-mascara/image.png'
        img_path = os.path.join(path, image)
        # Carrega a imagem no tamanho 224x224
        image = load_img(img_path, target_size=(224, 224))
        # imagem vira um array
        image = img_to_array(image)
        # Pré processa uma matriz numpy que codifica um lote de imagens
        # Mais informações em:
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input
        image = preprocess_input(image)

        images.append(image)
        labels.append(category)

# Manda as labels para binário
# Mais informações em:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
lb = LabelBinarizer()
# transforma labels multi-classe em binário
labels = lb.fit_transform(labels)
# Converte um vetor de inteiros para uma matriz binária
labels = to_categorical(labels)

# Converte uma lista normal do python em um array numpy
IMAGES = np.array(images, dtype="float32")
LABELS = np.array(labels)

# Divide vetores ou matrizes em subsets divido entre treino e teste
# Mais informações em:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
trainX, testX, trainY, testY = train_test_split(IMAGES, LABELS,
                                                test_size=0.20, stratify=labels, random_state=42)

# Gera batches com Tensor Imagens em real-time data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# MobileNetV2 retorna um modelo de classificação de imagens
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# headModel pega o resultado do modelo base e aplica filtro de imagens para o modelo ficar mais uniforme e
# as imagens vão ter o mesmo tratamento para melhorar a detecção
# AveragePooling2D: Reduz a qualidade fora da Região de Interesse (ROI) da imagem
# Flatten: Remove a profundidade da imagem
# Dense: Deixa a camada mais densamente-conectada, ou seja, pessoas com ponto de vista diferente enxergam da mesma maneira
# Dropout: Previne o Overfitting (resultados bons no treinamento e ruins no teste) durante o treinamento

# Mais informações em:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)

# Dense(integer > 0, activation="function")
# integer é o dimensão do resultado espacial
# ReLU (Rectified Linear Unit) é uma função ativadora definida com a parte positiva apenas
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)

# Softmax é uma função ativadora que retorna uma distribuição de probabilidade das categorias previstas
headModel = Dense(2, activation="softmax")(headModel)

# Cria um modelo a partir do modelo base e o modelo topo
model = Model(inputs=baseModel.input, outputs=headModel)

# Evitar que as camadas do modelo congele e não seja atualizada no treinamento
for layer in baseModel.layers:
    layer.trainable = False

# Compilação do Modelo
# Adam optimizer é um método estocástico de gradiente descendente
# que se baseia na estimativa adaptativa de momentos de primeira e segunda ordem.
# lr: taxa de aprendizagem
# decay: decaimento da taxa de aprendizagem https://stackoverflow.com/questions/39517431/should-we-do-learning-rate-decay-for-adam-optimizer

# loss:
# binary_crossentropy =  média da perda do categorical crossentropy
# categorical crossentropy quando so pode fazer parte de uma categoria e faz a quantificação da diferença entre 2 prováveis distribuições

# metrics: calcula a frequência de que a previsão é verdadeira e busca sempre a maior precisão
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS), loss="binary_crossentropy",
              metrics=["accuracy"])

# treina o modelo
# aug -> ImageDataGenerator
# aug.flow gera batches com os dados
history = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS)

# salva o modelo treinado
model.save("training.model", save_format="h5")
