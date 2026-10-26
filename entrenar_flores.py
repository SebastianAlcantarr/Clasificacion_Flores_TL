"""Transfer learning con flores con 2 distintas redes base (MobileNetV2 y EfficientNetB0).
Salidas:
    graficas/       -> accuracy.png, matriz.png
    predicciones/   -> muestras_mobilenet.png, muestras_efficientnet.png
    modelos/        -> modelos guardados como H5 para usarlos en tensorflow.js 
"""

import os
import random
import matplotlib
matplotlib.use("Agg")
import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf
from sklearn.metrics   import confusion_matrix
from tensorflow        import keras
from keras.layers      import Dense, Dropout, GlobalAveragePooling2D

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "flowers_jpg")
DIR_GRAFICAS = os.path.join(BASE, "graficas")
DIR_PREDICCIONES = os.path.join(BASE, "predicciones")
DIR_MODELOS = os.path.join(BASE, ' modelos')
os.makedirs(DIR_GRAFICAS, exist_ok=True)
os.makedirs(DIR_PREDICCIONES, exist_ok=True)
os.makedirs(DIR_MODELOS, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATA}/train", validation_split=0.15, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATA}/train", validation_split=0.15, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATA}/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

print(train_dataset.class_names[:5], len(train_dataset.class_names))

# RED Con  MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE)

base_model.summary()

pretrained_model_MobileNetV2 = tf.keras.applications.MobileNetV2(
    input_shape = IMG_SHAPE,
    include_top = False,
    weights     = 'imagenet',
)

pretrained_model_MobileNetV2.trainable = False

pretrained_model_MobileNetV2.summary()

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = pretrained_model_MobileNetV2(x, training=False)

average_pooling = GlobalAveragePooling2D()(x)
drop             = Dropout(0.2)(average_pooling)
outputs_MobileNetV2 = Dense(104, activation='softmax', name='prediccion')(drop)

modelo_flores = tf.keras.Model(inputs=inputs, outputs=outputs_MobileNetV2)

modelo_flores.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

modelo_flores.summary()

epocas_iniciales = 10
historial = modelo_flores.fit(train_dataset, validation_data=validation_dataset, epochs=epocas_iniciales)

modelo_base = modelo_flores.layers[1]
modelo_base.trainable = True

# Fine tune 
fine_tune_at = 130

# congelar capas del modelo base 
for layer in modelo_base.layers[:fine_tune_at]:
    layer.trainable = False

modelo_flores.summary()

modelo_flores.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  
    metrics=["accuracy"]
)

epocas_afinamiennto = 40
epocas_totales = epocas_iniciales + epocas_afinamiennto

historial_fine = modelo_flores.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epocas_totales,
    initial_epoch=historial.epoch[-1] + 1,
    verbose=0
)

test_loss, test_acc = modelo_flores.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

print(train_dataset.class_names == test_dataset.class_names)
print(modelo_flores.evaluate(validation_dataset))

y_true = np.concatenate([y.numpy() for _, y in test_dataset])
y_pred = np.argmax(modelo_flores.predict(test_dataset), axis=1)
cm = confusion_matrix(y_true, y_pred, normalize="true")

acc = [0.] + historial.history['accuracy']
val_acc = [0.] + historial.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1.1])
plt.grid(True)
plt.title('Training and Validation Accuracy')
plt.savefig(os.path.join(DIR_GRAFICAS, "accuracy.png"), dpi=150)
plt.close()

class_names = test_dataset.class_names

plt.figure(figsize=(14, 12))
plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(label='Aciertos por clase')
plt.title("Matriz de confusión , acierto")
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAFICAS, "matriz.png"), dpi=200)
plt.close()

# 5 clases con peor accuracy
acc_clase = np.diag(cm)
peores = np.argsort(acc_clase)[:5]
for i in peores:
    print(f"clase {class_names[i]}: accuracy {acc_clase[i]:.2f}")

TEST_DIR = os.path.join(DATA, "val")


def guardar_muestras(modelo, nombre_archivo):
    muestras = []
    for _ in range(9):
        c = random.choice(os.listdir(TEST_DIR))
        f = random.choice(os.listdir(f"{TEST_DIR}/{c}"))
        muestras.append((f"{TEST_DIR}/{c}/{f}", c))

    imgs = [tf.keras.utils.load_img(r, target_size=IMG_SIZE) for r, _ in muestras]
    batch = np.stack([tf.keras.utils.img_to_array(i) for i in imgs])
    probs = modelo.predict(batch)

    plt.figure(figsize=(10, 10))
    for i, (img, (_, real)) in enumerate(zip(imgs, muestras)):
        pred = class_names[np.argmax(probs[i])]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        color = "green" if pred == real else "red"
        plt.title(f"real {real} | pred {pred}\n{probs[i].max():.0%}", color=color, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PREDICCIONES, nombre_archivo), dpi=150)
    plt.close()


guardar_muestras(modelo_flores, "muestras_mobilenet.png")

"""
    RED BAEfficientNetB0
""" 
base = keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=IMG_SHAPE)

base.trainable = True

capas_modelo_base = len(base.layers)
print(capas_modelo_base)

# Fine-tune from this layer onwards
fine_tune_at = capas_modelo_base
# Freeze all the layers before the `fine_tune_at` layer
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

base.summary()

pretrained_model_EfficientNetB0 = tf.keras.applications.EfficientNetB0(
    input_shape = IMG_SHAPE,
    include_top = False,
    weights     = 'imagenet',
)

pretrained_model_EfficientNetB0.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = pretrained_model_EfficientNetB0(inputs, training=False)

average_pooling = GlobalAveragePooling2D()(x)
drop             = Dropout(0.2)(average_pooling)
outputs_EffcientNetB0 = Dense(104, activation='softmax', name='prediccion')(drop)

modelo_efficientB0 = tf.keras.Model(inputs=inputs, outputs=outputs_EffcientNetB0)

modelo_efficientB0.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),   # ajusta from_logits si hace falta
    metrics=["accuracy"])

epocas_afinamiennto = 40
epocas_totales = epocas_iniciales + epocas_afinamiennto

historial_fine = modelo_efficientB0.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epocas_totales,
    initial_epoch=historial.epoch[-1] + 1
)

test_loss, test_acc = modelo_efficientB0.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

# Fase 1: entrenar solo la cabeza
modelo_efficientB0.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"])
h1 = modelo_efficientB0.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# Fase 2: fine-tuning de las últimas capas
base = modelo_efficientB0.layers[1]
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
# Mantén las BatchNorm congeladas (EfficientNet es sensible a esto)
for layer in base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

modelo_efficientB0.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"])
h2 = modelo_efficientB0.fit(train_dataset, validation_data=validation_dataset,
                            epochs=30, initial_epoch=0,verbose=0)

test_loss, test_acc = modelo_efficientB0.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

guardar_muestras(modelo_efficientB0, "muestras_efficientnet.png")


modelo_flores.save(os.path.join(DIR_MODELOS, "flores_mobilenet.h5"))
modelo_efficientB0.save(os.path.join(DIR_MODELOS, "flores_efficientB0.h5"))
