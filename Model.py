import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout

def build_model():
    inputs = Input(shape=(160,160,3))
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs, alpha=0.35)

    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation= "relu")(x)

    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
