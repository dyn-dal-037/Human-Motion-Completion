import tensorflow as tf
from tensorflow.keras import layers, models

def build_gru_model(num_joints=21):
    inputs = layers.Input(shape=(None, num_joints * 3))

    x = layers.GRU(128, return_sequences=True)(inputs)
    x = layers.GRU(128, return_sequences=True)(x)
    outputs = layers.Dense(num_joints * 3)(x)

    model = models.Model(inputs, outputs)
    return model
