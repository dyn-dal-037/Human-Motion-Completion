import tensorflow as tf
model = tf.keras.models.load_model(
    "models/hand_completion_gru.h5",
    compile=False
)
print("Model loaded successfully")
