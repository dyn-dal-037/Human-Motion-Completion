import tensorflow as tf
import numpy as np
from dataset import load_dataset
from model import build_gru_model
from losses import masked_mse

X_list, Y_list, M_list = load_dataset()

model = build_gru_model()
optimizer = tf.keras.optimizers.Adam(1e-3)

EPOCHS = 20

for epoch in range(EPOCHS):
    epoch_loss = []

    for X, Y, M in zip(X_list, Y_list, M_list):
        # reshape to (1, T, 63)
        X = X.reshape(1, X.shape[0], -1)
        Y = Y.reshape(1, Y.shape[0], -1)
        # M: (T, 21, 1) → (T, 21, 3) → (T, 63)
        M = np.repeat(M, 3, axis=2)
        M = M.reshape(1, M.shape[0], -1)


        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)
            loss = masked_mse(Y, Y_pred, M)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss.append(loss.numpy())

    print(f"Epoch {epoch+1}, Loss: {np.mean(epoch_loss):.6f}")

model.save("models/hand_completion_gru.h5")
