import tensorflow as tf

def masked_mse(y_true, y_pred, mask):
    mask = tf.cast(mask, tf.float32)   # ← ADD THIS LINE
    missing = 1.0 - mask
    error = tf.square(y_true - y_pred)
    error = error * missing
    return tf.reduce_sum(error) / (tf.reduce_sum(missing) + 1e-6)
