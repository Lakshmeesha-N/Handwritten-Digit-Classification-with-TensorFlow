import tensorflow as tf

def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, y_train, X_test, y_test
