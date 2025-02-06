import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from data_loader import load_data

def evaluate_model():
    X_train, y_train, X_test, y_test = load_data()
    model = load_model('mnist_model.h5')
    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    history = model.history
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
