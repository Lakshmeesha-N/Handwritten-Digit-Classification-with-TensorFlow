from data_loader import load_data
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping

def train_model():
    X_train, y_train, X_test, y_test = load_data()
    model = create_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=25, validation_split=0.2, callbacks=[early_stop])
    model.save('mnist_model.h5')
    return history

if __name__ == "__main__":
    train_model()
