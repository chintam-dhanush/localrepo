

#week10
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, SimpleRNN, Dense, Dropout

# 1. Set Parameters
VOCAB_SIZE = 10000
MAX_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 5


tf.keras.utils.set_random_seed(42)


# 2. Load IMDB Dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 3. Pad Sequences
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding="post")
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding="post")

# 4. Build Bidirectional RNN
model = Sequential([
    Embedding(VOCAB_SIZE, 32, mask_zero=True),
    Bidirectional(SimpleRNN(32)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])


# 5. Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train Model
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)


# 7. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")


# 8. Predict Sentiment

predictions = model.predict(X_test[:10], verbose=0)

for i in range(10):
    pred = "Positive" if predictions[i][0] >= 0.5 else "Negative"
    actual = "Positive" if y_test[i] == 1 else "Negative"

    print("Review", i + 1, "| Predicted:", pred, "| Actual:", actual)


# 9. Plot Accuracy

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Sentiment Analysis")
plt.legend()
plt.show()


