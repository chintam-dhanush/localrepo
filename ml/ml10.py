

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X1, X2 = np.array_split(X_train, 2)
y1, y2 = np.array_split(y_train, 2)

def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
    return model

model1 = create_model()
model1.fit(X1, y1, epochs=2, verbose=0)
model2 = create_model()
model2.fit(X2, y2, epochs=2, verbose=0)

w1 = model1.get_weights()
w2 = model2.get_weights()

avg_weights = []
for i in range(len(w1)):
    avg_weights.append((w1[i] + w2[i]) / 2)

final_model = create_model()
final_model.set_weights(avg_weights)

loss, acc = final_model.evaluate(X_test, y_test, verbose=0)
print("Final Accuracy:", acc)