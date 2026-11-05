
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
X = np.random.randn(200, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 1. L2 Regularization
model_l2 = tf.keras.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(16, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1, activation='sigmoid')
])

# 2. Dropout
model_dropout = tf.keras.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 3. Early Stopping
model_es = tf.keras.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

es_callback = callbacks.EarlyStopping(
    monitor='loss', patience=3
)

# Compile and train
models = [
    ("L2", model_l2, []),
    ("Dropout", model_dropout, []),
    ("EarlyStopping", model_es, [es_callback])
]

for name, model, cb in models:
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        callbacks=cb,
        verbose=0
    )

    print(f"{name} Model final loss: {history.history['loss'][-1]:.4f}")
