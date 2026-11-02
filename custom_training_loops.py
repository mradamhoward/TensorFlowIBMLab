import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import Callback
import numpy as np

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

from tensorflow.keras.layers import Input, Dense

# Define the input layer
input_layer = Input(shape=(28, 28))  # Input layer with shape (28, 28)

# Flatten the 2D images into 1D vectors before Dense layers
flatten = Flatten()(input_layer)

# Define hidden layers
hidden_layer1 = Dense(64, activation='relu')(flatten)  # First hidden layer with 64 neurons and ReLU activation
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)  # Second hidden layer with 64 neurons and ReLU activation

output_layer = Dense(10, activation='softmax')(hidden_layer2)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Correct for integer labels 0-9
    metrics=['accuracy']
)

# Step 3: Define Loss Function and Optimizer

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()  # Metric to track accuracy during training


from tensorflow.keras.callbacks import Callback

# Step 4: Implement the Custom Callback
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'End of epoch {epoch + 1}, loss: {logs.get("loss")}, accuracy: {logs.get("accuracy")}')

# Step 4: Implement the Custom Training Loop

epochs = 2
custom_callback = CustomCallback()


# train_dataset = train_dataset.repeat(epochs)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
for epoch in range(epochs):
    print(f'Start of epoch {epoch + 1}')

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)  # Forward pass
            loss_value = loss_fn(y_batch_train, logits)  # Compute loss

        # Compute gradients and update weights
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        accuracy_metric.update_state(y_batch_train, logits)

        # Logging the loss every 200 steps
        if step % 200 == 0:
            print(f'Epoch {epoch + 1} Step {step}: Loss = {loss_value.numpy()}')

    custom_callback.on_epoch_end(epoch, logs={'loss': loss_value.numpy(), 'accuracy': accuracy_metric.result().numpy()})

    accuracy_metric.reset_state()

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32
)

# Example test data (in practice, use real dataset)
loss, accuracy = model.evaluate(x_test, y_test)

print(f'Test loss:     {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')