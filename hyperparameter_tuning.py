import sys

# Increase recursion limit to prevent potential issues
sys.setrecursionlimit(100000)


# Step 2: Import necessary libraries
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import os
import warnings

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out INFO and WARNING, 3 = ERROR only

# Step 3: Load and preprocess the MNIST dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

print(f'Training data shape: {x_train.shape}')
print(f'Validation data shape: {x_val.shape}')

def build_model(hp):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create a RandomSearch Tuner

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='intro_to_kt'
)

# Display a summary of the search space
tuner.search_space_summary()

# Run the hyperparameter search
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# Display a summary of the results
tuner.results_summary()

# Step 1: Retrieve the best hyperparameters

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f""" 

The optimal number of units in the first dense layer is {best_hps.get('units')}. 

The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. 

""")

# Step 2: Build and Train the Model with Best Hyperparameters
model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_val, y_val)
print(f'Test accuracy: {test_acc}') 