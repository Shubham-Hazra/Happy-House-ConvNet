import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras

train_dataset = h5py.File('train_happy.h5', "r")
x_train = np.array(train_dataset["train_set_x"][:])
y_train = np.array(train_dataset["train_set_y"][:])
classes = np.array(train_dataset["list_classes"][:])

test_dataset = h5py.File('test_happy.h5', "r")
x_test = np.array(test_dataset["test_set_x"][:])
y_test = np.array(test_dataset["test_set_y"][:])

# Normalize image vectors
x_train = x_train/255.0
x_test = x_test/255.0

# Reshape
y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))


# Define the model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(64, 64, 3)),
    keras.layers.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001),
    keras.layers.Conv2D(32, (7, 7), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=16,
          validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

# Print the results
print('Test accuracy:', test_acc*100, '%')

# Save the model
model.save('happy_house.h5')

# Load the model
model = keras.models.load_model('happy_house.h5')

# Predict
random_image = np.random.randint(0, x_test.shape[0])
plt.imshow(x_test[random_image])
prediction = np.round(model.predict(
    x_test[random_image].reshape(1, 64, 64, 3)))[0][0]
prediction = "Happy" if prediction == 1 else "Not Happy"
print('Prediction:', prediction)
plt.show()
