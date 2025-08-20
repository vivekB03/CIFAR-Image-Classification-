import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display sample image
print("Sample training image:")
plt.imshow(train_images[0])
plt.title(f"Label: {class_names[train_labels[0][0]]}")
plt.show()

# Normalize pixel values to [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential([
    # Feature extraction layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    # Classification layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Display model summary
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training the model...")
history = model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

predictions = model.predict(test_images)

# Display first test image with prediction
plt.imshow(test_images[0])
predicted_name = class_names[np.argmax(predictions[0])]
actual_name = class_names[test_labels[0][0]]

plt.title(f"Prediction: {predicted_name}\nActual: {actual_name}")
plt.show()

# Print prediction result
print(f"Model prediction: {predicted_name}")
print(f"Actual label: {actual_name}")
if predicted_name == actual_name:
    print("Correct classification!")
else:
    print("Incorrect classification.")