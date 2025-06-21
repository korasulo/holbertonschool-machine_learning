import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model.
    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR-10 data
        Y: numpy.ndarray of shape (m,) containing CIFAR-10 labels
    Returns:
        X_p: preprocessed X
        Y_p: one-hot encoded labels
    """
    Y_p = to_categorical(Y, 10)
    return X, Y_p


def resize_generator(generator, images, labels, batch_size):
    """
    Generator that resizes CIFAR-10 images to match model input size.
    """
    while True:
        for batch_x, batch_y in generator.flow(images, labels, batch_size=batch_size):
            batch_x_resized = tf.image.resize(batch_x, [224, 224])
            yield batch_x_resized, batch_y


def main():
    # Hyperparameters
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 30

    # Load and preprocess CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Train and test generators
    train_generator = resize_generator(train_datagen, x_train, y_train, batch_size=BATCH_SIZE)
    test_generator = resize_generator(test_datagen, x_test, y_test, batch_size=BATCH_SIZE)

    # Load EfficientNetB0 base model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Initial training
    model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        validation_data=test_generator,
        validation_steps=len(x_test) // BATCH_SIZE,
        epochs=EPOCHS,
    )

    # Fine-tune last 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fine-tune training
    model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        validation_data=test_generator,
        validation_steps=len(x_test) // BATCH_SIZE,
        epochs=EPOCHS,
        initial_epoch=EPOCHS
    )

    # Save model
    model.save('cifar10.h5')


if __name__ == '__main__':
    main()
