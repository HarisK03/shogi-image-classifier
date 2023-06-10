import tensorflow as tf
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from typing import Tuple


def generate_image_data_generators(batch_size: int, image_height: int, image_width: int) -> Tuple[DirectoryIterator, DirectoryIterator]:
    """
    Generate the train and validation data generators.
    """
    # set up the data paths
    train_data_path = 'data/tiles/'
    validation_data_path = 'data/tiles/'  # use different validation dataset

    # data preprocessing and augmentation
    train_data_generator = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    validation_data_generator = ImageDataGenerator(rescale=1.0/255)

    # load and prepare the training data
    train_data = train_data_generator.flow_from_directory(
        train_data_path,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # load and prepare the validation data
    validation_data = validation_data_generator.flow_from_directory(
        validation_data_path,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_data, validation_data


# train the model
if __name__ == '__main__':
    # set up hyperparameters
    batch_size = 32
    image_height, image_width = 32, 32
    num_epochs = 10

    # retrieve the train_data and validation_data
    generate_image_data_generators, validation_data = generate_image_data_generators(
        batch_size, image_height, image_width)

    # define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=(image_height, image_width, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(
            generate_image_data_generators.num_classes, activation='softmax')
    ])

    # compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # train the model
    model.fit(
        generate_image_data_generators,
        epochs=num_epochs,
        validation_data=validation_data
    )

    # save the trained model
    model.save('shogi_board_to_sfen_model.h5')

    # tile accuracy: 99.94%
    # board line detection: 86.43%
