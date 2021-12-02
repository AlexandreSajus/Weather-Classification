import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D


def generate_resnet(mode: str = "binary") -> Model:
    """
    Generates a non-trainable pretrained resnet50 model
    with two trainable Dense layers on top.

    Params:
        mode: "binary" or "categorical"

    Returns:
        model: resnet model
    """

    inputs = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
    float_inputs = tf.cast(inputs, tf.float32)
    preprocessed = tf.keras.applications.mobilenet.preprocess_input(
        float_inputs)

    model = ResNet50(include_top=False, weights='imagenet')
    for layer in model.layers:
        layer.trainable = False
    features = model(preprocessed)

    flat1 = GlobalAveragePooling2D()(features)
    dense1 = Dense(64, activation='relu')(flat1)
    dense2 = Dense(64, activation='relu')(dense1)
    if mode == "categorical":
        output = Dense(3, activation='softmax')(dense2)
    elif mode == "binary":
        output = Dense(1, activation='sigmoid')(dense2)
    else:
        raise ValueError("mode must be 'binary' or 'categorical'")

    model = Model(inputs=inputs, outputs=output)

    return model


def generate_CNN(mode: str = "binary") -> Model:
    """
    Generates a CNN model

    Params:
        mode: "binary" or "categorical"

    Returns:
        model: CNN model
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu')])

    if mode == "categorical":
        model.add(tf.keras.layers.Dense(3, activation='softmax'))
    elif mode == "binary":
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    else:
        raise ValueError("mode must be 'binary' or 'categorical'")

    return model
