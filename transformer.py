import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

# Generate example data

def transformer_model(input_shape, output_shape, d_model=128, num_heads=4, ff_dim=4, dropout=0.1):
    
    inputs = Input(shape=(input_shape,))

    # Embedding layer
    embedding_layer = Embedding(input_dim=input_shape, output_dim=d_model, mask_zero=True)(inputs)

    # Positional encoding
    positions = tf.range(start=0, limit=input_shape, delta=1)
    positions = Embedding(input_dim=input_shape, output_dim=d_model)(positions)
    encoded_inputs = embedding_layer + positions

    # Transformer block
    for _ in range(num_heads):
        x = tfa.layers.MultiHeadAttention(num_heads=num_heads,  head_size=d_model // num_heads)([encoded_inputs, encoded_inputs, encoded_inputs, encoded_inputs])
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x + encoded_inputs)

        y = Dense(ff_dim, activation="relu")(x)
        y = Dropout(dropout)(y)
        y = Dense(d_model, activation="relu")(y)
        encoded_inputs = LayerNormalization(epsilon=1e-6)(x + y)

    # Global Average Pooling
    outputs = GlobalAveragePooling1D()(encoded_inputs)

    # Output layer
    outputs = Dense(output_shape, activation="linear")(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def transformer_trainer(X,y):
    
    num_samples = len(X)
    input_sequence_length = len(X)
    output_sequence_length = len(y)
    model = transformer_model(input_sequence_length, output_sequence_length)

    # Generate random input data

    # Generate corresponding random output data (replace this with your actual data)

    # Train-test split
    split_ratio = 0.8
    split_index = int(num_samples * split_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train the model
    epochs = 10
    batch_size = 32
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    print(f"The model fit is {model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))}")
    summary = model.summary()
    print(f" The model summary is {summary}")
    prediction = model.predict(X)
    print(f"The predicition is {prediction}")
    return fit, summary, prediction




# Example usage:
# Replace input_shape and output_shape with the dimensions of your input and output data


# Compile the model

# Print model summary
