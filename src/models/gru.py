import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_gru_model(
    input_shape,
    gru_units: int = 64,
    dropout: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
    l2_value: float = 1e-5,
):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.GRU(
        gru_units,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(l2_value),
    )(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GRU(
        max(gru_units // 2, 16),
        return_sequences=False,
        kernel_regularizer=regularizers.l2(l2_value),
    )(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs=x, name="gru_forecaster")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model