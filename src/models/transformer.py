import tensorflow as tf
from tensorflow.keras import layers


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.projection = layers.Dense(d_model)
        self.pos_embedding = layers.Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        x = self.projection(inputs)
        x = x + self.pos_embedding(positions)
        return x


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.dropout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


def build_transformer_model(
    input_shape,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    dropout: float = 0.1,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
):
    inputs = tf.keras.Input(shape=input_shape)
    x = PositionalEmbedding(sequence_length=input_shape[0], d_model=d_model)(inputs)

    for _ in range(num_blocks):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model