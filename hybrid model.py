import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0

# EfficientNet as backbone
def create_efficientnet_backbone(input_shape=(224, 224, 3)):
    efficientnet = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    x = GlobalAveragePooling2D()(efficientnet.output)  # Reduce dimensions
    return efficientnet.input, x

# Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Hybrid model
def create_hybrid_model(input_shape=(224, 224, 3), num_classes=6):
    # EfficientNet Backbone
    model_input, efficientnet_output = create_efficientnet_backbone(input_shape)
    
    # Transformer layer
    transformer_block = TransformerBlock(embed_dim=256, num_heads=4, ff_dim=512)
    x = transformer_block(efficientnet_output)
    
    # Dense layers and output
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    
    # Classification output
    class_output = Dense(num_classes, activation="softmax", name="class_output")(x)
    
    # Optional: Regression output for predicting waste percentage (comment if not used)
    percentage_output = Dense(num_classes, activation="sigmoid", name="percentage_output")(x)
    
    # Model
    model = Model(inputs=model_input, outputs=[class_output, percentage_output])
    
    return model

# Compile model
def compile_hybrid_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={"class_output": "categorical_crossentropy", "percentage_output": "mean_squared_error"},
        metrics={"class_output": "accuracy", "percentage_output": "mean_absolute_error"}
    )
    return model

# Create and compile model
hybrid_model = create_hybrid_model(input_shape=(224, 224, 3), num_classes=len(class_labels))
compiled_model = compile_hybrid_model(hybrid_model)
