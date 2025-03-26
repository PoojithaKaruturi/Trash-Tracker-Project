import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define paths to your dataset (update paths accordingly)
dataset_path = r"C:\\miniproject - Copy\\Garbage classification\\Garbage classification"  # e.g., '/content/dataset-resized'
image_size = (224, 224)
batch_size = 32
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,   # Scale pixel values
    validation_split=0.2,  # Split for training and validation
    horizontal_flip=True,  # Augmentation
    rotation_range=30,     # Augmentation
    zoom_range=0.2         # Augmentation
)

# Load training data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, LayerNormalization, MultiHeadAttention,Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0

# EfficientNet as backbone
def create_efficientnet_backbone(input_shape=(224, 224, 3)):
    efficientnet = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    x = GlobalAveragePooling2D()(efficientnet.output)  # Reduce dimensions
    return efficientnet.input, x

# Transformer block

# Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),  # First layer with 256 units
    tf.keras.layers.Dense(1280)  # Change this to 1280 to match out1
])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def build(self, input_shape):
        # Here you can define any layer variables if needed
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training):
        # Expand dimensions to match transformer input requirements
        inputs = tf.expand_dims(inputs, axis=1)  # Add a sequence dimension

        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda,Dropout, Flatten, Reshape

import tensorflow as tf

def create_hybrid_model(input_shape=(224, 224, 3), num_classes=6):
    # EfficientNet Backbone
    model_input, efficientnet_output = create_efficientnet_backbone(input_shape)
    
    # Transformer layer
    transformer_block = TransformerBlock(embed_dim=256, num_heads=4, ff_dim=512)
    
    # Pass the output through the transformer block
    x = transformer_block(efficientnet_output, training=True)  # Set training=True during training

    # Squeeze to remove dimensions of size 1
    x = Lambda(lambda t: tf.squeeze(t, axis=1))(x)  # This will convert shape (None, 1, 1280) to (None, 1280)
    
    # Ensure the output shape is correct
    if len(x.shape) == 2:  # Only if x has the shape (None, 1280)
        x = Flatten()(x)

    # Dense layers
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    
    # Output layer for multi-class classification
    class_output = Dense(num_classes, activation="softmax", name="class_output")(x)  # Softmax for multi-class
    
    # Optional: Regression output for predicting waste percentage
    percentage_output = Dense(num_classes, activation="sigmoid", name="percentage_output")(x)  # Sigmoid for percentages
    
    # Create the model
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
