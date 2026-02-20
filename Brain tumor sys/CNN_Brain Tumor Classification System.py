import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import cv2

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
 rescale=1./255,
 rotation_range=20,
 zoom_range=0.2,
 horizontal_flip=True,
 width_shift_range=0.1,
 height_shift_range=0.1
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
 "brain_tumor_dataset/train",
 target_size=(img_size, img_size),
 batch_size=batch_size,
 class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
 "brain_tumor_dataset/test",
 target_size=(img_size, img_size),
 batch_size=batch_size,
 class_mode='categorical',
 shuffle=False
)


custom_model = Sequential([
 Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
 BatchNormalization(),
 MaxPooling2D(2,2),

 Conv2D(64, (3,3), activation='relu'),
 BatchNormalization(),
 MaxPooling2D(2,2),

 Conv2D(128, (3,3), activation='relu'),
 BatchNormalization(),
 MaxPooling2D(2,2),

 Flatten(),
 Dense(128, activation='relu'),
 Dropout(0.5),
 Dense(4, activation='softmax')
])

custom_model.compile(
 optimizer='adam',
 loss='categorical_crossentropy',
 metrics=['accuracy']
)

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history_custom = custom_model.fit(
 train_generator,
 validation_data=test_generator,
  epochs=20,
 callbacks=[early_stop]
)

base_model = MobileNetV2(
 weights='imagenet',
 include_top=False,
 input_shape=(img_size, img_size, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)
transfer_model = Model(inputs=base_model.input, outputs=output)

transfer_model.compile(
 optimizer='adam',
 loss='categorical_crossentropy',
 metrics=['accuracy']
)

history_transfer = transfer_model.fit(
 train_generator,
 validation_data=test_generator,
 epochs=15,
 callbacks=[early_stop]
)

predictions = transfer_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
print(classification_report(y_true, y_pred, target_names=train_generator.class_indices.keys()))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
 xticklabels=train_generator.class_indices.keys(),
 yticklabels=train_generator.class_indices.keys(),
 cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.plot(history_transfer.history['accuracy'], label='Train Acc')
plt.plot(history_transfer.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()




def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
 grad_model = Model(
  [model.inputs],
  [model.get_layer(last_conv_layer_name).output, model.output]
    )

 with tf.GradientTape() as tape:
  conv_outputs, predictions = grad_model(img_array)
  class_idx = tf.argmax(predictions[0])
  loss = predictions[:, class_idx]

  grads = tape.gradient(loss, conv_outputs)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
  conv_outputs = conv_outputs[0]

  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)
  heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

 return heatmap.numpy()

transfer_model.save("brain_tumor_model.h5")
print("Model Saved Successfully!")