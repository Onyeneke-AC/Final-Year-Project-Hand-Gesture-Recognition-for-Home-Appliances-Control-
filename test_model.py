# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:55:01 2023

@author: Hi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Load the saved model
model = keras.models.load_model('explo_model.h5')

fontsize = 22
# Create a test set generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(directory='C:/Users/Hi/Desktop/anthony/Final Year Project/Testing data/test/test',
                                            target_size=(64, 64),
                                            batch_size=1,
                                            color_mode='grayscale',
                                            class_mode='categorical',
                                            shuffle=False)

# Generate predictions for the test set
Y_pred = model.predict_generator(test_set, steps=test_set.samples)

# Convert predictions from probabilities to class labels
y_pred = np.argmax(Y_pred, axis=1)

# Get true labels from the test set generator
y_true = test_set.classes

# Get the class names
class_names = list(test_set.class_indices.keys())

# Print the classification report
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=class_names))

# Print the confusion matrix
print('Confusion Matrix:')
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(15,12))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Confusion Matrix', fontsize = fontsize)
plt.xlabel('Predicted Labels', fontsize = fontsize)
plt.ylabel('True Labels', fontsize = fontsize)
plt.show()

# Plot the precision, recall and f1-score
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
data = {'Precision': [], 'Recall': [], 'F1-score': []}
for label in class_names:
    data['Precision'].append(report[label]['precision'])
    data['Recall'].append(report[label]['recall'])
    data['F1-score'].append(report[label]['f1-score'])

plt.figure(figsize=(15,12))
sns.barplot(x=class_names, y=data['Precision'])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Precision by Class', fontsize = fontsize)
plt.xlabel('Class', fontsize = fontsize)
plt.ylabel('Precision', fontsize = fontsize)
plt.show()

plt.figure(figsize=(15,12))
sns.barplot(x=class_names, y=data['Recall'])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Recall by Class', fontsize = fontsize)
plt.xlabel('Class', fontsize = fontsize)
plt.ylabel('Recall', fontsize = fontsize)
plt.show()

plt.figure(figsize=(15,12))
sns.barplot(x=class_names, y=data['F1-score'])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('F1-score by Class', fontsize = fontsize)
plt.xlabel('Class', fontsize = fontsize)
plt.ylabel('F1-score', fontsize = fontsize)
plt.show()

