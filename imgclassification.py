import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# deep learning libraries
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, maxPooling2D, GlovalAveragePooling2D,
Flatten, Dense, Dropout 
from keras.preprocessing import image
import cv2

import warnings
warnings.filterwarnings('ignore')
#loading datasets and image folders
from google.colab import drive
drive.mount('/content/drive')
#datasets 
labels = pd.read_csv('/content/drive/MyDrive/dog/labels.csv')
sample = pd.read_csv('/content/drive/My Drive/dog/sample_submission.csv')
#folders paths
train_path = "/content/drive/MyDrive/dog/train"
test_path = "/content/drive/MyDrive/dog/test"
labels.head()