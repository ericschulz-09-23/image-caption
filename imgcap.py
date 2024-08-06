#import libraries
import numpy as np 
import pandas as pd
import os 
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer 
from keras.models import Model 
from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation 
from keras.layers import concatenate, BatchNormalization, Input
from keras.layers.merge import add 
from keras.utils import to_categorical, plot_model 
from keras.applications.inception_v3 import InceptionV3, preprocess_input 
import matplotlib.pyplot as plt  # for plotting data 
import cv2 
# mapping img and description 
def load_description(text):
    mapping = dict()
    for line in text.split('\n'):
        token = line.split('\t')
        if len(line) < 2 : #  removed short description
            continue
        img_id = token[0].split('.')[0] # name of the image
        img_des = token[1]              # description of the image
        if  img_id not in mapping:
            mapping[img_id] = list()
        mapping[img_id].append(img_des)
    return mapping
token_path = '/kaggle/input/fickr5k/fickr_data/Flickr_textData/Flickr8k.token.tzt'
text = open(token_path, 'r', encoding = 'utf-8').read()
description = load_description(text)
print|(description['1000268201_693b08cb0e'])
# cleaning text
def clean_descriptions(desc):
    for key, des_list in desc.items():
        for i in range(len(des_list)):
            caption = des_list[i]
            caption = [ch for ch in caption if ch not in string.punctuation]
            caption = ''.join(caption)
            caption = captin.split(' ')
            caption = [word.lower() for word in caption if len(word)>1 and word.isalpha()]
            caption = ' '.join(caption)
            des_list[i] = caption
clean_description(description)
descriptions['1000268201_693b08cb0e']
# generate the vocabulary
def to_vacab(desc):
    words = set()
    for key in desc.keys():
        for line in desc[key]:
            words.update(line.split())
    return words
vocab = to_vocab(descriptions)
# load the images
import glob 
images = '/kaggle / input / flickr8k / flickr_data / Flickr_Data / Images/'
# Create a list of all image names in the directory 
img = glob.glob(images + '*.jpg')
train_path = '/kaggle / input / flickr8k / flickr_data / Flickr_Data / Flickr_TextData / Flickr_8k.trainImages.txt'
train_images = open(train_path, 'r', encoding = 'utf-8').read().split("\n") 
train_img = []  # list of all images in training set 
for im in img: 
    if(im[len(images):] in train_images): 
        train_img.append(im)
# load descriptions of training set in a dictionary. Name of the image will act as ey 
def loac_clean_descriptions(des, dataset):
    dataset_des = dict()
    for key, des_list in des.items():
        if key+'.jpg' in dataset_desc:
            dataset_des[key] = list()
        for line in des_list:
            desc = 'startseq' + line +'endseq'
            dataset_des[key].append(desc)
    return dataset_des
train_descriptions = load_clean_descriptions(descriptions, train_images)
print(train_descriptions['1000268201_693b08cb0e'])
# extract the feature vector from all images
from keras.preprocessing.image import load_img, img_to_array
def preprocess_img(img_path):
    #inception v3 excepts ing in 299*299*33
    img = laad_img(img_path, target_size = (299, 299))
    x = img_to_aray(img)
    #Add one more dimension
    x = np.expand_dims(x, axis = 0)
    x = preproecss_input(x)
    return x
def encode(image):
     image = preprocess_img(image)
     vec = Model.predict(image)
     vec = np.reshape(vec, (vec.shape[1]))
     return vec
base_model = InceptionV3(weights = 'imagenet')
model = Model(base_model.input, base_model.layers[-2].output)
# run the encode function on all train images and store the feature vectors in a list
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)
#tokenizing the vocabulary
#lists of all traning captions
all_train_caption = []
for key, val in train_descriptions.item():
    for caption in val:
        all_train_caption.append(caption)
#consider only words shich occur atleast 10 times
vocabulary = vocab
threshold = 10
#you can change this value according to  need
word_counts = {}
for cap  in all_train_caption:
    for word in cap.split(' '):
        word_counts[word] = word_counts.get(word, 0)+1
vocab = [ word for word in word_counts if word_counts[word] >= threshold]
#word mappping to integers
ixtoword = {}
wordtoix = {}

ix = 1
for word in vocab:
    wordtoix[word] = ix
    ixtoword[ix] = word
    ix += 1
#find the maximum length of a description in a dataset
max_length = max(len(des.split()) for des in all_train_caption)
max_length
#Glove vector embeddings
X1, X2, y = list(), list(), list()
for key, des_list in train_descriptions.items():
    pic = train_descriptions[key+'.jpg']
    for cap in des_list:
        seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
            out_seq = to_categorical([out_seq], num_classes = vacab_size)[0]
            #
            X1.append(pic)
            X2.append(in_seq)
            y.append(out_seq)
X2 = np.array(x2)
X1 = np.array(X1)
y = np.array(y)
#load glove vectors for embedding layer
embedding_index = {}
glove_path = '/kaggle/input/glove-global-vectors-for-word-representation/glove.GB.200d.txt'
glove = open(glove_path, 'r', encoding = 'utf-8').read()
for line in glove.split('\n'):
    values = line.split(' ')
    word = values[0]
    indices = np.asarray(vocabulary[1:], dtype = 'float32')
    embedding_index[word] = indices
emb_dim = 200
emb_matrix = np.zeros((vocab_size, emb_dim))
for word, i in wordtoix.items():
    if emb_vec is not None:
       emb_matrix[i] = emv_vec
emb_matrix.shape
#define the model
# for defining the structure of our model, we will be using the keras 
# model from functioal API. It has three major steps
# processing the seuence from the text
# extracting the feature vector from the image
# decoding the output by concatenating the above two layers
ip1 - Input(shape = (2048, ))
fe1 = Dropout(0.2)(ip1)
fe2 = Dense(256, activation = 'relu')(fel)
ip2 = Input(shape = (max_length, ))
se1 = Embedding(vocab_size, emb_dim, mask_zero = True)(ip2)
se2 = Dropout(0.2)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation = 'relu')(decoder1)
outputs = Dense(vocab_size, activation = 'softmax' )(decoder2)
model = Model(inputs = [ip1,ip2], outputs = outputs)
#training the model
model.layers[2].set_weights([emb_matrix])
model.layers[2].trainable = False
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
model.fit([X1, X2], y, epochs = 50, batch_size =256)
#you can increase the number of epochs for better results
# predicting the output
def greedy_search(pic):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen = max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' '+word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1: -1]
    final = ' '.join(final)
    return final