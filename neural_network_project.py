# -*- coding: utf-8 -*-
"""Neural Network Project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16IRuH2lRgVIuc3A4i1_gTNLJGTdML1uB
"""

import os
import glob
import numpy as np
from PIL import Image, ImageOps
import time
from shutil import copyfile, move

import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import keras
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
import tensorflow as tf

from skimage.transform import resize

from sklearn.neural_network import MLPClassifier



def singleFolder(faceLoc, data):
  if not os.path.exists(data):
    os.makedirs(data)
  
  for root, subdirs, files in os.walk(faceLoc):
    for file in files:
      path = os.path.join(root, file)
      move(path, data)

def convert2Array(arrayLoc, faceLoc, nonFaceLoc, testImagesLoc):
  if not os.path.exists(arrayLoc):
    os.makedirs(arrayLoc)
  # if not os.path.exists(nonFaceArrayLoc):
  #   os.makedirs(faceArrayLoc)
  # if not os.path.exists(testImagesArrayLoc):
  #   os.makedirs(testImagesArrayLoc)

  filelist = glob.glob(faceLoc + "/*.jpg")
  nonFaceList = glob.glob(nonFaceLoc + "/*.jpg")
  testList = glob.glob(testImagesLoc + "/*.jpg")
  
  # Remove colour information with ImageOps.grayscale
  # Using np.array to store image data as a numpy array
  
  greyscaleFaceArray = np.array([np.array(ImageOps.grayscale(Image.open(image))) 
                               for image in filelist])

  greyscaleNonFaceArray = np.array([np.array(ImageOps.grayscale(Image.open(image))) 
                               for image in nonFaceList])
  greyscaleTestImagesArray = np.array([np.array(ImageOps.grayscale(Image.open(image))) 
                               for image in testList])
  print(len(greyscaleFaceArray))
  processedFaceArray = []
  for face in greyscaleFaceArray:
    # print(type(face))
    rescaledFace = resize(face,(32,32))
    flattenedFace = np.reshape(rescaledFace, (1,1024))
    processedFaceArray.append(flattenedFace[0])
  processedNonFaceArray = []
  for nonFace in greyscaleNonFaceArray:
    rescaledNonFace = resize(nonFace, (32,32))
    flattenedNonFace = np.reshape(rescaledNonFace,(1,1024))
    processedNonFaceArray.append(flattenedNonFace[0])

  print(len(processedFaceArray))
  print(processedFaceArray[0])
  print(len(processedFaceArray[0]))


  return processedFaceArray, processedNonFaceArray, greyscaleTestImagesArray

  





  # print(processedFaceArray)
  # print(len(processedFaceArray))

  
  np.save(os.path.join(arrayLoc, 'processedFace.npy'), processedFaceArray)
  np.save(os.path.join(arrayLoc, 'processedNonFace.npy'), processedNonFaceArray)  
  np.save(os.path.join(arrayLoc, 'processedTestImages.npy'), greyscaleTestImagesArray)

def cnn(processedFaceLoc, processedNonFaceLoc, modelLoc, predictionsLoc, testImagesArray):
  if not os.path.exists(modelLoc):
    os.makedirs(modelLoc)
  if not os.path.exists(predictionsLoc):
    os.makedirs(predictionsLoc)
  
  batchSize = 128
  numClasses = 2
  epochs = 2

  imgX, imgY = 32, 32

  x_face = np.load(processedFaceLoc)
  y_face = np.ones(13233)
  x_non_face = np.load(processedNonFaceLoc)
  y_non_face = np.zeros(10000)
  
  x = np.concatenate((x_face, x_non_face), axis=0)
  y = np.concatenate((y_face, y_non_face), axis=0)

  print(x)
  print(y)  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8873)
  # matrix_y_test = y_test

  x_train = x_train.reshape(x_train.shape[0], imgX, imgY, 1)
  x_test = x_test.reshape(x_test.shape[0], imgX, imgY, 1)
  inputShape = (imgX, imgY, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  y_train = to_categorical(y_train, numClasses)
  y_test = to_categorical(y_test, numClasses)

  model = Sequential()

  # Convolutional Layers
  # 1st layer
  model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1),
                   activation='relu', input_shape=inputShape))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  # 2nd layer
  model.add(Conv2D(32, (7, 7), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  # 3rd layer
  model.add(Conv2D(32, (7, 7), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  # Flatten the output
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(numClasses, activation='softmax'))

  # Configures the model for training.
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

  # Train model + Time
  # start = time.time()
  # model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, verbose=1,
  #           validation_data=(x_test, y_test))
  # timeTaken = time.time() - start
  # print("Time taken: {}".format(timeTaken))

  # # Test model
  # score = model.evaluate(x_test, y_test, verbose=1)

  # # Save model
  # model.save(os.path.join(modelLoc, "CNN_model.h5"))

  # Use model to predict on images in test-images
  # predictions = model.predict_classes(testImagesArray, verbose=1)
  model = load_model(os.path.join(modelLoc, "CNN_model.h5"))
  testArray = np.load(testImagesArray)
  predictions = []
  for i in range(len(testArray[0])):
    for j in testArray[0][i]:
      predictions.append(np.argmax(model.predict(j), axis=-1))

  np.savetxt(os.path.join(predictionsLoc, 'predictions2.txt'), predictions, fmt="%d")
  
  # Confusion matrix
  y_pred = np.loadtxt(os.path.join(predictionsLoc, "predictions.txt"))
  print(confusion_matrix(groundtruth, y_pred))

def trainMlp(processedFaceArray, processedNonFaceArray):
  facesLabels = np.ones(len(processedFaceArray))
  nonFacesLabels = np.zeros(len(processedNonFaceArray))
  train_Labels = np.concatenate((facesLabels, nonFacesLabels))
  train_data = np.concatenate((processedFaceArray, processedNonFaceArray))
  model = MLPClassifier().fit(train_data, train_Labels)
  filename = 'finalized_model.sav'
  pickle.dump(model, open(filename, 'wb'))

  return model



def main():
  trainData = os.path.join(os.getcwd(), "lfw-deepfnneled")
  faceLoc = os.path.join(os.getcwd(), "data")
  nonFaceLoc = os.path.join(os.getcwd(), "non_faces")  
  data = os.path.join(os.getcwd(), "data")
  testImagesLoc = os.path.join(os.getcwd(), "test images")

  arrayLoc = os.path.join(os.getcwd(), "arrays")
  faceArray = os.path.join(arrayLoc,"processedFace.npy")
  nonFaceArray = os.path.join(arrayLoc, "processedNonFace.npy")
  testImagesArray = os.path.join(arrayLoc, "processedTestImages.npy")

  modelLoc = os.path.join(os.getcwd(), "models")
  predictionsLoc = os.path.join(os.getcwd(), "predictions")

  singleFolder(trainData, data)
  processedFaceArray, processedNonFaceArray, greyscaleTestImagesArray = convert2Array(arrayLoc, faceLoc, nonFaceLoc, testImagesLoc)
  # cnn(faceArray, nonFaceArray, modelLoc, predictionsLoc, testImagesArray)
  model = trainMlp(processedFaceArray, processedNonFaceArray)

def test():
  filename = 'finalized_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))


main()
