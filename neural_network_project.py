import os
import glob
import numpy as np
from PIL import Image, ImageOps
import time
from shutil import move

import pickle
# import keras
# from keras.utils import to_categorical
# from keras.layers import Dense, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.models import Sequential, load_model
# import tensorflow as tf

from skimage.transform import resize

from sklearn.neural_network import MLPClassifier

model_name = 'finalized_model.sav'
trainData = os.path.join(os.getcwd(), "lfw-deepfunneled")
faceLoc = os.path.join(os.getcwd(), "data")
nonFaceLoc = os.path.join(os.getcwd(), "non_faces")  
data = os.path.join(os.getcwd(), "data")
arrayLoc = os.path.join(os.getcwd(), "arrays")
testImagesLoc = os.path.join(os.getcwd(), "test images")
modelLoc = os.path.join(os.getcwd(), "models")
predictionsLoc = os.path.join(os.getcwd(), "predictions")

def singleFolder(faceLoc, data):
  os.makedirs(data)  
  for root, subdirs, files in os.walk(faceLoc):
    for file in files:
      path = os.path.join(root, file)
      move(path, data)

def convert2Array(arrayLoc, faceLoc, nonFaceLoc, testImagesLoc):
  os.makedirs(arrayLoc)  
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
  processedFaceArray = []
  processedNonFaceArray = []
  # Rescale image to 32x32 size & Flatten image to input into training
  
  for face in greyscaleFaceArray:
    rescaledFace = resize(face,(32,32))
    flattenedFace = np.reshape(rescaledFace, (1,1024))
    processedFaceArray.append(flattenedFace[0])
  
  for nonFace in greyscaleNonFaceArray:
    rescaledNonFace = resize(nonFace, (32,32))
    flattenedNonFace = np.reshape(rescaledNonFace,(1,1024))
    processedNonFaceArray.append(flattenedNonFace[0])

  pickle.dump(processedFaceArray, open(os.path.join(arrayLoc, 'processedFace.npy'), 'wb'))
  pickle.dump(processedNonFaceArray, open(os.path.join(arrayLoc, 'processedNonFace.npy'), 'wb'))
  pickle.dump(greyscaleTestImagesArray, open(os.path.join(arrayLoc, 'greyscaleTestImagesArray.npy'), 'wb'))
  # np.save(os.path.join(arrayLoc, 'processedFace.npy'), processedFaceArray)
  # np.save(os.path.join(arrayLoc, 'processedNonFace.npy'), processedNonFaceArray)
  # np.save(os.path.join(arrayLoc, 'greyscaleTestImagesArray.npy'), greyscaleTestImagesArray)

def trainMlp(processedFaceArray, processedNonFaceArray):
  processedFaceArray = pickle.load(open(os.path.join(arrayLoc, "processedFace.npy")), 'rb')
  processedNonFaceArray = np.load(open(os.path.join(arrayLoc, "processedNonFace.npy")), 'rb')

  facesLabels = np.ones(len(processedFaceArray))
  nonFacesLabels = np.zeros(len(processedNonFaceArray))

  train_Labels = np.concatenate((facesLabels, nonFacesLabels))
  train_data = np.concatenate((processedFaceArray, processedNonFaceArray))
  model = MLPClassifier().fit(train_data, train_Labels)
  filename = 'finalized_model.sav'
  pickle.dump(model, open(filename, 'wb'))
  
def test():
  greyscaleTestImagesArray = pickle.load(open(os.path.join(arrayLoc, "greyscaleTestImagesArray.npy")), 'rb')
  # loaded_model = pickle.load(open(model_name, 'rb'))
  print([len(x) for x in greyscaleTestImagesArray[0]])

def main():
  if not os.path.exists(data):
    singleFolder(trainData, data)
  if not os.path.exists(arrayLoc):
    convert2Array(arrayLoc, faceLoc, nonFaceLoc, testImagesLoc)
  test()



main()

