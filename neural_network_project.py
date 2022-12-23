import os
import glob
import json
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

  # pickle.dump(processedFaceArray, open(os.path.join(arrayLoc, 'processedFace.npy'), 'wb'))
  # pickle.dump(processedNonFaceArray, open(os.path.join(arrayLoc, 'processedNonFace.npy'), 'wb'))
  # pickle.dump(greyscaleTestImagesArray, open(os.path.join(arrayLoc, 'greyscaleTestImagesArray.npy'), 'wb'))
  np.save(os.path.join(arrayLoc, 'processedFace.npy'), processedFaceArray)
  np.save(os.path.join(arrayLoc, 'processedNonFace.npy'), processedNonFaceArray)
  np.save(os.path.join(arrayLoc, 'greyscaleTestImagesArray.npy'), greyscaleTestImagesArray)

def trainMlp():
  processedFaceArray = np.load(os.path.join(arrayLoc, 'processedFace.npy'))
  processedNonFaceArray = np.load(os.path.join(arrayLoc, "processedNonFace.npy"))

  facesLabels = np.ones(len(processedFaceArray))
  nonFacesLabels = np.zeros(len(processedNonFaceArray))

  train_Labels = np.concatenate((facesLabels, nonFacesLabels))
  train_data = np.concatenate((processedFaceArray, processedNonFaceArray))

  startTime = time.time()
  model = MLPClassifier().fit(train_data, train_Labels)
  timeTaken = time.time() - startTime
  print("Training time taken: %s seconds"%timeTaken)
  pickle.dump(model, open(model_name, 'wb'))
  
def test():
  if not os.path.exists(predictionsLoc):
    os.mkdirs(predictionsLoc)
  greyscaleTestImagesArray = np.load(os.path.join(arrayLoc, "greyscaleTestImagesArray.npy"), allow_pickle=True)
  loaded_model = pickle.load(open(model_name, 'rb'))

  # Input array is a flattened array of a 32x32 sample of a test image
  # First row of 32 pixels, then second row of 32 pixels, etc... [1,2,3,...32,33,34,...1024]

  # allPred = []
  # for imageNo in range(len(greyscaleTestImagesArray)):
  #   imagePred = []
  #   for dshift in range(len(greyscaleTestImagesArray[imageNo])-31):
  #     for rshift in range(len(greyscaleTestImagesArray[imageNo][0])-31):
  #       inputArray = []
  #       for i in range(32):
  #         for j in range(32):
  #           inputArray.append(greyscaleTestImagesArray[imageNo][i+dshift][j+rshift])
  #       inputArray = np.array(inputArray)
  #       imagePred.append(int(loaded_model.predict(inputArray.reshape(1, -1)).tolist()[0]))
  #   allPred.append(imagePred)
  # print(allPred)

  allPred = {}
  startTime = time.time()
  for imageNo in range(len(greyscaleTestImagesArray)):
    imagePred = []
    for dshift in range(len(greyscaleTestImagesArray[imageNo])//32):
      for rshift in range(len(greyscaleTestImagesArray[imageNo][0])//32):
        inputArray = []
        for i in range(32):
          for j in range(32):
            inputArray.append(greyscaleTestImagesArray[imageNo][i+dshift*32][j+rshift*32])
        inputArray = np.array(inputArray)
        imagePred.append(int(loaded_model.predict(inputArray.reshape(1, -1)).tolist()[0]))
    allPred["Image {}".format(imageNo+1)] = imagePred
  timeTaken = time.time() - startTime
  print("Testing time taken: %s seconds"%timeTaken)
  # print(allPred)
  
  with open(os.path.join(predictionsLoc, "pred.json"), 'w') as f:
    json.dump(allPred, f, indent=4)

  confusionMatrix()
# Place groundtruth.json in the main folder
# crop test images into resolutions divisible by 32
def confusionMatrix():
  tp, fp, fn, tn = 0, 0, 0, 0
  cf = {}
  with open(os.path.join(os.getcwd(), "groundtruth.json"), "r") as f:
    groundTruth = json.load(f)
    with open(os.path.join(predictionsLoc, "pred.json"), "r") as g:
      predFile = json.load(g)
      for imageNo in range(1, 8):
        for idx, pred in enumerate(predFile["Image {}".format(imageNo)]):
          if pred == 1 and groundTruth["Image {}".format(imageNo)][idx] == 1:
            tp += 1
          elif pred == 0 and groundTruth["Image {}".format(imageNo)][idx] == 0:
            tn += 1
          elif pred == 1 and groundTruth["Image {}".format(imageNo)][idx] == 0:
            fp += 1
          else:
            fn += 1
        cf["Image {}".format(imageNo)] = {"TP":tp, "FP":fp, "FN":fn, "TN":tn}
  
  with open(os.path.join(predictionsLoc, "cf.json"), "w") as h:
    json.dump(cf, h, indent=4)

def main():
  if not os.path.exists(data):
    singleFolder(trainData, data)
  if not os.path.exists(arrayLoc):
    convert2Array(arrayLoc, faceLoc, nonFaceLoc, testImagesLoc)
  # trainMlp()
  if not os.path.exists(os.path.join(os.getcwd(), model_name)):
    trainMLP()
  # test()
  confusionMatrix()


main()

