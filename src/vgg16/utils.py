import skimage
import skimage.io
import skimage.transform
import numpy as np
import os 
import vgg16
import random
import tensorflow as tf
import re
import time

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

def load_Files(dirpath,fileNames):
	# List to store all the image	
	batchList = np.zeros((0,224,224,3))
	# traverse through every file in the fileList
	for fileName in fileNames:
		filepath = os.path.join(dirpath, fileName)				# get the actual path of the files
		img = 	load_image(filepath)				# load the file
		try:
			img_reshape = 	img.reshape((1, 224, 224, 3))
			batchList = np.concatenate((batchList, img_reshape), 0)	# prepare the final list - with all the images
		except ValueError:
			print("Cannot reshape image:\t",fileName, "\t Image shape:",img.shape)
	return(batchList)											# return the numpy array with all the images in the batch


def getImageFeatures(sess,vgg,images,dirpath,batchSize = 100,numImages = -1):
    imageFeatureList = np.zeros((0,14,14,512))
    imageIDList = []
    expr = re.compile(r'(COCO_train2014_000000)(\d+)')
    
    fileList = os.listdir(dirpath)
    random.shuffle(fileList)
    startIndex = 0
    endIndex = 0
    numImages = len(fileList) if numImages == -1 else min(numImages,len(fileList))

    if not len(fileList):
    	print("No image at path: ", dirpath)
    else:
    	while endIndex < numImages:
    		startIndex = endIndex
    		endIndex = min(startIndex+batchSize,numImages)
    		fileNames = fileList[startIndex:endIndex]
    		[batch,batchID] = load_Files(dirpath,fileNames,expr)
    		batch_feature = sess.run(vgg.pool4, feed_dict={images:batch})
    		imageFeatureList = np.concatenate((imageFeatureList,batch_feature),0)
    		imageIDList = imageIDList + batchID
    return[imageFeatureList,imageIDList]

def image2Dict(sess, vgg, images, imagePath, batchSize = 10,numImages = -1):
	startTime = time.time()
	print("Creating dictionary of all the images")
	featureList, idList = getImageFeatures(sess,vgg,images,imagePath,batchSize = batchSize,numImages = numImages)

	imageDict = {}
	for i,id in enumerate(idList):
		imageDict[int(id)] = featureList[i]
	print("Dictionary Created.\nNo of images in dict:{}".format(len(idList)))
	print("Time to create the dictionary:{:.1f}min ".format((time.time()-startTime)/60 ) )
	return(imageDict)

def getVGGhandle():
	sess = tf.Session()
	images = tf.placeholder("float", [None, 224, 224, 3])
	vgg = vgg16.Vgg16()
	with tf.name_scope("content_vgg"):
		vgg.build(images)
	return([sess,vgg,images])