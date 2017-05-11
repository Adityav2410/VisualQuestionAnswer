import skimage
import skimage.io
import skimage.transform
import numpy as np
import os 
import vgg16
import random
import tensorflow as tf

# synset = [l.strip() for l in open('synset.txt').readlines()]


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
	batchFileNames = []
	# traverse through every file in the fileList
	for fileName in fileNames:
		filepath = os.path.join(dirpath, fileName)				# get the actual path of the files
		img = 			load_image(filepath)				# load the file
		try:
			img_reshape = 	img.reshape((1, 224, 224, 3))
			batchFileNames.append(fileName)
		except ValueError:
			print("Cannot reshape image:\t",fileName, "\t Image shape:",img.shape)
			

		batchList = np.concatenate((batchList, img_reshape), 0)	# prepare the final list - with all the images

	return(batchList,batchFileNames)											# return the numpy array with all the images in the batch


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)



def getVGGhandle():
	sess = tf.Session()
	images = tf.placeholder("float", [None, 224, 224, 3])
	vgg = vgg16.Vgg16()
	with tf.name_scope("content_vgg"):
		vgg.build(images)
	return([sess,vgg,images])

def getImageFeatures(sess,vgg,images,dirpath,batchSize = 100,numImages = -1):
    imageFeatureList = np.zeros((0,14,14,512))
    featureFileList = []
    
    fileList = os.listdir(dirpath)
    random.shuffle(fileList)
    startIndex = 0
    endIndex = 0
    numImages = len(fileList) if numImages == -1 else min(numImages,len(fileList))

    if not len(fileList):
        print("No image at path ",dirpath)
    else:
		
		#images = tf.placeholder("float", [None, 224, 224, 3])
		while endIndex < numImages:
			startIndex = endIndex
			endIndex = min(startIndex+batchSize , numImages)
    		fileNames = fileList[startIndex:endIndex]

    		[batch,batchFileNames] = load_Files(dirpath, fileNames)
    		batch_feature = sess.run(vgg.pool4, feed_dict={images:batch})

    		imageFeatureList = np.concatenate((imageFeatureList,batch_feature),0)
    		featureFileList = featureFileList + batchFileNames

    print("Number of image features in the list:", imageFeatureList.shape[0])
    return [imageFeatureList,featureFileList]






def getImageFeatures1(dirpath,batchSize = 100,numImages = -1):
    imageFeatureList = np.zeros((0,14,14,512))
    featureFileList = []
    
    fileList = os.listdir(dirpath)
    random.shuffle(fileList)
    startIndex = 0
    endIndex = 0
    numImages = len(fileList) if numImages == -1 else min(numImages,len(fileList))

    if not len(fileList):
        print("No image at path ",dirpath)
    else:
        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                images = tf.placeholder("float", [None, 224, 224, 3])
                vgg = vgg16.Vgg16()
                with tf.name_scope("content_vgg"):
                    vgg.build(images)

                while endIndex < numImages:
                    startIndex = endIndex
                    endIndex = min(startIndex+batchSize , numImages)
                    fileNames = fileList[startIndex:endIndex]

                    [batch,batchFileNames] = load_Files(dirpath, fileNames)
                    batch_feature = sess.run(vgg.pool4, feed_dict={images:batch})

                    imageFeatureList = np.concatenate((imageFeatureList,batch_feature),0)
                    featureFileList = featureFileList + batchFileNames

    print("Number of image features in the list:", imageFeatureList.shape[0])
    return [imageFeatureList,featureFileList]




if __name__ == "__main__":
    test()
