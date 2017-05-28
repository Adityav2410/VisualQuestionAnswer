import argparse
import os
import sys
import tensorflow as tf
from model import *
from train import trainNetwork
#from test import testNetwork


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p','--qs_path',dest='question_data_path',help='Enter the path for questions data base',default=None,type=str)
	parser.add_argument('-i','--image_path',dest='image_base_path',help='Enter the path for image data base',default=None,type=str)
	parser.add_argument('-b','--batchsize',dest='batchSize',help='Enter the batch size',default=32,type=int)
	parser.add_argument('-e','--epochs',dest='num_epochs',help='Enter the number of epochs',default=10,type=int)
	parser.add_argument('-t','--task',dest='task_to_perform',help="Enter task to perform: train or test")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	
	args = parse_args()
	if not os.path.exists('../logs'):
		os.makedirs('../logs')
	
	with tf.Session() as sess:
		net = Model(sess=sess,question_data_path=args.question_data_path,image_base_path=args.image_base_path,batchSize=args.batchSize)
		net.build_network()
		#net.print_variables()
		#net.write_tensorboard()

		if(args.task_to_perform == 'train'):
			trainNetwork(net,args.num_epochs)

		if(args.task_to_perform == 'test'):
			testNetwork(net)



