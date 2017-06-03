import argparse
import os
import sys
import tensorflow as tf
from model import *
from train import trainNetwork
from config import Config
#from test import testNetwork
from pdb import set_trace as bp

def parse_args(C):
	parser = argparse.ArgumentParser()
	parser.add_argument('-p','--qs_path',dest='question_data_path',help='Enter the path for questions data base',default=None,type=str)
	parser.add_argument('-i','--image_path',dest='image_base_path',help='Enter the path for image data base',default=None,type=str)
	parser.add_argument('-m','--model_path',dest='model_dir',help='Enter the path from where to save/restore the model',default=None,type=str)
	parser.add_argument('-b','--batchsize',dest='batchSize',help='Enter the batch size',default=None,type=int)
	parser.add_argument('-e','--epochs',dest='num_epochs',help='Enter the number of epochs',default=None,type=int)
	parser.add_argument('-t','--task',dest='task_to_perform',help="Enter task to perform: train or test",default=None,type=str)
	args = parser.parse_args()

	if args.question_data_path:
		C.question_data_path = args.question_data_path
	if args.image_base_path:
		C.image_base_path = args.image_base_path
	if args.model_dir:
		C.model_dir = args.model_dir

	if args.batchSize:
		C.batchSize = args.batchSize
	if args.num_epochs:
		C.num_epochs = args.num_epochs
	if args.task_to_perform:
		C.task_to_perform = args.task_to_perform
	# return args


if __name__ == '__main__':
	
	C = Config()
	parse_args(C)
	# args = parse_args(C)
	if not os.path.exists('../logs'):
		os.makedirs('../logs')


	with tf.Session() as sess:

		###################### Load model #######################
		# bp() 
		net = Model(sess=sess,batchSize=C.batchSize,C = C)
		net.build_network()

		# either restore model or initialize model
		saver_all = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(C.model_dir)
		# if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		if ckpt and ckpt.model_checkpoint_path:
			print "Restoring Model from ", ckpt.model_checkpoint_path
			saver_all.restore(sess, ckpt.model_checkpoint_path)
		else:
			print "Initializing Model"
			sess.run(tf.global_variables_initializer())
		
		if(C.verbose):
			net.print_variables()
		net.write_tensorboard()


		if(C.task_to_perform == 'train'):
			trainNetwork(sess, net,C.num_epochs, C, saver_all)
			# saver_all.save(sess,'checkpoints/vqa')

		# if(args.task_to_perform == 'test'):
		# 	testNetwork(sess, net)



