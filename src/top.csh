# This is a top level wrapper file - to invoke training or testing

# source top.csh


############################################ Definition of different arguments #################################################################
				# -p: 	path to question answer database
				# -i:	base path to image directory
				# -b:	batch size
				# -e:	number of epochs
				# -t:	'train' or 'test'

# From the two lines below, comment one of the lines - and execute the other one. 



############################################ 1. Train the network ############################################################
python main.py  -p '/home/adityav/ADITYA/Project/Cogs_Image_Caption/VisualQuestionAnswer/data/' -i '/home/adityav/ADITYA/Project/Cogs_Image_Caption/VisualQuestionAnswer/data/' -b  32	-e  1	-t 'train'																		





############################################ 2. Test the network  ############################################################
# python main.py  -p '/home/adityav/ADITYA/Project/Cogs_Image_Caption/VisualQuestionAnswer/data/' -i '/home/adityav/ADITYA/Project/Cogs_Image_Caption/VisualQuestionAnswer/data/' -b  32	-e  1	-t 'train'																		
				