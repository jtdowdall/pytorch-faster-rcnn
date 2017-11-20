import os
from os import listdir
from os.path import isfile, join

#print(os.listdir())

writeFileName = 'train.txt'
writeFile = open(writeFileName, 'w')
appended_filenames = [];
for file in os.listdir():
	if '_train.txt' in file:
		currentFile = open(file, 'r');
		currentFile = [line.strip() for line in currentFile.readlines()];
		for line in currentFile:
			imageFileName = line.split()[0] + '\n';
			#print(imageFileName);
			if imageFileName not in appended_filenames:
				writeFile.write(imageFileName);
				appended_filenames.append(imageFileName);
	

writeFileName = 'val.txt'
writeFile = open(writeFileName, 'w')
appended_filenames = [];
for file in os.listdir():
	if '_val.txt' in file:
		currentFile = open(file, 'r');
		currentFile = [line.strip() for line in currentFile.readlines()];
		for line in currentFile:
			imageFileName = line.split()[0] + '\n';
			#print(imageFileName);
			if imageFileName not in appended_filenames:
				writeFile.write(imageFileName);
				appended_filenames.append(imageFileName);
	

writeFileName = 'trainval.txt'
writeFile = open(writeFileName, 'w')
appended_filenames = [];
trainFileName = 'train.txt';
valFileName = 'val.txt';
trainFile = open(trainFileName, 'r').readlines();
valFile = open(valFileName, 'r').readlines();
		
for line in trainFile:		
	imageFileName = line.split()[0] + '\n';
	#print(imageFileName);		
	if imageFileName not in appended_filenames:		
		writeFile.write(imageFileName);	
		appended_filenames.append(imageFileName);
for line in valFile:		
	imageFileName = line.split()[0] + '\n';
	#print(imageFileName);		
	if imageFileName not in appended_filenames:		
		writeFile.write(imageFileName);	
		appended_filenames.append(imageFileName);



