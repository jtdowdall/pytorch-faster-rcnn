import os
from os import listdir
from os.path import isfile, join

print(os.listdir())

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
	


