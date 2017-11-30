import sys
import os

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(subdir + "/" + file)                                                                         
    return r         

dir = sys.argv[1].rstrip('/')
out_csv = sys.argv[2]

classes = {}
for f in list_files(dir):
	file = f.split('/')
	if file[-1][-4:] == '.jpg':
		cls = file[-2]
		classes[cls] = '{}/{}/bb_info.txt'.format(dir, cls)

with open(out_csv, 'w') as f:
	f.write('#filename,region_count,region_id,region_shape_attributes,region_attributes\n')

	for c in classes:
		with open(classes[c], 'r') as bbox_info:
			lines = [line.strip().split(' ') for line in bbox_info.readlines()][1:]
			for i in range(len(lines)):
				line = lines[i]
				filename = line[0] + '.jpg'
				x = int(line[1]) + 1
				y = int(line[2]) + 1
				width = int(line[3]) - x + 1
				height = int(line[4]) - y + 1
				output = filename + \
						',1,0,"{""name"":""rect"",""x"":' + str(x) +\
						',""y"":' + str(y) + \
						',""width"":' + str(width) + \
						',""height"":' + str(height) + \
						'}","{""class"":""' + c + '""}"''\n'
				f.write(output)

