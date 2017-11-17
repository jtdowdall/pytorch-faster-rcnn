import sys

with open(sys.argv[1], 'r') as f:
	lines = [line.strip().split(' ') for line in f.readlines()]


with open(sys.argv[2], 'w') as f:
	f.write('#filename,region_count,region_id,region_shape_attributes,region_attributes\n')
	for i in range(len(lines)-1):
		line = lines[i]
		filename = line[0] + '.jpg'
		x = int(line[1])
		y = int(line[2])
		width = int(line[3]) - x
		height = int(line[4]) - y
		region_shape_attributes = {
		"name" : "rect",
		"x": x, "y": y,
		"width": width, "height": height}
		region_attributes = {'class' : sys.argv[3]}
		output = filename + \
				',1,0,"{""name"":""rect"",""x"":' + str(x) +\
				',""y"":' + str(y) + \
				',""width"":' + str(width) + \
				',""height"":' + str(height) + \
				'}","{""class"":""' + sys.argv[3] + '""}"''\n'
		f.write(output)

