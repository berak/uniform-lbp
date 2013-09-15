import os,sys

id = 0
for dirname, dirnames, filenames in os.walk(sys.argv[1]):
	for filename in filenames:
		src = os.path.join(dirname, filename)
		print src, id
	if len (filenames)>0:
		id += 1
		

