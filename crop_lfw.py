import os, sys, string, cv2

f = open("lfw2b.txt","rb")
data = f.read()
f.close()

txt  = open("lfw2fun.txt","wb")
fmax = 400

for r in data.split("\r\n"):
	fn,id=r.split(" ")

	orig = cv2.imread(fn)
	crop = orig[80:90+80, 80:90+80]
	
	name = os.path.split(fn)[-1:][0]
	fn   = "./funneled/" + name 
	txt.write( fn + " " + str(id) + "\r\n" )
	cv2.imwrite(fn,crop)
	fmax -= 1
	if fmax <= 0 : break 
		
txt.close()
