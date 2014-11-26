git clone https://github.com/berak/uniform-lbp
cd uniform-lbp/challenge
curl http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz > lfw.tgz
tar -xf lfw.tgz
cd lfw-deepfunneled
curl http://vis-www.cs.umass.edu/lfw/pairs.txt > pairs.txt
curl http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt > pairsDevTrain.txt
cd ..
rm -f  lfw.tgz

bash makefile
