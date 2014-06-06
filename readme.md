

#updates:
     ^ svm (on grayscale pixels)
     ^ Zernike moments
     ^ WLD, A Robust Detector based on Weber's Law  
     ^ ltp and var_lbp
     ^ lfw_funneled database
     ^ combining many small featuresets is quite powerful (combinedLBP),      
     ^ k fold cross validation
     ^ this thing has turned into a comparative test of face recognizers
     ^ added a ref impl that just compares pixls
     ^ added uniform version of lbp

(all code in the master branch is dependant on opencv *master* version, please use the 2.4 branch otherwise)


results vary pretty much with preprocessing:

<p align="center">
  <img src="https://github.com/berak/uniform-lbp/raw/master/img/res_att.png" width=520 height=340>
  <img src="https://github.com/berak/uniform-lbp/raw/master/img/res_yale.png" width=520 height=340>
  <img src="https://github.com/berak/uniform-lbp/raw/master/img/res_lfw.png" width=520 height=340>
</p>


 
