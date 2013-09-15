    
#updates:

     ^ added WLD, A Robust Detector based on Weber's Law  ( rocks !! )
     ^ ltp and var_lbp
     ^ lfw_funneled database
     ^ combining many small featuresets is quite powerful (combinedLBP), 
       eg, per patch: 
         * 1 3-bit(8 bins) histogram for the pixel center
         * 1 3-bit(8 bins) (8 indices) lpb histogram index of max value
         * 1 4-bit(16 bins) central symmetric (4 corners)   lpb histogram
         * 1 4-bit(16 bins) diagonal tangent (4 corners)   lpb histogram
         concatening those to a 48 bytes patch vector, times 8*8 patches gives 3072 bytes per image ;)
       
     ^ k fold cross validation, please look at results.txt
     ^ added another ref impl, the minmax (surprise)
     ^ this thing has turned into a comparative test of face recognizers
     ^ added a ref impl that just compares pixls
     ^ added uniform version of lbp
