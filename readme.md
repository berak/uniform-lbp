

#updates:

     ^ added negative tests to get tp/fp accuracy
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
       
     ^ k fold cross validation
     ^ added another ref impl, the minmax (surprise)
     ^ this thing has turned into a comparative test of face recognizers
     ^ added a ref impl that just compares pixls
     ^ added uniform version of lbp

(all code in the master branch is dependant on opencv *master* version, please use the 2.4 branch otherwise)

#results:

//
// method  mean_err   accuracy     t_train t_test (in seconds)
//

yale.txt 
10 fold, 15 subjects, 165 images, 11 per person (1 images skipped)
fisher       0.060      0.928      (117.784  1.059)
eigen        0.079      0.898      (240.078  9.782)
lbph         0.041      0.950      (81.595 13.345)
lbph2_u      0.031      0.965      (55.175  6.243)
minmax       0.051      0.945      (14.496  1.515)
lbp_comb     0.046      0.950      ( 9.840  1.465)
lbp_var      0.072      0.900      ( 5.586  0.431)
ltph         0.037      0.957      (25.639  4.374)
clbpdist     0.021      0.975      (47.852  5.199)
wld          0.053      0.940      (33.430  2.380)
norml2       0.073      0.910      ( 6.776  1.507)

att.txt 
9 fold, 41 subjects, 399 images, 9 per person (1 images skipped)
fisher       0.024      0.966      (735.535  6.724)
eigen        0.016      0.978      (715.585 16.573)
lbph         0.018      0.976      (366.704 80.808)
lbph2_u      0.017      0.977      (188.237 34.199)
minmax       0.018      0.976      (42.472  6.540)
lbp_comb     0.009      0.989      (36.771  7.479)
lbp_var      0.069      0.898      (12.654  1.376)
ltph         0.047      0.932      (104.203 23.134)
clbpdist     0.035      0.953      (138.576 22.444)
wld          0.005      0.994      (86.393  8.366)
norml2       0.016      0.978      (34.432  8.599)

tv.txt
10 fold, 24 subjects, 400 images, 16 per person
fisher       0.008      0.990      (944.188  2.422)
eigen        0.008      0.991      (931.141 11.853)
lbph         0.006      0.994      (303.518 55.257)
lbph2_u      0.006      0.993      (174.013 24.111)
minmax       0.004      0.994      (40.788  4.653)
lbp_comb     0.004      0.995      (32.728  5.384)
lbp_var      0.019      0.974      (13.339  0.983)
ltph         0.012      0.985      (90.933 16.811)
clbpdist     0.011      0.986      (133.428 15.980)
wld          0.004      0.994      (92.147  5.851)
norml2       0.008      0.991      (27.873  6.181)


f96.txt
10 fold, 21 subjects, 360 images, 17 per person (40 images skipped)
fisher       0.003      0.998      (741.877  2.198)
eigen        0.007      0.995      (740.477 11.278)
lbph         0.007      0.994      (321.306 58.243)
lbph2_u      0.007      0.994      (174.229 24.185)
minmax       0.003      0.998      (40.632  4.587)
lbp_comb     0.004      0.997      (32.287  5.221)
lbp_var      0.029      0.964      (13.023  0.985)
ltph         0.006      0.996      (88.539 16.056)
clbpdist     0.004      0.998      (129.291 15.697)
wld          0.002      0.999      (89.126  5.896)
norml2       0.007      0.995      (27.383  5.904)

crop.txt
10 fold, 27 subjects, 400 images, 14 per person
fisher       0.088      0.860      (1106.729  3.218)
eigen        0.083      0.869      (1057.881 13.400)
lbph         0.072      0.887      (352.976 65.788)
lbph2_u      0.071      0.890      (192.843 27.555)
minmax       0.080      0.876      (45.547  5.249)
lbp_comb     0.053      0.919      (36.603  6.123)
lbp_var      0.160      0.743      (14.457  1.084)
ltph         0.127      0.799      (100.011 18.756)
clbpdist     0.139      0.782      (145.503 17.929)
wld          0.042      0.937      (99.440  6.521)
norml2       0.083      0.869      (31.513  6.998)

lfw2fun.txt
10 fold, 20 subjects, 400 images, 20 per person
fisher       0.091      0.855      (998.585  2.146)
eigen        0.139      0.779      (1038.049 12.209)
lbph         0.134      0.789      (305.121 55.835)
lbph2_u      0.128      0.798      (176.088 24.765)
minmax       0.098      0.843      (42.261  4.762)
lbp_comb     0.125      0.803      (33.608  5.529)
lbp_var      0.206      0.669      (13.738  0.986)
ltph         0.163      0.742      (92.902 17.165)
clbpdist     0.107      0.829      (137.154 16.312)
wld          0.090      0.859      (96.062  5.975)
norml2       0.139      0.780      (29.162  6.445)

