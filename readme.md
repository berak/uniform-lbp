

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
       
     ^ k fold cross validation
     ^ added another ref impl, the minmax (surprise)
     ^ this thing has turned into a comparative test of face recognizers
     ^ added a ref impl that just compares pixls
     ^ added uniform version of lbp

(all code is dependant on opencv *master* version)

#results:

    //
    // method  mean_err   hitrate     t_train t_test (in seconds)
    //

    yale.txt
    10 fold, 15 subjects, 165 images, 11 per person (1 images skipped)
    fisher     0.060      0.940      (114.910  0.536)
    eigen      0.079      0.921      (214.872  4.888)
    lbph       0.041      0.959      (50.229  5.962)
    lbph2_u    0.031      0.969      (39.586  2.893)
    minmax     0.051      0.949      ( 9.028  0.659)
    lbp_comb   0.044      0.956      ( 5.429  0.556)
    lbp_var    0.072      0.928      ( 4.124  0.182)
    ltph       0.039      0.961      (15.625  2.256)
    clbpdist   0.016      0.984      (28.300  4.672)
    wld        0.053      0.947      (29.046  1.223)
    norml2     0.073      0.927      ( 3.348  0.742)

    att.txt
    10 fold, 40 subjects, 400 images, 10 per person
    fisher     0.023      0.977      (909.497  3.560)
    eigen      0.014      0.986      (866.771  8.563)
    lbph       0.018      0.982      (235.187 38.257)
    lbph2_u    0.017      0.983      (141.111 16.863)
    minmax     0.017      0.983      (31.850  3.272)
    lbp_comb   0.006      0.994      (21.982  3.061)
    lbp_var    0.067      0.933      (10.553  0.675)
    ltph       0.045      0.955      (73.943 12.845)
    clbpdist   0.036      0.964      (144.987 27.005)
    wld        0.005      0.995      (71.952  4.252)
    norml2     0.015      0.985      (21.243  4.522)

    tv.txt 
    10 fold, 24 subjects, 400 images, 16 per person
    fisher     0.008      0.992      (924.194  1.186)
    eigen      0.008      0.992      (873.906  5.931)
    lbph       0.006      0.994      (168.802 24.619)
    lbph2_u    0.006      0.994      (114.199 11.177)
    minmax     0.004      0.996      (26.063  2.168)
    lbp_comb   0.004      0.996      (16.927  2.068)
    lbp_var    0.019      0.981      ( 9.592  0.447)
    ltph       0.012      0.988      (52.443  8.665)
    clbpdist   0.011      0.989      (100.155 18.068)
    wld        0.004      0.996      (101.845  5.510)
    norml2     0.008      0.992      (13.792  3.051)

    faces96.txt
    10 fold, 21 subjects, 360 images, 17 per person (40 images skipped)
    fisher     0.003      0.997      (740.387  1.096)
    eigen      0.007      0.993      (686.877  5.658)
    lbph       0.007      0.993      (176.189 26.173)
    lbph2_u    0.007      0.993      (114.009 11.315)
    minmax     0.003      0.997      (25.986  2.145)
    lbp_comb   0.003      0.997      (16.546  1.994)
    lbp_var    0.029      0.971      ( 9.076  0.441)
    ltph       0.007      0.993      (50.860  8.208)
    clbpdist   0.003      0.997      (97.155 17.250)
    wld        0.002      0.998      (78.835  3.012)
    norml2     0.007      0.993      (13.590  2.928)

    crop.txt
    10 fold, 27 subjects, 400 images, 14 per person
    fisher     0.088      0.912      (1103.790  1.623)
    eigen      0.083      0.917      (1061.755  6.659)
    lbph       0.072      0.928      (190.804 29.116)
    lbph2_u    0.071      0.929      (124.256 12.718)
    minmax     0.080      0.920      (28.615  2.438)
    lbp_comb   0.055      0.945      (18.514  2.300)
    lbp_var    0.160      0.840      (10.084  0.492)
    ltph       0.142      0.858      (58.113  9.740)
    clbpdist   0.128      0.872      (113.202 20.671)
    wld        0.044      0.956      (86.753  3.302)
    norml2     0.083      0.917      (15.482  3.436)

    lfw_funneled.txt
    10 fold, 20 subjects, 400 images, 20 per person
    fisher     0.091      0.909      (1091.763  1.076)
    eigen      0.139      0.861      (1057.610  6.103)
    lbph       0.134      0.866      (160.593 24.447)
    lbph2_u    0.128      0.872      (113.603 11.373)
    minmax     0.098      0.902      (26.776  2.207)
    lbp_comb   0.118      0.882      (20.542  2.732)
    lbp_var    0.206      0.794      (10.013  0.457)
    ltph       0.165      0.835      (54.360  8.999)
    clbpdist   0.121      0.879      (103.970 18.762)
    wld        0.090      0.910      (85.888  3.049)
    norml2     0.139      0.861      (14.179  3.135)

