

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

(all code in the master branch is dependant on opencv *master* version, please use the 2.4 branch otherwise)

#results:

    //
    // method  mean_err   hitrate     t_train t_test (in seconds)
    //


    yale.txt 
    10 fold, 15 subjects, 165 images, 11 per person (1 images skipped)
    fisher       0.074      0.926      (117.387  0.550)
    eigen        0.074      0.926      (222.569  4.956)
    lbph         0.044      0.956      (52.051  6.734)
    lbph2_u      0.038      0.962      (41.261  3.135)
    minmax       0.055      0.945      (10.373  0.736)
    lbp_comb     0.048      0.952      ( 6.523  0.731)
    lbp_var      0.078      0.922      ( 4.519  0.209)
    ltph         0.040      0.960      (15.914  2.214)
    clbpdist     0.021      0.979      (34.757  2.554)
    wld          0.050      0.950      (28.428  1.205)
    norml2       0.063      0.937      ( 3.418  0.759)

    att.txt 
    10 fold, 40 subjects, 400 images, 10 per person
    fisher       0.020      0.980      (915.604  3.606)
    eigen        0.011      0.989      (869.801  8.718)
    lbph         0.017      0.983      (254.832 43.073)
    lbph2_u      0.017      0.983      (149.759 18.401)
    minmax       0.017      0.983      (34.645  3.469)
    lbp_comb     0.008      0.992      (27.139  4.003)
    lbp_var      0.066      0.934      (11.855  0.728)
    ltph         0.039      0.961      (73.131 12.452)
    clbpdist     0.031      0.969      (113.483 11.959)
    wld          0.004      0.996      (86.415  4.463)
    norml2       0.011      0.989      (21.719  4.607)


    tv.txt 
    10 fold, 24 subjects, 400 images, 16 per person
    fisher       0.011      0.989      (932.121  1.237)
    eigen        0.008      0.992      (910.474  6.013)
    lbph         0.006      0.994      (179.476 27.793)
    lbph2_u      0.006      0.994      (120.074 12.126)
    minmax       0.004      0.996      (28.902  2.310)
    lbp_comb     0.004      0.996      (20.825  2.738)
    lbp_var      0.019      0.981      (11.039  0.480)
    ltph         0.012      0.988      (52.608  8.386)
    clbpdist     0.011      0.989      (93.438  7.942)
    wld          0.004      0.996      (79.253  2.950)
    norml2       0.008      0.992      (13.960  3.136)


    f96.txt 
    10 fold, 21 subjects, 360 images, 17 per person (40 images skipped)
    fisher       0.000      1.000      (747.312  1.130)
    eigen        0.004      0.996      (723.811  5.655)
    lbph         0.008      0.992      (187.679 29.339)
    lbph2_u      0.008      0.992      (117.668 12.104)
    minmax       0.003      0.997      (28.340  2.281)
    lbp_comb     0.002      0.998      (20.005  2.596)
    lbp_var      0.038      0.962      (10.238  0.485)
    ltph         0.003      0.997      (50.379  7.977)
    clbpdist     0.004      0.996      (88.875  7.668)
    wld          0.003      0.997      (75.535  2.951)
    norml2       0.004      0.996      (13.823  3.068)


    lfw2fun.txt 
    10 fold, 20 subjects, 400 images, 20 per person
    fisher       0.101      0.899      (1034.157  1.169)
    eigen        0.172      0.828      (1024.828  6.294)
    lbph         0.136      0.864      (179.300 28.021)
    lbph2_u      0.130      0.870      (119.231 12.408)
    minmax       0.096      0.904      (29.346  2.327)
    lbp_comb     0.116      0.884      (21.065  2.768)
    lbp_var      0.204      0.796      (10.958  0.481)
    ltph         0.191      0.809      (53.292  8.556)
    clbpdist     0.110      0.890      (96.281  8.117)
    wld          0.118      0.882      (81.607  2.963)
    norml2       0.172      0.828      (14.496  3.211)




