
this started out as a try to improve opencv's lbph facereco, 

but meanwhile it morphed into a testbed for comparing preprocessing/extraction/classification methods

it all builds on top of opencv30 .

     the pipeline is:
        (preprocessing) -> filter -> reductor -> classifier (or verifier)

-----------------------------------------------------

4 projects in here:
* online.cpp : realtime webcam app with online training
* duel.cpp : shootout of different (identification) pipeline combinations (see below)
* fr_lfw_benchmark.cpp: the opencv (verification) lfw benchmark (from contrib/datasets)
* frontalize.cpp: 3d/2d frontal face alignment lib/standalone tool (using [dlib](http://sourceforge.net/projects/dclib/files/dlib/) landmarks)

------------------------------------------------------

#### some results:

------------------------------------------------------

<pre>
-------------------------------------------------------------------
att_faces :                10 fold, 39 classes, 390 images, retina
-------------------------------------------------------------------
[extra] [redu] [class]     [f_bytes]  [hit]  [miss]  [acc]   [time]
Pixels   none   N_L2           8100    729     12    0.984    4.504
Pixels   none   SVM_POL        8100    735      6    0.992   24.422
Pixels   none   PCA_LDA        8100    727     14    0.981  148.070
Lbp      DCT8   PCA_LDA       32000    738      3    0.996  125.304
Lbp_P    DCT8   PCA_LDA       32000    736      5    0.993  140.943
Lbpu_P   DCT8   PCA_LDA       32000    738      3    0.996  122.356
MTS_P    none   PCA_LDA       11136    736      5    0.993   59.872
COMB_P   none   PCA_LDA       33408    738      3    0.996  128.649
COMB_P   HELL   SVM_INT2      33408    738      3    0.996   37.473
TpLbp_P  DCT8   PCA_LDA       32000    737      4    0.995  141.724
FpLbp_P  none   PCA_LDA       11136    736      5    0.993   60.904
FpLbp_P  HELL   SVM_INT2      11136    738      3    0.996   11.355
HDGRAD   DCT12  PCA_LDA       48000    683     58    0.922  321.115
HDLBP    DCT6   PCA_LDA       24000    674     67    0.910  201.347
HDLBP_PCA none  PCA_LDA       51200    677     64    0.914  444.291
Sift     DCT12  PCA_LDA       48000    736      5    0.993  422.418
Sift     HELL   SVM_INT2     184832    737      4    0.995  483.212
Grad_P   none   PCA_LDA       32016    740      1    0.999  121.769
GradBin  DCT4   PCA_LDA       16000    729     12    0.984   97.514
GradMag  none   PCA_LDA       23552    733      8    0.989   97.937
GradMagP WHAD8  PCA_LDA       32000    739      2    0.997  143.955
GaborGB  none   PCA_LDA       36864    732      9    0.988  179.415
PCASIFT  none   PCA_LDA       25600    695     46    0.938  696.151
PCANET   none   SVM_LIN        3072    730     11    0.985 1306.950
RANDNET  none   SVM_LIN        3072    728     13    0.982 2424.438
WAVENET  none   SVM_LIN       18432    738      3    0.996 2563.668  * 9 [6 28][6 28]
LATCH    none   SVM_POL        1568    721     20    0.973  134.163
LATCH    none   PCA_LDA        1568    725     16    0.978  202.412
LATCH    none   PCA_LDA        1800    726     15    0.980  128.738  * ssd=1 step=4 bytes=8 
LATCH2   none   PCA_LDA        5120    727     14    0.981  699.934  * ssd=5 bytes=256
-------------------------------------------------------------------
data/yale:              10 fold, 15 classes, 165 images, retina
-------------------------------------------------------------------
[extra] [redu] [class]     [f_bytes]  [hit]  [miss]  [acc]   [time]
Pixels   none   N_L2           8100    290     10    0.967    0.814
Pixels   none   SVM_POL        8100    289     11    0.963    3.533
Pixels   none   PCA_LDA        8100    295      5    0.983   17.799
Lbp      DCT8   PCA_LDA       32000    288     12    0.960   18.678
Lbp_P    DCT8   PCA_LDA       32000    289     11    0.963   26.318
Lbpu_P   DCT8   PCA_LDA       32000    288     12    0.960   19.230
MTS_P    none   PCA_LDA       11136    287     13    0.957    8.502
COMB_P   none   PCA_LDA       33408    288     12    0.960   20.700
COMB_P   HELL   SVM_INT2      33408    282     18    0.940    9.881
TpLbp_P  DCT8   PCA_LDA       32000    287     13    0.957   27.119
FpLbp_P  none   PCA_LDA       11136    287     13    0.957    8.521
FpLbp_P  HELL   SVM_INT2      11136    279     21    0.930    2.789
HDGRAD   DCT12  PCA_LDA       48000    269     31    0.897   70.254
HDLBP    DCT6   PCA_LDA       24000    262     38    0.873   56.433
HDLBP_PCA none  PCA_LDA       51200    271     29    0.903  149.460
Sift     DCT12  PCA_LDA       48000    296      4    0.987  147.285
Sift     HELL   SVM_INT2     184832    291      9    0.970  153.513
Grad_P   none   PCA_LDA       32016    287     13    0.957   18.161
GradBin  DCT4   PCA_LDA       16000    288     12    0.960   13.073
GradMag  none   PCA_LDA       23552    291      9    0.970   14.966
GradMagP WHAD8  PCA_LDA       32000    287     13    0.957   21.069
PCASIFT  none   PCA_LDA       25600    280     20    0.933  277.623
GaborGB  none   PCA_LDA       36864    289     11    0.963   31.939
PCANET   none   SVM_LIN        3072    289     11    0.963  822.886   *    11 [4 28][4 23]
PCANET   none   SVM_LIN       98304    291      9    0.970 1903.782   *    11 [8 28][8 28]
PCANET   DCT8   SVM_LIN       32000    292      8    0.973 1897.975   *     9 [8 28][8 28]
PCANET   none   SVM_LIN       18432    292      8    0.973  875.066   * wave 9 [6 28][6 28]
RANDNET  none   SVM_LIN        3072    289     11    0.963 1025.370
RANDNET  none   SVM_LIN       98304    292      8    0.973 1896.199
LATCH    none   SVM_POL        1568    293      7    0.977   53.987   * N=8, PS=4
LATCH2   none   PCA_LDA        5120    295      5    0.983  268.179   * ssd=5 bytes=256
DAISY    none   PCA_LDA       64800    295      5    0.983   68.231

-------------------------------------------------------------------
lfw-deepfunneled/:       10 fold, 50 classes, 500 images, retina
-------------------------------------------------------------------
[extra] [redu] [class]     [f_bytes]  [hit]  [miss]  [acc]   [time]
Pixels   none   N_L2           8100    681    269    0.717    7.287
Pixels   none   SVM_POL        8100    821    129    0.864   48.634
Pixels   none   PCA_LDA        8100    874     76    0.920  277.359
Lbp      DCT8   PCA_LDA       32000    891     59    0.938  222.525
Lbp_P    DCT8   PCA_LDA       32000    903     47    0.951  243.516
Lbpu_P   DCT8   PCA_LDA       32000    905     45    0.953  222.174
MTS_P    none   PCA_LDA       11136    891     59    0.938  119.950
MTS_P    none   SVM_POL       11136    870     80    0.916   21.685
COMB_P   none   PCA_LDA       33408    901     49    0.948  233.759
COMB_P   HELL   SVM_INT2      33408    883     67    0.929   64.030
TpLbp_P  DCT8   PCA_LDA       32000    903     47    0.951  261.289
FpLbp_P  none   PCA_LDA       11136    891     59    0.938  124.814
FpLbp_P  none   SVM_POL       11136    870     80    0.916   22.687
FpLbp_P  HELL   SVM_INT2      11136    882     68    0.928   18.382
HDGRAD   DCT12  PCA_LDA       48000    905     45    0.953  473.703
HDLBP    WHAD8  PCA_LDA       32000    881     69    0.927  389.667
Sift     DCT12  PCA_LDA       48000    889     61    0.936  690.112
Sift     HELL   SVM_INT2     184832    899     51    0.946  793.148
Grad_P   none   PCA_LDA       32016    900     50    0.947  227.527
GradBin  DCT4   PCA_LDA       16000    891     59    0.938  190.586
GradMag  none   PCA_LDA       23552    874     76    0.920  179.792
GradMagP WHAD8  PCA_LDA       32000    905     45    0.953  259.194
GaborGB  none   PCA_LDA       36864    890     60    0.937  315.207
PCASIFT  none   PCA_LDA       25600    884     66    0.931 1069.912
LATCH    none   SVM_POL        1568    813    137    0.856  173.486

-------------------------------------------------------------------
lfw-3daligned:            10 fold, 50 classes, 500 images, retina
-------------------------------------------------------------------
[extra] [redu] [class]     [f_bytes]  [hit]  [miss]  [acc]   [time]
Pixels   none   N_L2           8100    765    185    0.805   10.365
Pixels   none   SVM_POL        8100    859     91    0.904   45.358
Pixels   none   PCA_LDA        8100    886     64    0.933  277.845
Lbp      DCT8   PCA_LDA       32000    911     39    0.959  220.734
Lbp_P    DCT8   PCA_LDA       32000    915     35    0.963  245.080
Lbpu_P   DCT8   PCA_LDA       32000    918     32    0.966  222.277
MTS_P    none   PCA_LDA       11136    905     45    0.953  119.736
COMB_P   DCT12  PCA_LDA       48000    920     30    0.968  291.551
COMB_P   HELL   SVM_INT2      33408    903     47    0.951   58.985
TpLbp_P  DCT8   PCA_LDA       32000    902     48    0.949  247.495
FpLbp_P  none   PCA_LDA       11136    905     45    0.953  119.186
FpLbp_P  HELL   SVM_INT2      11136    902     48    0.949   17.610
HDGRAD   DCT12  PCA_LDA       48000    929     21    0.978  486.974
HDLBP    DCT6   PCA_LDA       24000    914     36    0.962  310.240
HDLBP_PCA none  PCA_LDA       51200    923     27    0.972  637.759
Sift     DCT12  PCA_LDA       48000    912     38    0.960  661.536
Sift     HELL   SVM_INT2     184832    912     38    0.960  746.856
Grad_P   none   PCA_LDA       32016    915     35    0.963  215.729
GradBin  DCT4   PCA_LDA       16000    920     30    0.968  190.996
GradMag  none   PCA_LDA       23552    911     39    0.959  179.856
GradMagP WHAD8  PCA_LDA       32000    911     39    0.959  260.980
PCASIFT  none   PCA_LDA       25600    915     35    0.963 1021.913
GaborGB  none   PCA_LDA       36864    918     32    0.966  328.633
PCANET   none   SVM_LIN        3072    880     70    0.926 1683.532
RANDNET  none   SVM_LIN        3072    874     76    0.920 2506.335
WAVENET  none   SVM_LIN       18432    893     57    0.940 3232.364
LATCH    none   PCA_LDA        1568    883     67    0.929  318.049
LATCH2   none   PCA_LDA        5120    903     47    0.951 1004.513
DAISY    none   PCA_LDA       64800    902     48    0.949  422.905

-------------------------------------------------------------------
data/f94gender:         10 fold, 2 classes, 484 images, retina
-------------------------------------------------------------------
[extra] [redu] [class]     [f_bytes]  [hit]  [miss]  [acc]   [time]
Pixels   none   N_L2          12100    445     45    0.908    6.271
Pixels   none   SVM_POL       12100    467     23    0.953   13.195
Pixels   none   PCA_LDA       12100    466     24    0.951  406.137
Lbp      none   H_CHI         65536    407     83    0.831   42.218
Lbp      DCT8   PCA_LDA       32000    468     22    0.955  257.919
Lbp_P    DCT8   PCA_LDA       32000    470     20    0.959  271.282
Lbpu_P   DCT8   PCA_LDA       32000    472     18    0.963  248.733
MTS_P    none   PCA_LDA       11136    462     28    0.943  149.284
COMB_P   none   PCA_LDA       66816    466     24    0.951  392.213
COMB_P   HELL   SVM_INT2      66816    466     24    0.951   88.184
TpLbp_P  DCT8   PCA_LDA       32000    469     21    0.957  256.392
FpLbp_P  none   PCA_LDA       11136    462     28    0.943  138.421
FpLbp_P  HELL   SVM_INT2      11136    451     39    0.920   13.367
HDGRAD   DCT12  PCA_LDA       48000    426     64    0.869  467.374
HDLBP    DCT6   PCA_LDA       24000    412     78    0.841  358.071
HDLBP_PCA none   PCA_LDA      51200    428     62    0.873  706.529
Sift     DCT12  PCA_LDA       48000    448     42    0.914  790.889
Sift     HELL   SVM_INT2     184832    456     34    0.931  510.964
Grad_P   none   PCA_LDA       32016    468     22    0.955  231.309
GaborGB  none   PCA_LDA       36864    454     36    0.927  366.183
CDIKP    DCT8   PCA_LDA       32000    444     46    0.906  427.614
WAVENET  none   SVM_LIN       13824    452     38    0.922 2517.690


</pre>
