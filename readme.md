
this started out as a try to improve opencv's lbph facereco, 

but meanwhile it morphed into a testbed for comparing preprocessing/extraction/classification methods

it all builds on top of opencv30 .

     the pipeline is:
        (preprocessing) -> extractor -> filter -> classifier (or verifier)

-----------------------------------------------------

4 projects in here:
* online.cpp : realtime webcam app with online training
* duel.cpp : shootout of different (identification) pipeline combinations (see below)
* fr_lfw_benchmark.cpp: the opencv (verification) lfw benchmark (from contrib/datasets)
* frontalize.cpp: 3d/2d frontal face alignment lib/standalone tool (using [dlib](http://sourceforge.net/projects/dclib/files/dlib/) landmarks)

------------------------------------------------------

<pre>
face3> duel -o

[extractors]  :

    Pixels( 0)       Lbp( 1)     Lbp_P( 2)    Lbpu_P( 3)       Ltp( 4)
     TPLbp( 5)   TpLbp_P( 6)   TpLbp_G( 7)     FPLbp( 8)   FpLbp_P( 9)
       MTS(10)     MTS_P(11)      BGC1(12)    BGC1_P(13)      COMB(14)
    COMB_P(15)    COMB_G(16)  GaborLBP(17)   GaborGB(18)       Dct(19)
       Orb(20)      Sift(21)    Sift_G(22)      Grad(23)    Grad_G(24)
    Grad_P(25)   GradMag(26)  GradMagP(27)   GradBin(28)    HDGRAD(29)
     HDLBP(30) HDLBP_PCA(31)   PCASIFT(32)      PNET(33)    PCANET(34)
   RANDNET(35)   WAVENET(36)     CDIKP(37)    LATCH2(38)     DAISY(39)

[filters] :

      none( 0)      HELL( 1)       POW( 2)      SQRT( 3)     WHAD_( 4)
     WHAD4( 5)     WHAD8( 6)        RP( 7)      DCT_( 8)      DCT2( 9)
      DCT4(10)      DCT6(11)      DCT8(12)     DCT12(13)     DCT16(14)
     DCT24(15)     BITS2(16)     BITS4(17)     BITS8(18)      MEAN(19)

[classifiers] :

      N_L2( 0)   N_L2SQR( 1)      N_L1( 2)     N_HAM( 3)    H_HELL( 4)
     H_CHI( 5)    COSINE( 6)     KLDIV( 7)   SVM_LIN( 8)   SVM_POL( 9)
   SVM_RBF(10)   SVM_INT(11)  SVM_INT2(12)   SVM_HEL(13) SVM_HELSQ(14)
   SVM_LOW(15)   SVM_LOG(16)  SVM_KMOD(17)SVM_CAUCHY(18) SVM_MULTI(19)
       PCA(20)   PCA_LDA(21)       LDA(22)       MLP(23)       KNN(24)
     RTREE(25)

</pre>

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
WAVENET  none   SVM_LIN       18432    738      3    0.996 2563.668  * 9 [6 28][6 28] (pure luck, i guess)
PCANET   none   SVM_INT2      24576    737      4    0.995 3312.173  * SS_2D
LATCH    none   SVM_POL        1568    721     20    0.973  134.163
LATCH    none   PCA_LDA        1568    725     16    0.978  202.412
LATCH    none   PCA_LDA        1800    726     15    0.980  128.738  * ssd=1 step=4 bytes=8 
LATCH2   none   PCA_LDA        5120    727     14    0.981  699.934  * ssd=5 bytes=256
PNET     none   SVM_INT2      24576    739      2    0.997  241.797  * pca/gabor
PNET     none   SVM_INT2      23040    740      1    0.999  135.627  * pca[7,5] gabor[9, 5, 1.73f, 1.0f]

-------------------------------------------------------------------
data/yale_crop:         10 fold, 15 classes, 165 images, retina       (2d/3d aligned)
-------------------------------------------------------------------
[extra] [redu] [class]     [f_bytes]  [hit]  [miss]  [acc]   [time]
Pixels   none   N_L2          12100    280     20    0.933    1.195
Pixels   none   SVM_POL       12100    293      7    0.977    6.421
Pixels   none   PCA_LDA       12100    295      5    0.983   24.136
Lbp      none   H_CHI         65536    288     12    0.960    7.699
Lbp      DCT8   PCA_LDA       32000    295      5    0.983   18.980
Lbp_P    DCT8   PCA_LDA       32000    295      5    0.983   27.468
Lbpu_P   DCT8   PCA_LDA       32000    295      5    0.983   19.987
MTS_P    none   PCA_LDA       11136    295      5    0.983    9.065
COMB_P   none   PCA_LDA       66816    294      6    0.980   41.924
COMB_P   HELL   SVM_INT2      66816    292      8    0.973   27.169
TpLbp_P  DCT8   PCA_LDA       32000    292      8    0.973   28.746
FpLbp_P  none   PCA_LDA       11136    295      5    0.983    9.020
FpLbp_P  HELL   SVM_INT2      11136    293      7    0.977    3.174
LATCH2   none   PCA_LDA        5120    289     11    0.963  267.458
HDGRAD   DCT12  PCA_LDA       48000    296      4    0.987   64.147
HDLBP    DCT6   PCA_LDA       24000    294      6    0.980   66.411
HDLBP_PCA none   PCA_LDA      51200    295      5    0.983  160.765
Sift     DCT12  PCA_LDA       48000    295      5    0.983  212.618
Sift     HELL   SVM_INT2     492032    291      9    0.970  328.427
Grad_P   none   PCA_LDA       32016    291      9    0.970   18.854
GradMag  none   PCA_LDA       23552    295      5    0.983   15.507
GradMagP WHAD8  PCA_LDA       32000    292      8    0.973   22.403
PCASIFT  none   PCA_LDA       25600    294      6    0.980  253.712
GaborGB  none   PCA_LDA       36864    294      6    0.980   37.232
WAVENET  none   SVM_LIN       13824    283     17    0.943   75.294
PNET     none   SVM_POL       23040    296      4    0.987   51.674  * pca[7,5] gabor[9, 5, 1.73f, 1.0f]
CDIKP    DCT8   SVM_INT2      32000    297      3    0.990   34.882

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
PNET     none   SVM_INT2      23040    893     57    0.940  176.526

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
PCANET   none   SVM_INT2      24576    900     50    0.947 3919.794  * SS_2D_nuclear 6 filters 11x11
RANDNET  none   SVM_LIN        3072    874     76    0.920 2506.335
WAVENET  none   SVM_LIN       18432    893     57    0.940 3232.364
LATCH    none   PCA_LDA        1568    883     67    0.929  318.049
LATCH2   none   PCA_LDA        5120    903     47    0.951 1004.513
DAISY    none   PCA_LDA       64800    902     48    0.949  422.905
PNET     none   SVM_INT2      23040    912     38    0.960  181.208  * pca[7,5] gabor[9, 5, 1.73f, 1.0f]

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
