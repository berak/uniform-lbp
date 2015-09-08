
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
     HDLBP(30) HDLBP_PCA(31)   PCASIFT(32)      PNET(33)     CDIKP(34)
    LATCH2(35)     DAISY(36)       RBM(37)

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

(test/train times are averaged per fold)

------------------------------------------------------------------------------
att_faces :10 fold, 39 classes, 390 images, retina
------------------------------------------------------------------------------
[extra] [filt] [class]     [f_bytes]  [hit]  [miss]  [acc]  [t_train] [t_test]
Pixels   none   N_L2          12100    727     14    0.981    0.000    0.653
Pixels   none   SVM_POL       12100    735      6    0.992    3.900    0.690
Pixels   none   PCA_LDA       12100    725     16    0.978   16.518    0.221
Lbp      none   H_CHI         65536    728     13    0.982    0.001    4.142
Lbp      DCT8   PCA_LDA       32000    736      5    0.993   11.501    0.351
Lbp_P    DCT8   PCA_LDA       32000    738      3    0.996   11.952    0.353
Lbpu_P   DCT8   PCA_LDA       32000    737      4    0.995   12.686    0.478
MTS_P    none   PCA_LDA       11136    737      4    0.995    5.941    0.047
COMB_P   none   PCA_LDA       66816    738      3    0.996   18.738    0.321
COMB_P   HELL   SVM_INT2      66816    737      4    0.995    8.735    0.973
TpLbp_P  DCT8   PCA_LDA       32000    737      4    0.995   12.186    0.375
FpLbp_P  none   PCA_LDA       11136    737      4    0.995    6.088    0.052
FpLbp_P  HELL   SVM_INT2      11136    737      4    0.995    0.698    0.152
HDGRAD   DCT12  PCA_LDA       48000    689     52    0.930   16.050    0.238
HDLBP    DCT6   PCA_LDA       24000    684     57    0.923    9.584    0.178
Sift     DCT12  PCA_LDA       48000    731     10    0.987   17.828    0.220
Sift     HELL   SVM_INT2     492032    736      5    0.993   85.713   10.116
Grad_P   none   PCA_LDA       32016    738      3    0.996   12.195    0.356
GradMag  none   PCA_LDA       23552    732      9    0.988   10.617    0.206
GradMagP WHAD8  PCA_LDA       32000    740      1    0.999   14.221    0.375
PCASIFT  none   PCA_LDA       25600    691     50    0.933   12.157    0.193
GaborGB  none   PCA_LDA       36864    735      6    0.992   15.227    0.424
PNET     HELL   SVM_POL       23040    739      2    0.997    2.274    0.328
RBM      none   PCA_LDA       24560    647     94    0.873    6.930    0.170
CDIKP    DCT8   SVM_INT2      32000    733      8    0.989    2.122    0.393
PNET     none   SVM_INT2      23040    740      1    0.999    1.500    0.286

------------------------------------------------------------------------------
yale_cropped:         10 fold, 15 classes, 165 images, retina
------------------------------------------------------------------------------
[extra] [filt] [class]     [f_bytes]  [hit]  [miss]  [acc]  [t_train] [t_test]
Pixels   none   N_L2          12100    280     20    0.933    0.000    0.113
Pixels   none   SVM_POL       12100    293      7    0.977    0.599    0.109
Pixels   none   PCA_LDA       12100    295      5    0.983    2.463    0.028
Lbp      none   H_CHI         65536    288     12    0.960    0.001    0.661
Lbp      DCT8   PCA_LDA       32000    295      5    0.983    1.687    0.021
Lbp_P    DCT8   PCA_LDA       32000    295      5    0.983    1.777    0.021
Lbpu_P   DCT8   PCA_LDA       32000    295      5    0.983    1.663    0.022
MTS_P    none   PCA_LDA       11136    295      5    0.983    0.827    0.010
COMB_P   none   PCA_LDA       66816    294      6    0.980    3.154    0.036
COMB_P   HELL   SVM_INT2      66816    292      8    0.973    1.563    0.195
TpLbp_P  DCT8   PCA_LDA       32000    292      8    0.973    1.742    0.024
FpLbp_P  none   PCA_LDA       11136    295      5    0.983    0.820    0.009
FpLbp_P  HELL   SVM_INT2      11136    293      7    0.977    0.118    0.026
HDGRAD   DCT12  PCA_LDA       48000    297      3    0.990    2.713    0.027
HDLBP    DCT6   PCA_LDA       24000    294      6    0.980    1.433    0.015
Sift     DCT12  PCA_LDA       48000    295      5    0.983    2.514    0.028
Sift     HELL   SVM_INT2     492032    291      9    0.970   13.427    1.590
Grad_P   none   PCA_LDA       32016    291      9    0.970    1.719    0.024
GradMag  none   PCA_LDA       23552    295      5    0.983    1.384    0.019
GradMagP WHAD8  PCA_LDA       32000    292      8    0.973    1.798    0.021
PCASIFT  none   PCA_LDA       25600    294      6    0.980    1.562    0.015
GaborGB  none   PCA_LDA       36864    295      5    0.983    2.022    0.026
RBM      none   PCA_LDA       16320    286     14    0.953    0.877    0.011
PNET     none   SVM_POL       23040    296      4    0.987    0.345    0.055 * Pca[5,5] Gabor[9,5,1.73] Hashing[5,18]
CDIKP    DCT8   SVM_INT2      32000    297      3    0.990    0.387    0.070

------------------------------------------------------------------------------
lfw-3daligned:         10 fold, 50 classes, 500 images, retina
------------------------------------------------------------------------------
[extra] [filt] [class]     [f_bytes]  [hit]  [miss]  [acc]  [t_train] [t_test]
Pixels   none   N_L2          12100    761    189    0.801    0.000    1.051
Pixels   none   SVM_POL       12100    857     93    0.902    6.899    1.100
Pixels   none   PCA_LDA       12100    887     63    0.934   30.615    0.317
Lbp      none   H_CHI         65536    805    145    0.847    0.002    6.347
Lbp      DCT8   PCA_LDA       32000    913     37    0.961   20.844    0.546
Lbp_P    DCT8   PCA_LDA       32000    909     41    0.957   20.667    0.543
Lbpu_P   DCT8   PCA_LDA       32000    910     40    0.958   20.499    0.572
MTS_P    none   PCA_LDA       11136    914     36    0.962   11.007    0.093
COMB_P   none   PCA_LDA       66816    919     31    0.967   32.167    0.429
COMB_P   HELL   SVM_INT2      66816    906     44    0.954   13.080    1.476
TpLbp_P  DCT8   PCA_LDA       32000    912     38    0.960   21.198    0.569
FpLbp_P  none   PCA_LDA       11136    914     36    0.962   11.214    0.093
FpLbp_P  HELL   SVM_INT2      11136    902     48    0.949    1.108    0.234
HDGRAD   DCT12  PCA_LDA       48000    925     25    0.974   31.121    0.378
HDLBP    DCT6   PCA_LDA       24000    922     28    0.971   17.716    0.243
Sift     DCT12  PCA_LDA       48000    912     38    0.960   30.303    0.301
Sift     HELL   SVM_INT2     492032    913     37    0.961  122.865   14.067
Grad_P   none   PCA_LDA       32016    915     35    0.963   20.595    0.550
GradMag  none   PCA_LDA       23552    900     50    0.947   17.282    0.239
GradMagP WHAD8  PCA_LDA       32000    917     33    0.965   24.676    0.550
PCASIFT  none   PCA_LDA       25600    919     31    0.967   23.822    0.304
GaborGB  none   PCA_LDA       36864    906     44    0.954   28.726    0.640
PNET     none   SVM_POL       23040    908     42    0.956    3.625    0.530
RBM      none   PCA_LDA       24560    850    100    0.895   12.319    0.257
CDIKP    DCT8   SVM_INT2      32000    898     52    0.945    3.602    0.658

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
