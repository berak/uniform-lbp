
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
att_faces : 10 fold, 39 classes, 390 images, retina
------------------------------------------------------------------------------
[extra] [filt] [class]     [f_bytes]  [hit]  [miss]  [acc]  [t_train] [t_test]
Pixels   none   N_L2          12100    709     32    0.957    0.000    0.641
Pixels   none   SVM_POL       12100    722     19    0.974    5.269    0.748
Pixels   none   PCA_LDA       12100    689     52    0.930   16.371    0.236
Lbp      none   H_CHI         65536    700     41    0.945   -0.001    4.261
Lbp      none   H_HELL        65536    727     14    0.981    0.001    5.701
Lbp      none   SVM_LIN       65536    727     14    0.981   10.661    2.792
Lbpu_P   DCT8   SVM_LIN       32000    732      9    0.988    3.394    1.168
MTS_P    none   SVM_INT2      11136    731     10    0.987    0.704    0.178
COMB_P   HELL   SVM_INT2      66816    732      9    0.988    9.312    1.009
TpLbp_P  DCT8   SVM_INT2      32000    730     11    0.985    2.399    0.483
FpLbp_P  none   SVM_INT2      11136    731     10    0.987    0.796    0.186
FpLbp_P  none   MLP           11136    737      4    0.995  102.610    0.162
HDGRAD   DCT24  SVM_LIN       96000    662     79    0.893   15.342    3.709
HDLBP    DCT24  SVM_INT2      96000    647     94    0.873   11.912    1.336
Sift     HELL   SVM_INT2     492032    716     25    0.966  101.979   11.528
Sift     DCT24  SVM_LIN       96000    694     47    0.937   17.222    4.562
Grad_P   MEAN   SVM_LIN       32016    731     10    0.987    4.156    1.505
GradMag  none   MLP           23552    694     47    0.937  165.413    0.549
GradMagP WHAD8  SVM_LIN       32000    733      8    0.989    3.822    1.196
PCASIFT  none   SVM_LIN       25600    658     83    0.888    2.689    1.041
GaborGB  MEAN   SVM_LIN       36864    727     14    0.981    4.494    1.490
PNET     none   SVM_LIN       23040    737      4    0.995    2.444    0.948
PNET     none   MLP           23040    741      0    1.000  113.726    0.461
CDIKP    DCT8   SVM_INT2      32000    697     44    0.941    2.438    0.487

------------------------------------------------------------------------------
yale_cropped :         10 fold, 15 classes, 165 images, retina
------------------------------------------------------------------------------
[extra] [filt] [class]     [f_bytes]  [hit]  [miss]  [acc]  [t_train] [t_test]
Pixels   none   N_L2          12100    247     53    0.823    0.000    0.102
Pixels   none   SVM_POL       12100    268     32    0.893    0.841    0.120
Pixels   none   PCA_LDA       12100    256     44    0.853    2.388    0.037
Lbp      none   H_CHI         65536    273     27    0.910    0.001    0.647
Lbp      none   H_HELL        65536    293      7    0.977    0.001    1.002
Lbp      none   SVM_LIN       65536    296      4    0.987    1.577    0.158
Lbpu     DCT2   MLP            8000    296      4    0.987   14.785    0.011
Lbpu_P   DCT8   SVM_LIN       32000    296      4    0.987    0.535    0.056
MTS_P    none   SVM_INT2      11136    294      6    0.980    0.111    0.011
COMB_P   HELL   SVM_INT2      66816    293      7    0.977    1.321    0.144
TpLbp_P  DCT8   SVM_INT2      32000    296      4    0.987    0.377    0.073
FpLbp_P  none   SVM_INT2      11136    294      6    0.980    0.113    0.023
FpLbp_P  none   MLP           11136    295      5    0.983   11.644    0.024
HDGRAD   DCT24  SVM_LIN       96000    296      4    0.987    2.247    0.212
HDLBP    DCT24  SVM_INT2      96000    291      9    0.970    2.222    0.246
Sift     HELL   SVM_INT2     492032    289     11    0.963   14.645    1.683
Sift     DCT24  SVM_LIN       96000    286     14    0.953    2.804    0.255
Grad_P   MEAN   SVM_LIN       32016    295      5    0.983    0.674    0.080
GradMag  none   MLP           23552    283     17    0.943   35.060    0.104
GradMagP WHAD8  SVM_LIN       32000    296      4    0.987    0.684    0.083
PCASIFT  none   SVM_LIN       25600    285     15    0.950    0.422    0.067
GaborGB  MEAN   SVM_LIN       36864    293      7    0.977    0.987    0.104
PNET     none   SVM_LIN       23040    294      6    0.980    0.430    0.069
PNET     none   MLP           23040    297      3    0.990   18.541    0.060
CDIKP    DCT8   SVM_INT2      32000    283     17    0.943    0.446    0.082

------------------------------------------------------------------------------
lfw3d-aligned:       10 fold, 50 classes, 500 images, retina
------------------------------------------------------------------------------
[extra] [filt] [class]     [f_bytes]  [hit]  [miss]  [acc]  [t_train] [t_test]
Pixels   none   N_L2          12100    371    579    0.391    0.000    1.200
Pixels   none   SVM_POL       12100    578    372    0.608    9.938    1.300
Pixels   none   PCA_LDA       12100    647    303    0.681   31.688    0.319
Lbp      none   H_CHI         65536    480    470    0.505    0.002    6.909
Lbp      none   H_HELL        65536    590    360    0.621   -0.000    9.209
Lbp      none   SVM_LIN       65536    693    257    0.729   19.759    6.148
Lbpu_P   DCT8   SVM_LIN       32000    672    278    0.707    6.266    2.538
MTS_P    none   SVM_INT2      11136    691    259    0.727    1.175    0.321
COMB_P   HELL   SVM_INT2      66816    708    242    0.745   16.009    1.757
TpLbp_P  DCT8   SVM_INT2      32000    705    245    0.742    4.113    0.851
FpLbp_P  none   SVM_INT2      11136    691    259    0.727    1.166    0.291
FpLbp_P  none   MLP           11136    745    205    0.784  193.376    0.432
HDGRAD   DCT24  SVM_LIN       96000    810    140    0.853   31.486    9.809
HDLBP    DCT24  SVM_INT2      96000    756    194    0.796   26.997    3.022
Sift     HELL   SVM_INT2     492032    759    191    0.799  136.982   15.049
Sift     DCT24  SVM_LIN       96000    732    218    0.771   26.509    8.290
Grad_P   MEAN   SVM_LIN       32016    693    257    0.729    7.148    2.639
GradMag  none   MLP           23552    703    247    0.740  295.474    1.147
GradMagP WHAD8  SVM_LIN       32000    734    216    0.773    7.123    2.594
PCASIFT  none   SVM_LIN       25600    750    200    0.789    4.883    2.089
GaborGB  MEAN   SVM_LIN       36864    720    230    0.758    8.862    3.098
PNET     none   SVM_LIN       23040    756    194    0.796    4.783    1.884
PNET     none   MLP           23040    852     98    0.897  161.164    0.846
CDIKP    DCT8   SVM_INT2      32000    658    292    0.693    4.398    0.806

</pre>
