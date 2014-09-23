
####this branch is a complete re-write.

since the contrib module got removed from 3.0, ( and *might* resurface in the contrib *repo* ),   

now i'm trying a more lego like approach, an extractor / classifier toolkit much like the feature2d api.


```


namespace TextureFeature	
{
    struct Extractor 
    {
        virtual int extract( const Mat &img, Mat &features ) const = 0;
    };

    struct Classifier 
    {
        virtual int train( const Mat &features, const Mat &labels ) = 0;
        virtual int predict( const Mat &test, Mat &result ) const = 0;
    };
};


```

the goal now is to achieve more flexibility, make it easy to swap features, knn against svm, distance measures.



    --------------------------------------------------------------
    aberdeen:          5 fold, 98 classes, 500 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_l2         14400    361     88    0.804    4.568
    pixels_svm         3600    390     59    0.869    5.552
    pixels_multi       3600      0    449    0.000   14.453
    lbp_l2            65536    338    111    0.753    9.590
    lbp_svm           65536    370     79    0.824   40.966
    lbp_hell          65536    391     58    0.871   36.448
    lbpu_red_hell      4352    325    124    0.724    4.255
    bgc1_hell         65536    368     81    0.820   36.526
    wld_hell          45056    279    170    0.621   40.767
    mts_svm            4096    394     55    0.878    3.086
    mts_hell           4096    358     91    0.797    3.521
    stu_svm           16384    410     39    0.913    8.623
    glcm_svm          65536    300    149    0.668   46.678
    gabor_hell        60416    401     48    0.893   50.987
    gabor_red         17408    392     57    0.873   27.797
    gabor_svm         60416    412     37    0.918   49.143
    dct_l2            12544    363     86    0.808    3.665
    dct_svm           12544    390     59    0.869    7.435
    eigen             14400    361     88    0.804   97.675
    fisher            14400    401     48    0.893   77.305
    --------------------------------------------------------------
    att:               10 fold, 40 classes, 400 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_l2         14400    740     20    0.974    8.931
    pixels_svm         3600    744     16    0.979   13.711
    pixels_multi       3600    705     55    0.928   41.910
    lbp_l2            65536    718     42    0.945   19.584
    lbp_svm           65536    737     23    0.970  107.094
    lbp_hell          65536    733     27    0.964   70.904
    lbpu_red_hell      4352    715     45    0.941    8.528
    bgc1_hell         65536    737     23    0.970   71.093
    wld_hell          45056    753      7    0.991   82.840
    mts_svm            4096    738     22    0.971    6.449
    mts_hell           4096    727     33    0.957    6.906
    stu_svm           16384    724     36    0.953   19.571
    glcm_svm          65536    749     11    0.986  115.771
    gabor_hell        60416    726     34    0.955  101.071
    gabor_red         17408    743     17    0.978   56.272
    gabor_svm         60416    727     33    0.957  115.706
    dct_l2            12544    730     30    0.961    8.248
    dct_svm           12544    736     24    0.968   17.566
    eigen             14400    740     20    0.974  246.252
    fisher            14400    720     40    0.947  207.481
    --------------------------------------------------------------
    lfw2fun:           10 fold, 20 classes, 400 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_l2         14400    222    276    0.446    6.425
    pixels_svm         3600    357    141    0.717   15.578
    pixels_multi       3600    281    217    0.564   37.059
    lbp_l2            65536    361    137    0.725   14.946
    lbp_svm           65536    422     76    0.847  108.177
    lbp_hell          65536    381    117    0.765   50.370
    lbpu_red_hell      4352    302    196    0.606    6.586
    bgc1_hell         65536    375    123    0.753   50.531
    wld_hell          45056    133    365    0.267   67.659
    mts_svm            4096    417     81    0.837    6.952
    mts_hell           4096    350    148    0.703    5.495
    stu_svm           16384    336    162    0.675   21.439
    glcm_svm          65536    187    311    0.376  128.983
    gabor_hell        60416    316    182    0.635   83.028
    gabor_red         17408    282    216    0.566   51.788
    gabor_svm         60416    335    163    0.673  116.583
    dct_l2            12544    230    268    0.462    7.400
    dct_svm           12544    362    136    0.727   18.474
    eigen             14400    223    275    0.448  272.145
    fisher            14400    346    152    0.695  233.791
    --------------------------------------------------------------
    senthil:           10 fold, 5 classes, 80 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_l2         14400     87     13    0.870    0.404
    pixels_svm         3600     94      6    0.940    0.522
    pixels_multi       3600     93      7    0.930    1.111
    lbp_l2            65536     93      7    0.930    1.597
    lbp_svm           65536     92      8    0.920    4.607
    lbp_hell          65536     94      6    0.940    3.050
    lbpu_red_hell      4352     94      6    0.940    0.965
    bgc1_hell         65536     94      6    0.940    3.055
    wld_hell          45056     98      2    0.980    8.841
    mts_svm            4096     92      8    0.920    1.275
    mts_hell           4096     94      6    0.940    0.713
    stu_svm           16384     96      4    0.960    1.399
    glcm_svm          65536     99      1    0.990    6.209
    gabor_hell        60416     94      6    0.940   10.023
    gabor_red         17408     94      6    0.940    8.508
    gabor_svm         60416     94      6    0.940   10.767
    dct_l2            12544     87     13    0.870    1.248
    dct_svm           12544     95      5    0.950    1.589
    eigen             14400     88     12    0.880    9.676
    fisher            14400     94      6    0.940    7.083
    --------------------------------------------------------------
    sheffield:         10 fold, 19 classes, 500 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_l2         14400    581      7    0.988    9.336
    pixels_svm         3600    581      7    0.988   19.742
    pixels_multi       3600    572     16    0.973   34.272
    lbp_l2            65536    572     16    0.973   22.468
    lbp_svm           65536    577     11    0.981  145.740
    lbp_hell          65536    576     12    0.980   74.394
    lbpu_red_hell      4352    551     37    0.937    9.455
    bgc1_hell         65536    576     12    0.980   75.753
    wld_hell          45056    587      1    0.998   92.920
    mts_svm            4096    578     10    0.983    9.411
    mts_hell           4096    570     18    0.969    7.728
    stu_svm           16384    568     20    0.966   28.108
    glcm_svm          65536    582      6    0.990  152.146
    gabor_hell        60416    572     16    0.973  114.289
    gabor_red         17408    577     11    0.981   67.891
    gabor_svm         60416    576     12    0.980  152.041
    dct_l2            12544    577     11    0.981    8.738
    dct_svm           12544    578     10    0.983   22.977
    eigen             14400    578     10    0.983  472.288
    fisher            14400    537     51    0.913  424.227
    --------------------------------------------------------------
    yale:              10 fold, 15 classes, 165 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_l2         14400    252     48    0.840    1.651
    pixels_svm         3600    267     33    0.890    1.966
    pixels_multi       3600    262     38    0.873    5.184
    lbp_l2            65536    241     59    0.803    4.466
    lbp_svm           65536    266     34    0.887   14.004
    lbp_hell          65536    278     22    0.927   12.964
    lbpu_red_hell      4352    240     60    0.800    2.302
    bgc1_hell         65536    271     29    0.903   12.908
    wld_hell          45056    242     58    0.807   20.438
    mts_svm            4096    266     34    0.887    1.570
    mts_hell           4096    254     46    0.847    1.753
    stu_svm           16384    262     38    0.873    3.866
    glcm_svm          65536    254     46    0.847   17.286
    gabor_hell        60416    288     12    0.960   26.436
    gabor_red         17408    284     16    0.947   19.135
    gabor_svm         60416    282     18    0.940   26.120
    dct_l2            12544    249     51    0.830    2.758
    dct_svm           12544    266     34    0.887    3.975
    eigen             14400    252     48    0.840   40.255
    fisher            14400    257     43    0.857   28.471
    --------------------------------------------------------------
    yaleB:             10 fold, 8 classes, 500 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_l2         14400    469     81    0.853    9.006
    pixels_svm         3600    168    382    0.305   16.055
    pixels_multi       3600    411    139    0.747   23.350
    lbp_l2            65536    533     17    0.969   19.935
    lbp_svm           65536    547      3    0.995  111.467
    lbp_hell          65536    542      8    0.985   72.184
    lbpu_red_hell      4352    534     16    0.971    9.592
    bgc1_hell         65536    540     10    0.982   72.937
    wld_hell          45056    247    303    0.449   91.548
    mts_svm            4096    547      3    0.995    7.994
    mts_hell           4096    540     10    0.982    7.618
    stu_svm           16384    407    143    0.740   25.794
    glcm_svm          65536    195    355    0.355  124.902
    gabor_hell        60416    502     48    0.913  110.900
    gabor_red         17408    457     93    0.831   68.337
    gabor_svm         60416    518     32    0.942  138.727
    dct_l2            12544    473     77    0.860    9.707
    dct_svm           12544    195    355    0.355   21.497
    eigen             14400    469     81    0.853  503.998
    fisher            14400    537     13    0.976  454.957
