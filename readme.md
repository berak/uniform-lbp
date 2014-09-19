
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




    -------------------------------------------------------
    senthil    :  10 fold, 5 classes, 80 images, no preproc
    -------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels            14400     87     13    0.870    0.404
    pixels_svm         3600     94      6    0.940    0.545    * 60x60 pixels only
    lbp               65536     93      7    0.930    1.644
    lbp_svm           65536     92      8    0.920    4.543
    lbp_hell          65536     94      6    0.940    3.036
    lbpu_red_hell      4352     94      6    0.940    0.975
    bgc1_hell         65536     94      6    0.940    3.046
    wld_hell          45056     98      2    0.980    8.783
    mts_svm            4096     92      8    0.920    0.713
    mts_hell           4096     94      6    0.940    0.683
    glcm_svm          65536     99      1    0.990    5.852
    gabor_lbp         60416     94      6    0.940    9.853    * hellinger distance
    gabor_lbp_red     17408     94      6    0.940    8.384    * hellinger distance
    dct               12544     95      5    0.950    1.572
    eigen             14400     88     12    0.880    9.725
    fisher            14400     94      6    0.940    7.068


    -------------------------------------------------------
    aberdeen   : 5 fold, 98 classes, 500 images, no preproc
    -------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels            14400    361     88    0.804    4.548
    pixels_svm         3600    390     59    0.869    5.418
    lbp               65536    338    111    0.753    9.491
    lbp_svm           65536    370     79    0.824   40.729
    lbp_hell          65536    391     58    0.871   36.348
    lbpu_red_hell      4352    325    124    0.724    4.192
    bgc1_hell         65536    368     81    0.820   36.130
    wld_hell          45056    279    170    0.621   39.998
    mts_svm            4096    394     55    0.878    3.026
    mts_hell           4096    358     91    0.797    3.458
    glcm_svm          65536    300    149    0.668   47.528
    gabor_lbp         60416    401     48    0.893   50.041
    gabor_lbp_red     17408    392     57    0.873   27.719
    dct               12544    390     59    0.869    7.945
    eigen             14400    361     88    0.804   98.133
    fisher            14400    401     48    0.893   74.918


    -------------------------------------------------------
    att       : 10 fold, 40 classes, 400 images, no preproc
    -------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels            14400    740     20    0.974    9.253
    pixels_svm         3600    744     16    0.979   14.163
    lbp               65536    718     42    0.945   21.081
    lbp_svm           65536    737     23    0.970  106.231
    lbp_hell          65536    733     27    0.964   69.259
    lbpu_red_hell      4352    715     45    0.941    7.808
    bgc1_hell         65536    737     23    0.970   69.208
    wld_hell          45056    753      7    0.991   82.885
    mts_svm            4096    738     22    0.971    6.602
    mts_hell           4096    727     33    0.957    6.791
    glcm_svm          65536    749     11    0.986  116.688
    gabor_lbp         60416    726     34    0.955   97.435
    gabor_lbp_red     17408    743     17    0.978   55.959
    dct               12544    736     24    0.968   16.370
    eigen             14400    740     20    0.974  242.654
    fisher            14400    720     40    0.947  203.818


    -------------------------------------------------------
    yale :      10 fold, 15 classes, 165 images, no preproc
    -------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels            14400    252     48    0.840    1.638
    pixels_svm         3600    267     33    0.890    1.995
    lbp               65536    241     59    0.803    4.417
    lbp_svm           65536    266     34    0.887   13.847
    lbp_hell          65536    278     22    0.927   12.835
    lbpu_red_hell      4352    240     60    0.800    2.322
    bgc1_hell         65536    271     29    0.903   12.798
    wld_hell          45056    242     58    0.807   20.284
    mts_svm            4096    266     34    0.887    1.544
    mts_hell           4096    254     46    0.847    1.765
    glcm_svm          65536    254     46    0.847   16.874
    gabor_lbp         60416    288     12    0.960   26.226
    gabor_lbp_red     17408    284     16    0.947   18.864
    dct               12544    266     34    0.887    3.577
    eigen             14400    252     48    0.840   41.665
    fisher            14400    264     36    0.880   28.760

    -------------------------------------------------------
    lfw2fun :   10 fold, 20 classes, 400 images, no preproc
    -------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels            14400    222    276    0.446    6.359
    pixels_svm         3600    357    141    0.717   15.471
    lbp               65536    361    137    0.725   15.107
    lbp_svm           65536    422     76    0.847  108.340
    lbp_hell          65536    381    117    0.765   50.016
    lbpu_red_hell      4352    302    196    0.606    6.619
    bgc1_hell         65536    375    123    0.753   49.994
    wld_hell          45056    133    365    0.267   67.101
    mts_svm            4096    417     81    0.837    6.918
    mts_hell           4096    350    148    0.703    5.937
    glcm_svm          65536    187    311    0.376  129.878
    gabor_lbp         60416    316    182    0.635   79.389
    gabor_lbp_red     17408    282    216    0.566   51.405
    dct               12544    362    136    0.727   19.294
    eigen             14400    223    275    0.448  273.521
    fisher            14400    346    152    0.695  235.756


    -------------------------------------------------------
    yaleB :      10 fold, 8 classes, 500 images, no preproc
    -------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels            14400    469     81    0.853    9.240
    pixels_svm         3600    168    382    0.305   18.879
    lbp               65536    533     17    0.969   22.312
    lbp_svm           65536    547      3    0.995  120.891
    lbp_hell          65536    542      8    0.985   72.382
    lbpu_red_hell      4352    534     16    0.971    8.887
    bgc1_hell         65536    540     10    0.982   70.654
    wld_hell          45056    247    303    0.449   91.412
    mts_svm            4096    547      3    0.995    7.790
    mts_hell           4096    540     10    0.982    7.391
    glcm_svm          65536    195    355    0.355  122.773
    gabor_lbp         60416    502     48    0.913  110.086
    gabor_lbp_red     17408    457     93    0.831   67.366
    dct               12544    195    355    0.355   21.210
    eigen             14400    469     81    0.853  498.872
    fisher            14400    537     13    0.976  446.668


    -------------------------------------------------------
    sheffield:  10 fold, 19 classes, 500 images, no preproc
    -------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels            14400    581      7    0.988    9.293
    pixels_svm         3600    581      7    0.988   19.362
    lbp               65536    572     16    0.973   20.524
    lbp_svm           65536    577     11    0.981  138.316
    lbp_hell          65536    576     12    0.980   73.784
    lbpu_red_hell      4352    551     37    0.937    9.595
    bgc1_hell         65536    576     12    0.980   73.899
    wld_hell          45056    587      1    0.998   91.402
    mts_svm            4096    578     10    0.983    9.442
    mts_hell           4096    570     18    0.969    7.760
    glcm_svm          65536    582      6    0.990  146.904
    gabor_lbp         60416    572     16    0.973  111.585
    gabor_lbp_red     17408    577     11    0.981   66.303
    dct               12544    578     10    0.983   22.356
    eigen             14400    578     10    0.983  468.160
    fisher            14400    535     53    0.910  412.951
