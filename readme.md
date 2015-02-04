
this started out as a try to improve opencv's lbph facereco, 

but meanwhile it morphed into a testbed for comparing preprocessing/extraction/classification methods

it all builds on top of opencv30 .

     the pipeline is:
        (preprocessing) -> filter -> reductor -> classifier (or verifier)

-----------------------------------------------------

4 projects in here:
* online.cpp : realtime webcam app with online training
* duel.cpp : shootout of different (identification) pipeline combinations
* fr_lfw_benchmark.cpp: the opencv (verification) lfw benchmark (from contrib/datasets)
* frontalize.cpp: 3d/2d frontal face alignment lib/standalone tool (using [dlib](http://sourceforge.net/projects/dclib/files/dlib/) landmarks)

------------------------------------------------------

####results:

------------------------------------------------------


    --------------------------------------------------------------
    aberdeen:          5 fold, 98 classes, 500 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    361     88    0.804    4.568    * 120x120
    pixels_svm         3600    390     59    0.869    5.552    * 60x60, multi-class, single svm
    pixels_multi       3600      0    449    0.000   14.453    * 60x60, multi-svm, single class
    lbp_L2            65536    338    111    0.753    9.590
    lbp_svm           65536    370     79    0.824   40.966
    lbp_hell          65536    391     58    0.871   36.448    * hellinger distance
    lbpu_red_hell      4352    325    124    0.724    4.255    * reduced uniform2 (17 bins)
    fplbp_svm          4096    398     51    0.886    3.517
    tplbp_svm         65536    374     75    0.833   41.345
    bgc1_hell         65536    368     81    0.820   36.526
    wld_hell          45056    279    170    0.621   40.767
    mts_svm            4096    394     55    0.878    3.086
    mts_hell           4096    358     91    0.797    3.521
    stu_svm           16384    410     39    0.913    8.623
    glcm_svm          65536    300    149    0.668   46.678
    gabor_hell        60416    401     48    0.893   50.987    * 4 gabor filters -> lbp
    gabor_red         17408    392     57    0.873   27.797
    gabor_svm         60416    412     37    0.918   49.143
    dct_L2            12544    363     86    0.808    3.665
    dct_svm           12544    390     59    0.869    7.435
    orb_L1             2592    424    118    0.782    2.803    * 600 images
    sift_L2          115200    417     32    0.929  251.698
    eigen             14400    361     88    0.804   97.675
    fisher            14400    401     48    0.893   77.305
    --------------------------------------------------------------
    att:               10 fold, 40 classes, 400 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    740     20    0.974    8.931
    pixels_svm         3600    744     16    0.979   13.711
    pixels_multi       3600    705     55    0.928   41.910
    lbp_L2            65536    718     42    0.945   19.584
    lbp_svm           65536    737     23    0.970  107.094
    lbp_hell          65536    733     27    0.964   70.904
    lbpu_red_hell      4352    715     45    0.941    8.528
    fplbp_svm          4096    735     25    0.967    7.733
    tplbp_svm         65536    734     26    0.966  107.602
    bgc1_hell         65536    737     23    0.970   71.093
    wld_hell          45056    753      7    0.991   82.840
    mts_svm            4096    738     22    0.971    6.449
    mts_hell           4096    727     33    0.957    6.906
    stu_svm           16384    724     36    0.953   19.571
    glcm_svm          65536    749     11    0.986  115.771
    gabor_hell        60416    726     34    0.955  101.071
    gabor_red         17408    743     17    0.978   56.272
    gabor_svm         60416    727     33    0.957  115.706
    dct_L2            12544    730     30    0.961    8.248
    dct_svm           12544    736     24    0.968   17.566
    orb_L1             2592    619    141    0.814    4.775
    sift_L2          115200    723     37    0.951  533.849
    eigen             14400    740     20    0.974  246.252
    fisher            14400    720     40    0.947  207.481
    --------------------------------------------------------------    lfw-funneled subset 
    lfw2fun:           10 fold, 20 classes, 400 images, no preproc  * of persons with >= 10 images, 
    --------------------------------------------------------------    cropped to 90x90 center
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    222    276    0.446    6.425
    pixels_svm         3600    357    141    0.717   15.578
    pixels_multi       3600    281    217    0.564   37.059
    lbp_L2            65536    361    137    0.725   14.946
    lbp_svm           65536    422     76    0.847  108.177
    lbp_hell          65536    381    117    0.765   50.370
    lbpu_red_hell      4352    302    196    0.606    6.586
    fplbp_svm          4096    404     94    0.811    8.052
    tplbp_svm         65536    418     80    0.839  116.441
    bgc1_hell         65536    375    123    0.753   50.531
    wld_hell          45056    133    365    0.267   67.659
    mts_svm            4096    417     81    0.837    6.952
    mts_hell           4096    350    148    0.703    5.495
    stu_svm           16384    336    162    0.675   21.439
    glcm_svm          65536    187    311    0.376  128.983
    gabor_hell        60416    316    182    0.635   83.028
    gabor_red         17408    282    216    0.566   51.788
    gabor_svm         60416    335    163    0.673  116.583
    dct_L2            12544    230    268    0.462    7.400
    dct_svm           12544    362    136    0.727   18.474
    orb_L1             2592    322    176    0.647    4.553
    sift_L2          115200    367    131    0.737  514.785
    eigen             14400    223    275    0.448  272.145
    fisher            14400    346    152    0.695  233.791
    --------------------------------------------------------------
    senthil:           10 fold, 5 classes, 80 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400     87     13    0.870    0.404
    pixels_svm         3600     94      6    0.940    0.522
    pixels_multi       3600     93      7    0.930    1.111
    lbp_L2            65536     93      7    0.930    1.597
    lbp_svm           65536     92      8    0.920    4.607
    lbp_hell          65536     94      6    0.940    3.050
    lbpu_red_hell      4352     94      6    0.940    0.965
    tplbp_svm         65536     94      6    0.940    5.083
    bgc1_hell         65536     94      6    0.940    3.055
    wld_hell          45056     98      2    0.980    8.841
    mts_svm            4096     92      8    0.920    1.275
    mts_hell           4096     94      6    0.940    0.713
    stu_svm           16384     96      4    0.960    1.399
    glcm_svm          65536     99      1    0.990    6.209
    gabor_hell        60416     94      6    0.940   10.023
    gabor_red         17408     94      6    0.940    8.508
    gabor_svm         60416     94      6    0.940   10.767
    dct_L2            12544     87     13    0.870    1.248
    dct_svm           12544     95      5    0.950    1.589
    orb_L1             2592     86     14    0.860    0.887
    sift_L2          115200     94      6    0.940  103.974
    eigen             14400     88     12    0.880    9.676
    fisher            14400     94      6    0.940    7.083
    --------------------------------------------------------------
    sheffield:         10 fold, 19 classes, 500 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    581      7    0.988    9.336
    pixels_svm         3600    581      7    0.988   19.742
    pixels_multi       3600    572     16    0.973   34.272
    lbp_L2            65536    572     16    0.973   22.468
    lbp_svm           65536    577     11    0.981  145.740
    lbp_hell          65536    576     12    0.980   74.394
    lbpu_red_hell      4352    551     37    0.937    9.455
    fplbp_svm          4096     93      7    0.930    1.013
    tplbp_svm         65536    577     11    0.981  130.829
    bgc1_hell         65536    576     12    0.980   75.753
    wld_hell          45056    587      1    0.998   92.920
    mts_svm            4096    578     10    0.983    9.411
    mts_hell           4096    570     18    0.969    7.728
    stu_svm           16384    568     20    0.966   28.108
    glcm_svm          65536    582      6    0.990  152.146
    gabor_hell        60416    572     16    0.973  114.289
    gabor_red         17408    577     11    0.981   67.891
    gabor_svm         60416    576     12    0.980  152.041
    dct_L2            12544    577     11    0.981    8.738
    dct_svm           12544    578     10    0.983   22.977
    orb_L1             2592    587     91    0.866    6.828      * 600 images
    sift_L2          115200    575     13    0.978  653.038
    eigen             14400    578     10    0.983  472.288
    fisher            14400    537     51    0.913  424.227
    --------------------------------------------------------------
    yale:              10 fold, 15 classes, 165 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    252     48    0.840    1.651
    pixels_svm         3600    267     33    0.890    1.966
    pixels_multi       3600    262     38    0.873    5.184
    lbp_L2            65536    241     59    0.803    4.466
    lbp_svm           65536    266     34    0.887   14.004
    lbp_hell          65536    278     22    0.927   12.964
    lbpu_red_hell      4352    240     60    0.800    2.302
    fplbp_svm          4096    273     27    0.910    2.233
    tplbp_svm         65536    260     40    0.867   14.470
    bgc1_hell         65536    271     29    0.903   12.908
    wld_hell          45056    242     58    0.807   20.438
    mts_svm            4096    266     34    0.887    1.570
    mts_hell           4096    254     46    0.847    1.753
    stu_svm           16384    262     38    0.873    3.866
    glcm_svm          65536    254     46    0.847   17.286
    gabor_hell        60416    288     12    0.960   26.436
    gabor_red         17408    284     16    0.947   19.135
    gabor_svm         60416    282     18    0.940   26.120
    dct_L2            12544    249     51    0.830    2.758
    dct_svm           12544    266     34    0.887    3.975
    orb_L1             2592    285     15    0.950    1.841
    sift_L2          115200    295      5    0.983  215.252
    eigen             14400    252     48    0.840   40.255
    fisher            14400    257     43    0.857   28.471
    --------------------------------------------------------------
    yaleB:             10 fold, 8 classes, 500 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    469     81    0.853    9.006
    pixels_svm         3600    168    382    0.305   16.055
    pixels_multi       3600    411    139    0.747   23.350
    lbp_L2            65536    533     17    0.969   19.935
    lbp_svm           65536    547      3    0.995  111.467
    lbp_hell          65536    542      8    0.985   72.184
    lbpu_red_hell      4352    534     16    0.971    9.592
    fplbp_svm          4096    548      2    0.996    8.397
    tplbp_svm         65536    549      1    0.998  100.864
    bgc1_hell         65536    540     10    0.982   72.937
    wld_hell          45056    247    303    0.449   91.548
    mts_svm            4096    547      3    0.995    7.994
    mts_hell           4096    540     10    0.982    7.618
    stu_svm           16384    407    143    0.740   25.794
    glcm_svm          65536    195    355    0.355  124.902
    gabor_hell        60416    502     48    0.913  110.900
    gabor_red         17408    457     93    0.831   68.337
    gabor_svm         60416    518     32    0.942  138.727
    dct_L2            12544    473     77    0.860    9.707
    dct_svm           12544    195    355    0.355   21.497
    orb_L1             2592    650     10    0.985    7.126
    sift_L2          115200    534     16    0.971  665.632
    eigen             14400    469     81    0.853  503.998
    fisher            14400    537     13    0.976  454.957
    --------------------------------------------------------------
    f94gender:         10 fold, 2 classes, 484 images, no preproc
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    461     29    0.941    7.877
    pixels_svm         3600    467     23    0.953    7.566
    pixels_cosine     14400    461     29    0.941    5.698
    pixels_multi       3600    474     16    0.967    5.090
    lbp_L2            65536    473     17    0.965   18.827
    lbp_svm           65536    456     34    0.931   50.091
    lbp_hell          65536    477     13    0.973   62.588
    lbpu_red_hell      4352    460     30    0.939    8.651
    fplbp_svm          4096    471     19    0.961    8.891
    tplbp_svm         65536    465     25    0.949   39.720
    bgc1_hell         65536    471     19    0.961   62.711
    wld_hell          45056    462     28    0.943   84.162
    mts_svm            4096    467     23    0.953    5.227
    mts_hell           4096    463     27    0.945    6.730
    stu_svm           16384    450     40    0.918   13.858
    glcm_svm          65536    452     38    0.922   61.488
    gabor_red         17408    450     40    0.918   64.076
    gabor_svm         60416    420     70    0.857   85.903
    dct_cosine        12544    465     25    0.949   10.211
    dct_L2            12544    456     34    0.931    8.284
    dct_svm           12544    454     36    0.927   12.905
    orb_L1             2592    481    109    0.815    6.886
    sift_L2          115200    470     20    0.959  647.266
    eigen             14400    462     28    0.943  462.823
    fisher            14400    349    141    0.712  406.520


for those, who stayed until the end, here's a bonus track ;)

retina / parvo preprocessing !!

    --------------------------------------------------------------
    lfw2fun:           10 fold, 20 classes, 400 images, retina
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    416     82    0.835    6.411
    pixels_svm         3600    462     36    0.928   11.517
    pixels_cosine     14400    438     60    0.880    4.640
    pixels_multi       3600    416     82    0.835   29.548
    lbp_L2            65536    460     38    0.924   16.195
    lbp_svm           65536    481     17    0.966   93.569
    lbp_hell          65536    476     22    0.956   51.524
    lbpu_red_hell      4352    427     71    0.857    6.956
    fplbp_svm          4096    466     32    0.936    7.607
    tplbp_svm         65536    480     18    0.964  103.811
    bgc1_hell         65536    473     25    0.950   51.583
    mts_svm            4096    482     16    0.968    6.071
    mts_hell           4096    467     31    0.938    5.568
    stu_svm           16384    409     89    0.821   17.180
    glcm_svm          65536    303    195    0.608  116.967
    gabor_red         17408    402     96    0.807   49.522
    gabor_svm         60416    444     54    0.892  100.774
    dct_cosine        12544    435     63    0.873    8.431
    dct_L2            12544    413     85    0.829    6.648
    dct_svm           12544    462     36    0.928   14.496
    orb_L1             2048    445     53    0.894    3.957
    sift_L2           115200   467     31    0.938  513.593
    eigen             14400    416     82    0.835  267.483
    fisher            14400    474     24    0.952  228.848
    --------------------------------------------------------------
    yaleB:             10 fold, 8 classes, 500 images, retina
    --------------------------------------------------------------
    [method]       [f_bytes]  [pos]  [neg]   [hit]   [time]
    pixels_L2         14400    538     12    0.978    8.995
    pixels_svm         3600    550      0    1.000    8.943
    pixels_cosine     14400    548      2    0.996    6.481
    pixels_multi       3600    549      1    0.998   17.433
    lbp_L2            65536    550      0    1.000   22.942
    lbp_svm           65536    550      0    1.000   84.998
    lbp_hell          65536    550      0    1.000   76.246
    lbpu_red_hell      4352    548      2    0.996    9.736
    fplbp_svm          4096    550      0    1.000    8.158
    bgc1_hell         65536    550      0    1.000   76.223
    mts_svm            4096    549      1    0.998    6.373
    mts_hell           4096    550      0    1.000    7.767
    stu_svm           16384    529     21    0.962   14.969
    glcm_svm          65536    334    216    0.607  133.906
    gabor_red         17408    517     33    0.940   66.573
    gabor_svm         60416    545      5    0.991  101.786
    dct_cosine        12544    548      2    0.996   11.243
    dct_L2            12544    537     13    0.976    8.709
    dct_svm           12544    550      0    1.000   13.708
    orb_L1             2048    549      1    0.998    5.064
    sift_L2          115200    550      0    1.000  658.560
    eigen             14400    538     12    0.978  499.964
    fisher            14400    550      0    1.000  451.933
    sift_gfft_svm     32768    550      0    1.000   66.906
