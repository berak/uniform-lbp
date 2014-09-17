
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



    method              hit    miss hitrate
    --------------------------------------------
    yaleB.txt
    --------------------------------------------
    pixels              391     49  0.889
    pixels_svm          176    264  0.400
    lbp                 431      9  0.980
    lbp_svm             435      5  0.989
    lbp_hell            437      3  0.993
    lbpu_red_hell       427     13  0.970
    bgc1_hell           435      5  0.989
    wld_hell            221    219  0.502
    mts_svm             435      5  0.989
    mts_hell            434      6  0.986
    glcm_svm            171    269  0.389


    --------------------------------------------
    att.txt
    --------------------------------------------
    pixels              740     20  0.974
    pixels_svm          744     16  0.979
    lbp                 718     42  0.945
    lbp_svm             737     23  0.970
    lbp_hell            733     27  0.964
    lbpu_red_hell       715     45  0.941
    bgc1_hell           737     23  0.970
    wld_hell            753      7  0.991
    mts_svm             738     22  0.971
    mts_hell            727     33  0.957
    glcm_svm            749     11  0.986

    --------------------------------------------
    lfw2fun.txt
    --------------------------------------------
    pixels              222    276  0.446
    pixels_svm          357    141  0.717
    lbp                 361    137  0.725
    lbp_svm             422     76  0.847
    lbp_hell            381    117  0.765
    lbpu_red_hell       302    196  0.606
    bgc1_hell           375    123  0.753
    wld_hell            133    365  0.267
    mts_svm             417     81  0.837
    mts_hell            350    148  0.703
    glcm_svm            187    311  0.376


    --------------------------------------------
    yale.txt
    --------------------------------------------
    pixels              252     48  0.840
    pixels_svm          268     32  0.893
    lbp                 241     59  0.803
    lbp_svm             266     34  0.887
    lbp_hell            278     22  0.927
    lbpu_red_hell       240     60  0.800
    bgc1_hell           271     29  0.903
    wld_hell            242     58  0.807
    mts_svm             266     34  0.887
    mts_hell            254     46  0.847
    glcm_svm            254     46  0.847


    --------------------------------------------
    brodatz.txt
    --------------------------------------------
    pixels              114    146  0.438
    pixels_svm          146    114  0.562
    lbp                 214     46  0.823
    lbp_svm             253      7  0.973
    lbp_hell            167     93  0.642
    lbpu_red_hell       156    104  0.600
    bgc1_hell           220     40  0.846
    wld_hell            260      0  1.000
    mts_svm             235     25  0.904
    mts_hell            194     66  0.746
    glcm_svm            231     29  0.888
