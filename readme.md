
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
