#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
using std::vector;


//
// original code from:
// https://github.com/Ldpe2G/PCANet
//
// * changed everything to float Mat's (lower memory footprint)
// * removed the indexing
// * enabled variable stage count
//
class PCANet
{
    struct Stage
    {
        int numFilters;
        int histBlockSize;
        cv::Mat filters;
    };

    // pca filter bank data
    vector<Stage> stages;
    int numStages;
    int patchSize;
    double blockOverLapRatio;    

    // (optional) lda data
    cv::Mat projVecPCA;
    cv::Mat projVecLDA;

    cv::Mat hashingHist(const vector<cv::Mat> &Imgs) const;

    cv::String _type;

public :

    // larger patchSize seems to improve more than more filters
    PCANet(int p=7) : numStages(0), patchSize(p), blockOverLapRatio(0), _type("Pca") {}
    int addStage(int nfil, int blocs);

    void randomProjection();
    void waveProjection(float freq=1.0f);
    //void haarProjection(float freq=1.0f);

    cv::Mat extract(const cv::Mat &img) const;

    cv::Mat trainPCA(vector<cv::Mat> &InImg, bool extract_feature=true);
    cv::Mat trainLDA(const cv::Mat &features, const cv::Mat &labels, int dimensionLDA=100);

    bool load(const cv::String &fn);
    bool save(const cv::String &fn) const;

    bool saveFilterVis(const cv::String &fn) const;
    cv::String settings() const;
};


