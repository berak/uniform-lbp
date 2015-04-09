#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
using namespace std;

//
// original code from:
// https://github.com/Ldpe2G/PCANet
//
// * changed everything to float Mat's (lower memory footprint)
// * removed the indexing
//
class PCANet
{
    struct Stage
    {
        int numFilters;
        int histBlockSize;
        cv::Mat filters;
    };

    int dimensionLDA;
    int numStages;
    int patchSize;
    double blkOverLapRatio;
    
    vector<Stage> stages;

    cv::Mat projVecPCA;
    cv::Mat projVecLDA;
    cv::Mat hashingHist(const vector<cv::Mat> &Imgs) const;

public :

    PCANet(int p=7) : dimensionLDA(0),numStages(0),patchSize(p),blkOverLapRatio(0) {}


    cv::Mat trainPCA(vector<cv::Mat> &InImg, bool extract_feature=true);
    cv::Mat trainLDA(const cv::Mat &features, const cv::Mat &labels);

    cv::Mat extract(const cv::Mat &img) const;

    cv::String settings() const;

    bool save(const cv::String &fn) const;
    bool load(const cv::String &fn);

    int addStage(int nfil, int blocs);
};


