#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
using namespace std;

//
// original code from:
// https://github.com/Ldpe2G/PCANet
//
struct PCANet
{
    int dimensionLDA;
    int numStages;
    int patchSize;
    vector<int> numFilters;
    vector<int> histBlockSize;
    double blkOverLapRatio;
    
    vector<cv::Mat> filters;

    cv::Mat projVecPCA;
    cv::Mat projVecLDA;

    cv::Mat hashingHist(const vector<cv::Mat> &Imgs) const;

    cv::Mat trainPCA(vector<cv::Mat> &InImg, bool extract_feature=true);
    cv::Mat trainLDA(const cv::Mat &features, const cv::Mat &labels);

    cv::Mat extract(const cv::Mat &img) const;

    cv::String settings() const;

    bool save(const cv::String &fn) const;
    bool load(const cv::String &fn);
};


