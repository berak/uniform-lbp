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
	int NumStages;
	int PatchSize;
	vector<int> NumFilters;
	vector<int> HistBlockSize;
	double BlkOverLapRatio;
    
	vector<cv::Mat> Filters;

    cv::Mat ProjVecPCA;
    cv::Mat ProjVecLDA;

    cv::Mat hashingHist(const vector<cv::Mat> &Imgs) const;

    cv::Mat trainPCA(vector<cv::Mat> &InImg, bool extract_feature=true);
    cv::Mat trainLDA(const cv::Mat &features, const cv::Mat &labels);

    cv::Mat extract(const cv::Mat &img) const;

    cv::String settings() const;

    bool save(const cv::String &fn) const;
    bool load(const cv::String &fn);
};


