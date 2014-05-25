#ifndef _lalalala_
#define _lalalala_

#include "opencv2/contrib.hpp"

extern cv::Ptr<cv::FaceRecognizer> createLBPHFaceRecognizer2(int radius, int neighbors, int grid_x, int grid_y, double threshold, bool uniform );
extern cv::Ptr<cv::FaceRecognizer> createMinMaxFaceRecognizer(int raster,int compression );
extern cv::Ptr<cv::FaceRecognizer> createLinearFaceRecognizer(int NORM);
extern cv::Ptr<cv::FaceRecognizer> createCombinedLBPHFaceRecognizer(int grid_x, int grid_y, double threshold);
extern cv::Ptr<cv::FaceRecognizer> createVarLBPFaceRecognizer(int grid_x, int grid_y, double threshold);
extern cv::Ptr<cv::FaceRecognizer> createLTPHFaceRecognizer(int ic, int grid_x, int grid_y, double threshold);
extern cv::Ptr<cv::FaceRecognizer> createClbpDistFaceRecognizer(double threshold);
extern cv::Ptr<cv::FaceRecognizer> createWLDFaceRecognizer(int gridx,int gridy,double threshold);
extern cv::Ptr<cv::FaceRecognizer> createMomFaceRecognizer(int grid,int w);
extern cv::Ptr<cv::FaceRecognizer> createZernikeFaceRecognizer(int grid, int nfeatures);
extern cv::Ptr<cv::FaceRecognizer> createAnnFaceRecognizer();
extern cv::Ptr<cv::FaceRecognizer> createSvmFaceRecognizer(int preprocessing);

#endif _lalalala_
