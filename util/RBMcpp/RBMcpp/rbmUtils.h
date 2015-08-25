/* 
 * File:   rbmUtils.h
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#ifndef RBMUTILS_H
#define	RBMUTILS_H

#include <opencv2/core.hpp>
#include "RBM.h"
//#include "../utils/FileInfo.h"

namespace artelab
{
    
    cv::Mat weightImageOf(RBM* rbm, const int hidden_unit_index, const cv::Size base_size, const bool rgb=false);
    
    cv::Mat weights_image(RBM* rbm, const cv::Size base_size, const bool rgb=false, const int images_column=0);
    cv::Mat show_bases(RBM* rbm, cv::Size base_size, const bool rgb=false, cv::Size canvas=cv::Size(800,800));

    float average_mse(RBM* rbm, const cv::Mat& patterns);
    
    void feature_patterns(RBM* rbm, cv::Mat patterns, cv::Mat& featurePatterns, const bool probabilities=true);
    
    cv::Mat weight_distribution(RBM* rbm, int bins, float& min, float& max);
    
    cv::Mat updates_distribution(RBM* rbm, int bins, float& min, float& max);
    
    //void save_histogram_image(cv::Mat hist, FileInfo file = FileInfo(), std::string title = "Weights distribution");
    //cv::Mat save_and_load_histogram_image(cv::Mat hist, FileInfo file = FileInfo(), std::string title = "Weights distribution", bool remove_tmp_image=true);
    
    void train_and_monitor_learning(RBM* rbm);
}
#endif	/* RBMUTILS_H */

