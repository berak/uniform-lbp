/* 
 * File:   RBMglu.h
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#ifndef RBMGLU_H
#define	RBMGLU_H

#include "RBM.h"

namespace artelab
{
    class RBMglu : public RBM
    {
    public:

        RBMglu() {}
        RBMglu(const int nVis, const int nHid, float (*sampleFun)(float)=NULL);
        RBMglu(const std::string filepath, float (*sampleFun)(float)=NULL);

        std::string description();

        void hidden_activations_for(const cv::Mat& pattern, cv::Mat& activations, bool probabilities=true, int mc_steps=1);
        void reconstruct(const cv::Mat& pattern, cv::Mat& reconstruction, int mc_steps = 1);
        float avg_free_energy(const cv::Mat& patterns);


    protected:

        void cd(const int k, const cv::Mat& data_cases, Gradient& gradient);
        void hidden_to_visible_linear(const cv::Mat& hidden_state, cv::Mat& visible_probabilities);
        void gaussian_noise(const cv::Mat& visible_probabilities, cv::Mat& visible_state);
        void visible_to_hidden_probabilities_logistic(const cv::Mat& visible_state, cv::Mat& hidden_probabilities);
        void binary_sample_hidden_state(const cv::Mat& visible_state, cv::Mat& hidden_state);

    };
}

#endif	/* RBMGLU_H */

