/* 
 * File:   RBMbb.h
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 * Created on 21 Jan 2013, 12.01
 */

#ifndef RBMbb_H
#define	RBMbb_H

#include <opencv2/core.hpp>
#include "RBM.h"

namespace artelab
{
    class RBMbb : public RBM
    {

    public:

        RBMbb();
        RBMbb(const int nVis, const int nHid, float (*sampleFun)(float)=NULL);
        RBMbb(const std::string filepath, float (*sampleFun)(float)=NULL);
        RBMbb(cv::Mat w, cv::Mat vb, cv::Mat hb, float (*sampleFun)(float)=NULL);
        virtual ~RBMbb() { }

        std::string description();

        void hidden_activations_for(const cv::Mat& pattern, cv::Mat& activations, bool probabilities=true, int mc_steps=1);
        void reconstruct(const cv::Mat& pattern, cv::Mat& reconstruction, int mc_steps=1);
        float avg_free_energy(const cv::Mat& patterns);

        /* Public for test purposes */
        void cd(const int k, const cv::Mat& data_cases, Gradient& gradient);
        void visible_to_hidden_probabilities_logistic(const cv::Mat& visible_state, cv::Mat& hidden_probabilities);
        void hidden_to_visible_probabilities_logistic(const cv::Mat& hidden_state, cv::Mat& visible_probabilities);

    protected:

        void binary_sample_hidden_state(const cv::Mat& visible_state, cv::Mat& hidden_state);
        void binary_sample_visible_state(const cv::Mat& hidden_state, cv::Mat& visible_state);


    };

}

#endif	/* RBMbb_H */

