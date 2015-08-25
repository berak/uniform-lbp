/* 
 * File:   RBMglu.cpp
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#include <opencv2/core.hpp>

#include "RBMglu.h"


namespace artelab
{
    cv::RNG rng;

    RBMglu::RBMglu(const int nVis, const int nHid, float(*sampleFun)(float) ) : RBM(nVis, nHid, sampleFun)
    { }

    RBMglu::RBMglu(const std::string filepath, float(*sampleFun)(float)) : RBM(filepath, sampleFun)
    { }


    void RBMglu::hidden_to_visible_linear(const cv::Mat& hidden_state, cv::Mat& visible_probabilities)
    {
        visible_probabilities = weights.t() * hidden_state + cv::repeat(vis_bias, 1, hidden_state.cols);
    }

    void RBMglu::gaussian_noise(const cv::Mat& visible_probabilities, cv::Mat& visible_state)
    {
        cv::Mat noise(visible_probabilities.size(), CV_32F);
        rng.fill(noise, cv::RNG::NORMAL, 0, 1);
        visible_state = visible_probabilities + noise;
    }

    inline
    float logistic(float x)
    {
        return 1.0f / (1.0f + exp(-x));
    }

    void RBMglu::visible_to_hidden_probabilities_logistic(const cv::Mat& visible_state, cv::Mat& hidden_probabilities)
    {
        hidden_probabilities = weights * visible_state + cv::repeat(hid_bias, 1, visible_state.cols);
        matfunc<float>(hidden_probabilities, logistic, hidden_probabilities);
    }

    void RBMglu::binary_sample_hidden_state(const cv::Mat& visible_state, cv::Mat& hidden_state)
    {
        cv::Mat hidden_probabilities;
        visible_to_hidden_probabilities_logistic(visible_state, hidden_probabilities);
        sample_binary(hidden_probabilities, hidden_state);
    }

    /* Contrastive Divergence
     * "data_cases" eventually binary matrix of size <number of visible units> x <number of data cases>
     * OUT "gradient": gradient approximation produced by CD-1. Size <number of hidden units> x <number of visible units>
     */
    void RBMglu::cd(const int k, const cv::Mat& data_cases, Gradient& gradient)
    {
        CV_Assert( k > 0 );

        cv::Mat visible_state, hidden_state;
        cv::Mat visible_input;
        Gradient gradient_0, gradient_k;

        data_cases.copyTo(visible_state);

        int mcmc_round = 0;
        while(1)
        {
            if(mcmc_round == 0) // get <Si*Sj>_0 -- Positive statistics
            {
                binary_sample_hidden_state(visible_state, hidden_state);
                goodness_gradient(visible_state, hidden_state, gradient_0);
            }
            else if(mcmc_round == k) // get <Si*Sj>_k -- Negative statistics
            {
                visible_to_hidden_probabilities_logistic(visible_state, hidden_state);
                goodness_gradient(visible_input, hidden_state, gradient_k);
                break;
            }
            else
            {
                binary_sample_hidden_state(visible_state, hidden_state);
            }

            hidden_to_visible_linear(hidden_state, visible_input);
            gaussian_noise(visible_input, visible_state);

            mcmc_round++;
        }

        gradient = gradient_0 - gradient_k;
    }


    void RBMglu::hidden_activations_for(const cv::Mat& pattern, cv::Mat& activations, bool probabilities, int mc_steps)
    {
        cv::Mat visible_state = pattern.t();

        for(int step=0; step < mc_steps; step++)
        {
            if(step == mc_steps-1 && probabilities)
            {
                visible_to_hidden_probabilities_logistic(visible_state, activations);
                break;
            }

            binary_sample_hidden_state(visible_state, activations);

            hidden_to_visible_linear(activations, visible_state);
            gaussian_noise(visible_state, visible_state);
        }
        activations = activations.t();
    }

    void RBMglu::reconstruct(const cv::Mat& pattern, cv::Mat& reconstruction, int mc_steps)
    {
        reconstruction = pattern.t();
        cv::Mat hidden_state;

        for(int step=0; step < mc_steps; step++)
        {
            binary_sample_hidden_state(reconstruction, hidden_state);
            hidden_to_visible_linear(hidden_state, reconstruction);

            if(step == mc_steps-1)
                break;

            gaussian_noise(reconstruction, reconstruction);
        }
        reconstruction = reconstruction.t();
    }

    float RBMglu::avg_free_energy(const cv::Mat& patterns)
    {
        cv::Mat tmp, a = cv::repeat(vis_bias.t(), patterns.rows, 1);

        cv::pow(patterns - a, 2, tmp);
        cv::reduce(tmp / 2, tmp, 1, CV_REDUCE_SUM);

        float first_term = float(cv::mean(tmp)[0]);

        tmp = patterns * weights.t() + cv::repeat(hid_bias.t(), patterns.rows, 1);
        cv::exp(tmp, tmp);
        cv::log(tmp + 1, tmp);
        cv::reduce(tmp, tmp, 1, CV_REDUCE_SUM);

        float second_term = float(cv::mean(tmp)[0]);

        return first_term - second_term;
    }

    std::string RBMglu::description()
    {
        std::ostringstream ss;
        ss << "GB Restricted Boltzmann Machine, Visible=" << num_visible() << " Hidden=" << num_hidden();
        return ss.str();
    }

}