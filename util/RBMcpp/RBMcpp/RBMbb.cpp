/* 
 * File:   RBMbb.cpp
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#include "RBMbb.h"
#include <cmath>
extern cv::RNG rng;

typedef float (*sample_fun_ptr)(float);

namespace artelab
{
    double log2( double n )  
    {  
        // log(n)/log(2) is log2.  
        return log( n ) / log( 2.0 );  
    }


    RBMbb::RBMbb() : RBM()
    { }

    RBMbb::RBMbb(const int nVis, const int nHid, float (*sampleFun)(float)) :
                  RBM(nVis, nHid, sampleFun)
    { }


    RBMbb::RBMbb(const std::string filepath, sample_fun_ptr sampleFun) :
                  RBM(filepath, sampleFun)
    { }


    RBMbb::RBMbb(cv::Mat w, cv::Mat vb, cv::Mat hb, float(*sampleFun)(float)) :
                  RBM(w, vb, hb, sampleFun)
    { }

    std::string RBMbb::description()
    {
        std::ostringstream ss;
        ss << "BB Restricted Boltzmann Machine, Visible=" << num_visible() << " Hidden=" << num_hidden();
        return ss.str();
    }

    inline
    float logistic(float x)
    {
        return 1.0f / (1.0f + exp(-x));
    }

    /* "visible_state": eventually binary matrix of size <number visible units> x <number of data cases>
     * OUT "hidden_probabilities": probabilities of activation of the neurons conditional the visible state, 
     *                             size <number of hidden units> x <number of data cases>
     */
    void RBMbb::visible_to_hidden_probabilities_logistic(const cv::Mat& visible_state, cv::Mat& hidden_probabilities)
    {
        hidden_probabilities = weights * visible_state + cv::repeat(hid_bias, 1, visible_state.cols);
        matfunc<float>(hidden_probabilities, logistic, hidden_probabilities);
    }

    /* "hidden_state": binary matrix of size <number of hidden units> x <number of data cases>
     * OUT "visible_probabilities": probabilities of activation of the neurons conditional the hidden state,
     *                              size <number of visible units> by <number of data cases>
     */
    void RBMbb::hidden_to_visible_probabilities_logistic(const cv::Mat& hidden_state, cv::Mat& visible_probabilities)
    {
        visible_probabilities = weights.t() * hidden_state + cv::repeat(vis_bias, 1, hidden_state.cols);
        matfunc<float>(visible_probabilities, logistic, visible_probabilities);
    }

    void RBMbb::binary_sample_hidden_state(const cv::Mat& visible_state, cv::Mat& hidden_state)
    {
        cv::Mat hidden_probabilities;
        visible_to_hidden_probabilities_logistic(visible_state, hidden_probabilities);
        sample_binary(hidden_probabilities, hidden_state);
    }

    void RBMbb::binary_sample_visible_state(const cv::Mat& hidden_state, cv::Mat& visible_state)
    {
        cv::Mat visible_probabilities;
        hidden_to_visible_probabilities_logistic(hidden_state, visible_probabilities);
        sample_binary(visible_probabilities, visible_state);
    }


    /* Contrastive Divergence
     * "data_cases" eventually binary matrix of size <number of visible units> x <number of data cases>
     * OUT "gradient": gradient approximation produced by CD-1. Size <number of hidden units> x <number of visible units>
     */
    void RBMbb::cd(const int k, const cv::Mat& data_cases, Gradient& gradient)
    {
        CV_Assert( k > 0 );

        cv::Mat visible_state, hidden_state;
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
                goodness_gradient(visible_state, hidden_state, gradient_k);
                break;
            }
            else
            {
                // TODO: maybe there should use probabilities rather than sampling.
                // The guide recommends to sample when coming from real data
                // and to use probabilities when using reconstructions from the model
                binary_sample_hidden_state(visible_state, hidden_state);
            }

            hidden_to_visible_probabilities_logistic(hidden_state, visible_state);

            mcmc_round++;
        }

        gradient = gradient_0 - gradient_k;
    }


    /* "pattern": row input vector, size <1>x<number visible units>
     * OUT "activation": row vector, size <1>x<number hidden units>
     * "mc_steps": number of markov chain monte carlo steps before sampling. DEFAULT: 1
     */
    void RBMbb::hidden_activations_for(const cv::Mat& pattern, cv::Mat& activations, bool probabilities, int mc_steps)
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
            binary_sample_visible_state(activations, visible_state);
        }
        activations = activations.t();
    }

    /* "pattern": row input vector, size <1>x<number visible units>
     * OUT "reconstruction": row vector, size <1>x<number visible units>
     * "mc_steps": number of markov chain monte carlo steps before sampling. DEFAULT: 1
     */
    void RBMbb::reconstruct(const cv::Mat& pattern, cv::Mat& reconstruction, int mc_steps)
    {
        reconstruction = pattern.t();
        cv::Mat hidden_state;

        for(int step=0; step < mc_steps; step++)
        {
            binary_sample_hidden_state(reconstruction, hidden_state);
            if(step == mc_steps-1)
            {
                hidden_to_visible_probabilities_logistic(hidden_state, reconstruction);
                break;
            }
            binary_sample_visible_state(hidden_state, reconstruction);
        }
        reconstruction = reconstruction.t();
    }

    /* Average free energy for each row patter "patterns".
     * The free energy of a vector "v" is the energy that a single configuration 
     * would need to have in order to have the same probability as all of the 
     * configurations that contain "v".
     * See http://www.cs.utoronto.ca/~hinton/absps/guideTR.pdf chapter 16
     */
    float RBMbb::avg_free_energy(const cv::Mat& patterns)
    {
        CV_Assert(patterns.rows > 0 && patterns.cols == num_visible());

        float ret = 0;
        for(int r=0; r < patterns.rows; r++)
        {
            cv::Mat v = patterns.row(r);

            float f1 = cv::Mat(v * vis_bias).at<float>(0,0);

            float f2 = 0;
            cv::Mat x = weights * v.t() + hid_bias; // hidden activations (column vector)
            for(int j=0; j < num_hidden(); j++)
            {
                float x_j = x.at<float>(j, 0);
                f2 += log2(1 + exp(x_j));
            }

            ret = -(f1 + f2) / patterns.rows;
        }

        return ret;
    }
}