/* 
 * File:   RBM.cpp
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#include "RBM.h"


typedef float (*sample_fun_ptr)(float);

namespace artelab
{
    extern cv::RNG rng;

    inline
    float sample_neuron_activation(float probability)
    {
        float v = rng.uniform(0.0f, 1.0f);
        return (v - probability) < 0? 1.0f : 0.0f;
    }

    void RBM::init_rng_and_sampler(sample_fun_ptr sampleFun)
    {
        rng = cv::RNG(cv::getTickCount());

        if(sampleFun == NULL)
            _sampler = sample_neuron_activation;
        else
            _sampler = sampleFun;
    }



    RBM::RBM() 
    { }

    RBM::RBM(const int num_visible, const int num_hidden, sample_fun_ptr sampleFun) 
    {
        init_rng_and_sampler(sampleFun);
        weights = cv::Mat::zeros(num_hidden, num_visible, CV_32F);
        vis_bias = cv::Mat::zeros(num_visible, 1, CV_32F);
        hid_bias = cv::Mat::zeros(num_hidden, 1, CV_32F);
        _trained = false;
        _current_iteration = 0;
        _step_type = EPOCHS;
        _iteration_step = 1;
    }

    RBM::RBM(const std::string filepath, sample_fun_ptr sampleFun)
    {
        init_rng_and_sampler(sampleFun);
        load(filepath);
        _current_iteration = 0;
        _step_type = EPOCHS;
        _iteration_step = 1;
    }

    RBM::RBM(cv::Mat w, cv::Mat vb, cv::Mat hb, float(*sampleFun)(float))
    {
        CV_Assert(w.rows == hb.rows);
        CV_Assert(w.cols == vb.rows);
        CV_Assert(vb.cols == 1 && hb.cols == 1);
        init_rng_and_sampler(sampleFun);
        weights = w.clone();
        vis_bias = vb.clone();
        hid_bias = hb.clone();
        _trained = true;
        _current_iteration = 0;
        _step_type = EPOCHS;
        _iteration_step = 1;
    }

    RBM::~RBM() { }



    void RBM::save(std::string filepath)
    {
        cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
        fs << "trained" << _trained;
        _train_params.write(fs);
        fs << "weights" << weights;
        fs << "hid_bias" << hid_bias;
        fs << "vis_bias" << vis_bias;
        fs.release();
    }

    void RBM::write(cv::FileStorage &fs)
    {
        fs << "{:";
        fs << "trained" << _trained;
        _train_params.write(fs);
        fs << "weights" << weights;
        fs << "hid_bias" << hid_bias;
        fs << "vis_bias" << vis_bias;
        fs << "}";
    }

    RBM& RBM::load(std::string filepath)
    {
        cv::FileStorage fs(filepath, cv::FileStorage::READ);
        CV_Assert(fs.isOpened());
        fs["trained"] >> _trained;
        _train_params.read(fs);
        fs["weights"] >> weights;
        fs["hid_bias"] >> hid_bias;
        fs["vis_bias"] >> vis_bias;
        fs.release();
        return *this;
    }
    RBM& RBM::read(cv::FileNode &fn)
    {
        fn["trained"] >> _trained;
        _train_params.read(fn);
        fn["weights"] >> weights;
        fn["hid_bias"] >> hid_bias;
        fn["vis_bias"] >> vis_bias;
        return *this;
    }


    int RBM::num_hidden() { return weights.rows; }

    int RBM::num_visible() { return weights.cols; }

    bool RBM::is_trained() { return _trained; }

    RBM::TrainParams RBM::get_params() { return _train_params; }

    RBM& RBM::set_train_params(RBM::TrainParams params)
    {
        _train_params = params;
        return *this;
    }

    RBM& RBM::set_seed(int seed)
    {
        rng = cv::RNG(seed);
        return *this;
    }

    /* Data is stored transposed for convenience */
    RBM& RBM::set_datasets(cv::Mat train, cv::Mat validation)
    {
        CV_Assert(train.data);
        CV_Assert(train.cols == num_visible());
        CV_Assert(!validation.data || validation.cols == num_visible());
        CV_Assert(train.type() == CV_32F && (!validation.data || validation.type() == CV_32F));

        _trainset = train.t();
        _valset = validation.data? validation.t() : validation;
        return *this;
    }

    cv::Mat RBM::trainset() { return _trainset.data? _trainset.t() : _trainset; }
    cv::Mat RBM::valset() { return _valset.data? _valset.t() : _valset; }

    int RBM::current_iteration() { return _current_iteration; }

    RBM& RBM::set_step_type(int t)
    {
        _step_type = t;
        return *this;
    }

    RBM& RBM::set_iteration_step(int step) 
    {
        _iteration_step = step > 0? step : 1;
        return *this;
    }

    /* trainPatterns has size <number of visible units> x <data cases>
    */
    void extractBatch(const cv::Mat& trainPatterns, cv::Mat& batch, int& start_batch, const int batch_size)
    {
        int total_number_pattern = trainPatterns.cols;
        int end_batch = start_batch + batch_size;

        if(end_batch > total_number_pattern)
        {
            end_batch = total_number_pattern;

            batch = cv::Mat(trainPatterns, cv::Range::all(), cv::Range(start_batch, end_batch));
            start_batch = (start_batch + batch_size) % total_number_pattern;
            cv::Mat batch2 = cv::Mat(trainPatterns, cv::Range::all(), cv::Range(0, start_batch)).t();

            batch = batch.t();
            batch.push_back(batch2);
            batch = batch.t();
        }
        else
        {
            batch = cv::Mat(trainPatterns, cv::Range::all(), cv::Range(start_batch, end_batch));
            start_batch = (start_batch + batch_size) % total_number_pattern;
        }
    }


    /* "visible_state": binary matrix of size <number of visible units> x <number of data cases>.
     * "hidden_state": eventually binary matrix of size <number of hidden units> by <number of data cases>.
     * OUT "gradient": the gradient of the mean configuration goodness (negative energy) with respect to the model parameters.
     *                 It's <Si*Sj> over all data cases. It's the mean over them, not the sum.
     *                 Size is <number of hidden units> x <number of visible units>, same shape of the model's parameters
    */
    void RBM::goodness_gradient(const cv::Mat& visible_state, const cv::Mat& hidden_state, Gradient& gradient)
    {
        const int batch_size = visible_state.cols;

        gradient.w = (hidden_state * visible_state.t()) / batch_size;    

        cv::Mat bias_neurons_activations = cv::Mat::ones(batch_size, 1, CV_32F);
        gradient.h_bias = (hidden_state * bias_neurons_activations) / batch_size;
        gradient.v_bias = (visible_state * bias_neurons_activations) / batch_size;
    }


    bool RBM::first_iteration() { return _current_iteration == 0; }
    bool RBM::last_iteration() { return _current_iteration == _train_params.iterations - 1; }


    void RBM::init_structs_for_train()
    {
        rng.fill(weights, cv::RNG::NORMAL, 0, 0.01);
        hid_bias = hid_bias * 0.0f;
        vis_bias = vis_bias * 0.0f;

        _momentum_speed_w = cv::Mat::zeros(weights.rows, weights.cols, CV_32F);
        _momentum_speed_vb = cv::Mat::zeros(vis_bias.rows, vis_bias.cols, CV_32F);
        _momentum_speed_hb = cv::Mat::zeros(hid_bias.rows, hid_bias.cols, CV_32F);

        _start_next_minibatch = 0;
        _current_iteration = 0;
    }

    /* "probabilities": matrix of probabilities
     * OUT "binary_sample": matrix of sampled values in {0, 1} 
     */
    void RBM::sample_binary(const cv::Mat& probabilities, cv::Mat& binary_sample)
    {
        probabilities.copyTo(binary_sample);
        matfunc<float>(binary_sample, _sampler, binary_sample);
    }

    inline
    float sign_element(float v)
    {
        if(v > 0) return 1;
        if(v < 0) return -1;
        else      return 0;
    }

    /* Typical values of L2 delta ranges from 0.00001 to 0.01
     * Start with 0.0001 */
    void RBM::weight_decay(int type, float delta, cv::Mat& wd) 
    {
        switch(type)
        {
            case TrainParams::L1_WEIGHT_DECAY:
            {
                weights.copyTo(wd);
                matfunc<float>(wd, sign_element, wd);
                wd = wd * delta;
                break;
            }
            case TrainParams::L2_WEIGHT_DECAY:
            {
                wd = weights * delta;
                break;
            }
            default:
            {
                wd.create(weights.size(), CV_32F);
                wd.setTo(cv::Scalar(0));
            }
        }

    }



    bool RBM::step()
    {
        if(first_iteration() && !is_trained())
        {
            init_structs_for_train();
            _trained = true;
        }

        int num_iter = _step_type == EPOCHS? _trainset.cols / _train_params.batch_size : 1;
        num_iter = num_iter < 1? 1 : num_iter;

        _weight_updates = weights * 0;

        for(int i=0; i < num_iter; i++)
        {
            cv::Mat batch; // <number of visible units> x <num patterns>
            extractBatch(_trainset, batch, _start_next_minibatch, _train_params.batch_size);

            Gradient gradient;
            cd(_train_params.cdk, batch, gradient);

            cv::Mat wd;
            weight_decay(_train_params.weight_decay, _train_params.wd_delta, wd);

            _momentum_speed_w = (_momentum_speed_w * _train_params.momentum) 
                                + _train_params.learning_rate * gradient.w 
                                - _train_params.learning_rate * wd;
            _momentum_speed_vb = (_momentum_speed_vb * _train_params.momentum)
                                + _train_params.learning_rate * gradient.v_bias;
            _momentum_speed_hb = (_momentum_speed_hb * _train_params.momentum)
                                + _train_params.learning_rate * gradient.h_bias;

            weights = weights + _momentum_speed_w;
            vis_bias = vis_bias + _momentum_speed_vb;
            hid_bias = hid_bias + _momentum_speed_hb;

            _weight_updates += _momentum_speed_w;

            if(last_iteration())
            {
                _current_iteration = 0;
                return false;
            }

            _current_iteration++;
        }

        return true;
    }

    void RBM::train()
    {
        CV_Assert(_trainset.data);

        while(_train_params.iterations > 0 && step())
        { }

    }


    cv::Mat RBM::weights_update()
    {
        return _weight_updates;
    }




    const RBM::Gradient RBM::Gradient::operator -(const Gradient& other)
    {
        Gradient ret;
        ret.w = this->w - other.w;
        ret.h_bias = this->h_bias - other.h_bias;
        ret.v_bias = this->v_bias - other.v_bias;
        return ret;
    }

    void RBM::TrainParams::write(cv::FileStorage &fs)
    {
        fs << "batch" << this->batch_size;
        fs << "cdk" << this->cdk;
        fs << "iter" << this->iterations;
        fs << "lr" << this->learning_rate;
        fs << "momentum" << this->momentum;
        fs << "wd_delta" << this->wd_delta;
        fs << "wd_type" << this->weight_decay;
    }

    void RBM::TrainParams::read(cv::FileStorage &fs)
    {
        fs["batch"] >> this->batch_size;
        fs["cdk"] >> this->cdk;
        fs["iter"] >> this->iterations;
        fs["lr"] >> this->learning_rate;
        fs["momentum"] >> this->momentum;
        fs["wd_delta"] >> this->wd_delta;
        fs["wd_type"] >> this->weight_decay;
    }
    void RBM::TrainParams::read(cv::FileNode &fn)
    {
        fn["batch"] >> this->batch_size;
        fn["cdk"] >> this->cdk;
        fn["iter"] >> this->iterations;
        fn["lr"] >> this->learning_rate;
        fn["momentum"] >> this->momentum;
        fn["wd_delta"] >> this->wd_delta;
        fn["wd_type"] >> this->weight_decay;
    }

}