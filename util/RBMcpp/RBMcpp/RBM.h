/* 
 * File:   RBM.h
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#ifndef RBM_H
#define	RBM_H

#include <opencv2/core.hpp>

namespace artelab
{
    class RBM 
    {

    public:

        enum { EPOCHS = 201, BATCH = 202 };

        class Gradient
        {
        public:
            cv::Mat w;
            cv::Mat h_bias;
            cv::Mat v_bias;

            const Gradient operator -(const Gradient& other);
        };

        typedef struct _TrainParams
        {

            enum {
                NO_WEIGHT_DECAY = 100,
                L1_WEIGHT_DECAY = 101,
                L2_WEIGHT_DECAY = 102
            };

            _TrainParams()
            {
                learning_rate = 0.02f;
                momentum = 0.9f;
                iterations = 1000; 
                batch_size = 100;
                cdk = 1;
                weight_decay = NO_WEIGHT_DECAY;
                wd_delta = 0.00001f;
            }

            void write(cv::FileStorage &fs);
            void read(cv::FileStorage &fs);
            void read(cv::FileNode &fn);

            float learning_rate;
            float momentum;
            int iterations; 
            int batch_size; 
            int cdk;
            int weight_decay;
            float wd_delta;
        } TrainParams;

        RBM();
        RBM(const int nVis, const int nHid, float (*sampleFun)(float)=NULL);
        RBM(const std::string filepath, float (*sampleFun)(float)=NULL);
        RBM(cv::Mat w, cv::Mat vb, cv::Mat hb, float (*sampleFun)(float)=NULL);
        virtual ~RBM();


        void save(std::string filepath);
        void write(cv::FileStorage &fs);
        RBM& load(std::string filepath);
        RBM& read(cv::FileNode &fn);

        virtual std::string description() = 0;

        int num_hidden();
        int num_visible();
        bool is_trained();

        RBM& set_train_params(TrainParams params);
        TrainParams get_params();

        RBM& set_datasets(cv::Mat train, cv::Mat validation=cv::Mat());
        cv::Mat trainset();
        cv::Mat valset();

        RBM& set_seed(int seed);

        RBM& set_step_type(int t = EPOCHS);
        RBM& set_iteration_step(int step);
        int current_iteration();

        void train();
        bool step();

        virtual void hidden_activations_for(const cv::Mat& data, cv::Mat& activations, bool probabilities=true, int mc_steps=1) = 0;
        virtual void reconstruct(const cv::Mat& data, cv::Mat& reconstruction, int mc_steps=1) = 0;
        virtual float avg_free_energy(const cv::Mat& patterns) = 0;
        cv::Mat weights_update();

        cv::Mat weights;
        cv::Mat vis_bias;
        cv::Mat hid_bias;


    protected:

        float (*_sampler)(float);
        bool _trained;
        TrainParams _train_params;
        int _seed;

        cv::Mat _trainset;
        cv::Mat _valset;

        // training info
        cv::Mat _momentum_speed_w;
        cv::Mat _momentum_speed_vb;
        cv::Mat _momentum_speed_hb;
        int _start_next_minibatch;
        int _current_iteration;
        int _step_type;
        int _iteration_step;

        cv::Mat _weight_updates;

        void sample_binary(const cv::Mat& probabilities, cv::Mat& binary_sample);
        void goodness_gradient(const cv::Mat& visible_state, const cv::Mat& hidden_state, Gradient& gradient);
        virtual void cd(const int k, const cv::Mat& data_cases, Gradient& gradient) = 0;

        void init_rng_and_sampler(float (*sampleFun)(float));
        void init_structs_for_train();
        bool first_iteration();
        bool last_iteration();
        void weight_decay(int type, float delta, cv::Mat& wd);

        template<class C>
        void matfunc(cv::Mat mat, C (*func)(C), cv::Mat& dst)
        {
            mat.copyTo(dst);

            for(register int row=0; row < dst.rows; row++)
            {
                C* values = dst.ptr<C>(row);
                for(register int col=0; col < dst.cols; col++)
                {
                    values[col] = func(values[col]);
                }
            }
        }

    private:

    };
}

#endif	/* RBM_H */

