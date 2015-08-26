
#include <opencv2/opencv.hpp>
#include "../../landmarks.h"
#include "../../texturefeature.h"


#include "RBMcpp/rbmUtils.h"
#include "RBMcpp/RBMglu.h"

//using namespace artelab;
using std::string;
using std::cout;
using std::endl;
using std::vector;

const int PSIZE = 16; // @#* global..

struct RBMExtractor : public TextureFeature::Extractor
{
    artelab::RBMglu rbm[20]; // one rbm per landmark
    cv::Ptr<Landmarks> land;

    RBMExtractor(const cv::String & xmlpath)
    {
        cv::FileStorage fs(xmlpath, cv::FileStorage::READ);
        CV_Assert(fs.isOpened());
        cv::FileNode pnodes = fs["RBMS"];
        int i=0;
        for (cv::FileNodeIterator it=pnodes.begin(); it!=pnodes.end(); ++it)
        {
            rbm[i++].read(*it);
        }
        fs.release();

        land = createLandmarks();
    }

    virtual int extract(const cv::Mat &img, cv::Mat &features) const
    {
        std::vector<cv::Point> kp;
        land->extract(img,kp);
        for (size_t i=0; i<kp.size(); i++)
        {
            cv::Mat patch;
            cv::getRectSubPix(img, cv::Size(PSIZE,PSIZE), kp[i], patch);

            cv::Mat im;
            cv::normalize(patch, im, 1.0, 0, cv::NORM_L2, CV_32F);
            //patch.convertTo(im, CV_32F, 5.0f/255);

            cv::Mat feat;
            artelab::feature_patterns((artelab::RBM*)(&(rbm[i])), im.reshape(1,1), feat, true);
            features.push_back(feat);
        }
        features = features.reshape(1,1);
        return features.total() * features.elemSize();
    }
};

//
// entry point(factory):
//
cv::Ptr<TextureFeature::Extractor> createRBMExtractor(const cv::String & xmlpath) 
{ 
    return cv::makePtr<RBMExtractor>(xmlpath);
}


//
// ---8<------------ cutoff for library usage only ------------------------------------
//
#ifdef TRAIN_RBM_STANDALONE

int main(int argc, char** argv) 
{
    int start=0; // you may have to break & continue later
    if (argc>1) start = atoi(argv[1]);

    cv::String path("e:/code/opencv_p/face3/data/lfw3d_9000/*.jpg");
    vector<cv::String> fn;
    cv::glob(path,fn,true);
    cv::Ptr<Landmarks> land = createLandmarks();
    for (int k=start; k<20; ++k)
    {
        cv::Mat train;
        std::cerr << endl << "keypoint " << k << endl;
        for (size_t i=0; i<4000; ++i)
        {
            //int id = cv::theRNG().uniform(0, fn.size()); // randomize or not ?
            cv::Mat im = cv::imread(fn[i], 0);

            std::vector<cv::Point> kp;
            land->extract(im,kp);
            cv::Mat patch;
            cv::getRectSubPix(im, cv::Size(PSIZE,PSIZE), kp[k], patch);

            cv::Mat  m;
            cv::normalize(patch, m, 1.0, 0, cv::NORM_L2, CV_32F);
            //patch.convertTo(m, CV_32F, 5.0f/255);
            train.push_back(m.reshape(1,1));
        }
        cout << "Train data: " << train.rows << "x" << train.cols << endl;

        // Train RBM
        cout << "Training RBM" << endl;
        const int epochs = 10;
        artelab::RBMglu::TrainParams params;
        params.learning_rate = 0.05f;
        params.batch_size = 10;
        params.momentum = 0.5f;
        params.iterations = train.rows / params.batch_size * epochs;
        params.weight_decay = artelab::RBMglu::TrainParams::L2_WEIGHT_DECAY;
        params.wd_delta = 0.0002f;
        
        const int num_hid = (15 * train.cols) / 10;
        artelab::RBMglu rbm = artelab::RBMglu(train.cols, num_hid);
        rbm.set_seed(345)
           .set_datasets(train)
           .set_step_type(artelab::RBM::EPOCHS)
           .set_train_params(params);
        
        cout << rbm.description() << endl << endl;
        
        int epoch = 0;
        while(++epoch && rbm.step())
        {
            artelab::show_bases(&rbm, cv::Size(PSIZE,PSIZE));
            cv::waitKey(10);

            float mset = artelab::average_mse(&rbm, train);
            float ft = rbm.avg_free_energy(train);
            std::cerr << cv::format("%4d %7.6f %7.4f : %4.4f %4.4f",epoch,mset,ft,params.learning_rate,params.momentum) << std::endl;

            if ((mset < 0.0001f) || (epoch >= epochs))
                break;

            if (epoch==5)
            {
                params.learning_rate = 0.01f;
                params.momentum = 0.9f;
                rbm.set_train_params(params);
            }
            //float decay = 0.95f;
            //params.learning_rate *= decay;
            //params.momentum *= (1.0f/decay);
            //rbm.set_train_params(params);
        }
        artelab::show_bases(&rbm, cv::Size(PSIZE,PSIZE));
        rbm.save(cv::format("rbm_%d.xml", k));
        cout << "RBM " << k << " succesfully trained in " << epoch << " epochs." << endl;
    }
    // concateneate the 20 xml's to a zipped one:
    cv::FileStorage fw("rbm.xml.gz", cv::FileStorage::WRITE);
    fw << "RBMS" << "[";
    for (int i=0; i<20; i++)
    {
        artelab::RBMglu rbm(cv::format("rbm_%d.xml",i), 0);
        rbm.write(fw);
    }
    fw << "]";
    fw.release();

    // cv::waitKey();
    cout << endl;
    return 1;
}


#endif TRAIN_RBM_STANDALONE
