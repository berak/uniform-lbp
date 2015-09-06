//
// (ab)using: https://github.com/SimoneAlbertini/RBMcpp
//
#include <opencv2/opencv.hpp>
//#include "../../landmarks.h"
#include "../../texturefeature.h"
#include "../../profile.h"

#include "RBMcpp/rbmUtils.h"
#include "RBMcpp/RBMglu.h"

//using namespace artelab;
using std::string;
using std::cout;
using std::endl;
using std::vector;

int PSIZE = 26;

cv::Mat processRect(const cv::Mat &im, cv::Point2f c)
{
    PROFILE;
    double fac = 2.0;
    cv::Size siz(int(PSIZE*fac),int(PSIZE*fac));
    cv::Mat patch;
    cv::getRectSubPix(im, siz, c, patch);

    cv::resize(patch,patch,cv::Size(),1.0/fac,1.0/fac);

    cv::Scalar me,sd;
    cv::meanStdDev(patch, me, sd);
    patch -= me[0];
    patch /= sd[0];
    cv::Mat  m;
    cv::normalize(patch, m, 1.0, 0, cv::NORM_L2, CV_32F);
    return m;
}


// big static patches for now.
struct Facemarks
{
    enum { SIZE=5 };

    vector<cv::Point> p;

    Facemarks()
    {
        p.push_back(cv::Point(25,25));
        p.push_back(cv::Point(85,25));
        p.push_back(cv::Point(55,55));
        p.push_back(cv::Point(35,75));
        p.push_back(cv::Point(75,75));
    }
    int extract(const cv::Mat &img, std::vector<cv::Point> &kp) const
    {
        kp=p;
        return SIZE;
    }
};
cv::Ptr<Facemarks> _createLandmarks() { return cv::makePtr<Facemarks>(); }

struct RBMExtractor : public TextureFeature::Extractor
{
    artelab::RBMglu rbm[Facemarks::SIZE]; // one rbm per landmark
    cv::Ptr<Facemarks> land;

    RBMExtractor(const cv::String & xmlpath)
    {
        PROFILE;
        cv::FileStorage fs(xmlpath, cv::FileStorage::READ);
        CV_Assert(fs.isOpened());

        int ps=-1;
        fs["PSIZE"] >> ps;
        if (ps>0) PSIZE = ps;;

        cv::FileNode pnodes = fs["RBMS"];
        int i=0;
        for (cv::FileNodeIterator it=pnodes.begin(); it!=pnodes.end(); ++it)
        {
            rbm[i++].read(*it);
        }
        fs.release();

        land = _createLandmarks();
    }

    virtual int extract(const cv::Mat &img, cv::Mat &features) const
    {
        PROFILE;
        std::vector<cv::Point> kp;
        land->extract(img,kp);
        for (size_t i=0; i<kp.size(); i++)
        {
            cv::Mat im = processRect(img, kp[i]);

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

int concatenate()
{
    // concateneate the 20 xml's to a single zipped one:
    cv::FileStorage fw("rbm.xml.gz", cv::FileStorage::WRITE);
    fw << "PSIZE" << PSIZE;
    fw << "RBMS" << "[";
    for (int i=0; i<Facemarks::SIZE; i++)
    {
        artelab::RBMglu rbm(cv::format("rbm_%d.xml",i), 0);
        rbm.write(fw);
    }
    fw << "]";
    fw.release();

    return 1;
}

int main(int argc, char** argv) 
{
    int argn=1;
    int start = 0; // you may have to break & continue later
    if (argc>1) start = atoi(argv[1]);
    if (start>=20) return concatenate();

    int epochs = 32;
    if (argc>2) epochs = atoi(argv[2]);

    double learn = 0.05;
    if (argc>3) learn = atof(argv[3]);

    double eta = 0.0001;
    if (argc>4) eta = atof(argv[4]);

    if (argc>5) PSIZE = atoi(argv[5]);

    cv::Ptr<Facemarks> land = _createLandmarks();

    cv::String path("e:/code/opencv_p/face3/data/lfw3d_9000/*.jpg");
    vector<cv::String> fn;
    cv::glob(path,fn,true);
    std::cerr << fn.size() << " files, " << epochs << " epochs, " << eta << " eta, " << PSIZE << " patchsize." << std::endl;
    cv::namedWindow("mean",0);
    for (int k=start; k<Facemarks::SIZE; ++k)
    {
        PROFILEX("per_landmark");
        cv::Mat means(PSIZE,PSIZE,CV_32F,0.0f);
        cv::Mat train;
        std::cerr << endl << "keypoint " << k << endl;
        for (size_t i=0; i<fn.size(); i+=3)
        {
            PROFILEX("per_image");
            //int id = cv::theRNG().uniform(0, fn.size()); // randomize or not ?
            cv::Mat img = cv::imread(fn[i], 0);

            std::vector<cv::Point> kp;
            land->extract(img,kp);

            cv::Mat m = processRect(img, kp[k]);
            train.push_back(m.reshape(1,1));

            cv::accumulate(m, means);
        }
        cv::normalize(means,means,1,0,cv::NORM_MINMAX);
        cv::imshow("mean",means);
        cv::waitKey(100);
        cout << "Train data: " << train.rows << "x" << train.cols << endl;
       
        // Train RBM
        artelab::RBMglu::TrainParams params;
        params.learning_rate = learn;
        params.batch_size = 10;
        params.momentum = 0.5f;
        params.iterations = train.rows / params.batch_size * epochs;
        params.weight_decay = artelab::RBMglu::TrainParams::L2_WEIGHT_DECAY;
        params.wd_delta = 0.002f;
        
        const int num_hid = (8 * train.cols) / 10;
        artelab::RBMglu rbm = artelab::RBMglu(train.cols, num_hid);
        rbm.set_seed(345)
           .set_datasets(train)
           .set_step_type(artelab::RBM::EPOCHS)
           .set_train_params(params);
        
        cout << rbm.description() << endl << endl;
        
        int epoch = 0;
        float mset=0,ft=0;
        int64 t0=cv::getTickCount(),t1=t0;
        while(++epoch && rbm.step())
        {
            PROFILEX("per_epoch");

            t1 = cv::getTickCount();
            double dt = (t1 - t0)/cv::getTickFrequency();
            t0 = t1;

            artelab::show_bases(&rbm, cv::Size(PSIZE,PSIZE));
            cv::waitKey(10);

            mset = artelab::average_mse(&rbm, train);
            ft = rbm.avg_free_energy(train);
            std::cerr << cv::format("%4d %7.6f %7.4f : %4.4f %4.4f : %4.4f sec.",epoch,mset,ft,params.learning_rate,params.momentum,dt) << std::endl;
            if ((mset < eta) || (epoch >= epochs))
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
        cout << "RBM " << k << " succesfully trained with "<<mset<<" err in " << epoch-1 << " epochs." << endl;
    }

    return concatenate();
}


#endif TRAIN_RBM_STANDALONE
