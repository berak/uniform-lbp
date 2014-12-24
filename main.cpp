//
// run k fold crossvalidation train/test on  person db
//

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
using namespace cv;


#include <iostream>
#include <fstream>
#include <map>
using namespace std;


#include "TextureFeature.h"
#include "Preprocessor.h"


using TextureFeature::Extractor;
using TextureFeature::Reductor;
using TextureFeature::Classifier;

bool debug = false;
RNG rng(getTickCount());

double ct(int64 t)
{
    return double(t) / cv::getTickFrequency();
}


//
// read a 'path <blank> label' list
//
int readtxt(const char *fname, std::vector<std::string> &names, std::vector<int> &labels, size_t maxim)
{
    int maxid=-1;
    std::ifstream in(fname);
    while(in.good() && !in.eof())
    {
        std::string path;
        in >> path;
        names.push_back(path);

        int label;
        in >> label;
        labels.push_back(label);

        maxid=std::max(maxid,label);
        if (labels.size() >= maxim)
            break;
    }
    return maxid;
}


//
// imglists per person.
//  no really, you can't just draw a random probability set from a set of multiple classes and call it a day ...
//
void setupPersons(const vector<int> &labels, vector<vector<int>> &persons)
{
    // find out which index belongs to which person
    //
    persons.resize(1);
    int previd=0;
    for (size_t j=0; j<labels.size(); j++)
    {
        int id = labels[j];
        if (previd!=id)
        {
            persons.push_back(vector<int>());
            previd=id;
        }
        persons.back().push_back(j);
    }
}

int extractDB(const string &txtfile, vector<Mat> &images, Mat &labels, int preproc, int precrop, int maxim, int fixed_size)
{
    // read face db
    vector<string> vec;
    vector<int> vlabels;
    int nsubjects = 1 + readtxt(txtfile.c_str(), vec, vlabels, maxim);

    Preprocessor pre(preproc,precrop,fixed_size);

    //
    // read the images,
    //   correct labels if empty images are skipped
    //   also apply preprocessing,
    //
    int load_flag = preproc==-1 ? 1 : 0;
    for (size_t i=0; i<vec.size(); i++)
    {
        Mat m1 = imread(vec[i], load_flag);
        if (m1.empty())
            continue;

        Mat m2;
        resize(m1, m2, Size(fixed_size, fixed_size));

        Mat m3 = pre.process(m2);

        images.push_back(m3);
        labels.push_back(vlabels[i]);
        //if ( i%33==0) imshow("i",mm), waitKey(0);
    }
    return nsubjects;
}

int crossfoldData(Ptr<Extractor> ext,
                  Ptr<Reductor> red,
                  Mat & trainFeatures,
                  Mat & trainLabels,
                  Mat & testFeatures,
                  Mat & testLabels,
                  const vector<Mat> &images,
                  const vector<int> &labels,
                  const vector<vector<int>> &persons,
                  size_t f, size_t fold)
{
    int fsiz=0;

    // split train/test set per person:
    for (size_t j=0; j<persons.size(); j++)
    {
        size_t n_per_person = persons[j].size();
        if (n_per_person < fold)
            continue;
        int r = (fold != 0) ? (n_per_person/fold) : -1;
        for (size_t n=0; n<n_per_person; n++)
        {
            int index = persons[j][n];

            Mat feature;
            ext->extract(images[index],feature);

            if (!red.empty())
            {
                red->reduce(feature, feature);
            }

            fsiz = feature.total() * feature.elemSize();

            // sliding window per fold
            if ((fold>1) && (n >= f*r) && (n <= (f+1)*r))
            {
                testFeatures.push_back(feature);
                testLabels.push_back(labels[index]);
            }
            else
            {
                trainFeatures.push_back(feature);
                trainLabels.push_back(labels[index]);
            }
        }
    }
    return fsiz;
}


double runtest(string name, Ptr<Extractor> ext, Ptr<Reductor> red, Ptr<Classifier> cls, const vector<Mat> &images, const vector<int> &labels, const vector<vector<int>> &persons, size_t fold=10)
{
    //
    // for each fold, take alternating n/fold items for test, the others for training
    //
    // each test is confused on its own over a lot of folds..
    Mat confusion = Mat::zeros(persons.size(),persons.size(),CV_32F);

    int64 t0=getTickCount();
    int fsiz=0;
    for (size_t f=0; f<fold; f++)
    {
        int64 t1 = cv::getTickCount();
        Mat trainFeatures, trainLabels;
        Mat testFeatures,  testLabels;

        fsiz = crossfoldData(ext,red,trainFeatures,trainLabels,testFeatures,testLabels,images,labels,persons,f,fold);
        trainFeatures = trainFeatures.reshape(1,trainLabels.rows);

        cls->train(trainFeatures,trainLabels);

        Mat conf = Mat::zeros(confusion.size(), CV_32F);
        for (int i=0; i<testFeatures.rows; i++)
        {
            Mat res;
            Mat feat = testFeatures.row(i);
            cls->predict(feat.reshape(1,1), res);

            int pred = int(res.at<float>(0));
            int ground = testLabels.at<int>(i);
            if (pred<0 || ground<0)
            {
                cerr << "neg prediction " << f << " " << i << " " << pred << " " << ground << endl;
                continue;
            }
            conf.at<float>(ground, pred) ++;
        }
        confusion += conf;

        double all = sum(confusion)[0];
        double neg = all - sum(confusion.diag())[0];
        double err = double(neg)/all;
        cout << format("%-13s %-2d %6d %6d %6d %8.3f",name.c_str(),(f+1), fsiz, int(all-neg), int(neg), (1.0-err)) << '\r';
    }


    // evaluate. this is probably all too simple.
    double all = sum(confusion)[0];
    double neg = all - sum(confusion.diag())[0];
    double err = double(neg)/all;
    int64 t1=getTickCount() - t0;
    double t(t1/getTickFrequency());
    cout << format("%-16s %6d %6d %6d %8.3f %8.3f",name.c_str(), fsiz, int(all-neg), int(neg), (1.0-err), t) << endl;
    if (debug) cout << "confusion" << endl << confusion(Range(0,min(20,confusion.rows)), Range(0,min(20,confusion.cols))) << endl;
    return err;
}


//
//
// face att.txt 5     5             1        0
// face db      fold  reco    preprocessing  debug
//
// special: reco==0 will run *all* recognizers available on a given db
//
int main(int argc, const char *argv[])
{
    vector<Mat> images;
    Mat labels;

    std::string db_path("senthil.txt");
    //std::string db_path("att.txt");
    //std::string db_path("yale.txt");
    if (argc>1) db_path = argv[1];

    size_t fold = 4;
    if (argc>2) fold = atoi(argv[2]);

    int rec = 50;
    if (argc>3) rec = atoi(argv[3]);

    int preproc = 0; 
    if (argc>4) preproc = atoi(argv[4]);

    if (argc>5) debug = atoi(argv[5])!=0;


    extractDB(db_path, images, labels, preproc, 0, 500, 90);

    // per person id lookup
    vector<vector<int>> persons;
    setupPersons( labels, persons );
    fold = std::min(fold,images.size()/persons.size());

    // some diagnostics:
    String dbs = db_path.substr(0,db_path.find_last_of('.')) + ":";
    char *pp[] = { "no preproc","eqhist","clahe","retina","tan-triggs","crop",0 };
    if (rec == 0)
        cout << "--------------------------------------------------------------" << endl;
    cout << format("%-19s",dbs.c_str()) << fold  << " fold, " << persons.size()  << " classes, " << images.size() << " images, " << pp[preproc] << endl;
    if ( rec==0 )
    {
        cout << "--------------------------------------------------------------" << endl;
        cout << "[method]       [f_bytes]  [hit]  [miss]  [acc]   [time]  " << endl;
    }

    // loop through all tests for rec==0, do one test else.
    int n=64; // it's gettin a *bit* crowded ;)
    if (rec > 0)
    {
        n = rec+1;
    }
    for (; rec<n; rec++)
    {
        switch(rec)
        {
        default: continue;
        //case 1:  runtest("pixels_L2",    createExtractorPixels(120,120),  Ptr<TextureFeature::Reductor>(),   createClassifierNearest(),               images,labels,persons, fold); break;
        case 2:  runtest("pixels_svm",   createExtractorPixels(60,60),    Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 3:  runtest("pixels_cosine",createExtractorPixels(120,120),  Ptr<TextureFeature::Reductor>(),   createClassifierCosine(),                images,labels,persons, fold); break;
        case 5:  runtest("lbp_L2",       createExtractorLbp(),            Ptr<TextureFeature::Reductor>(),   createClassifierNearest(),               images,labels,persons, fold); break;
        case 6:  runtest("lbp_svm",      createExtractorLbp(),            Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 7:  runtest("lbp_hell",     createExtractorLbp(),            Ptr<TextureFeature::Reductor>(),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        //case 8:  runtest("lbp_o_svm",    createExtractorOverlapLbp(),     Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 9:  runtest("lbp_e_svm",    createExtractorElasticLbp(),     Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 10: runtest("lbp_p_whd_svm",    createExtractorPyramidLbp(), createReductorWalshHadamard(12000),    createClassifierSVM(),                   images,labels,persons, fold); break;
        case 11: runtest("fplbp_svm",    createExtractorFPLbp(),          Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 12: runtest("fplbp_hell",   createExtractorFPLbp(),          Ptr<TextureFeature::Reductor>(),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        //case 13: runtest("fplbp_o_svm",  createExtractorOverlapFpLbp(),   Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 14: runtest("fplbp_e_svm",  createExtractorElasticFpLbp(),   Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 15: runtest("fplbp_p_svm",  createExtractorPyramidFpLbp(),   Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 16: runtest("tplbp_hell",   createExtractorTPLbp(),          Ptr<TextureFeature::Reductor>(),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 17: runtest("tplbp_svm",    createExtractorTPLbp(),          Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 18: runtest("tplbp_o_svm",  createExtractorOverlapTpLbp(),   Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 19: runtest("tplbp_e_svm",  createExtractorElasticTpLbp(),   Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 20: runtest("tplbp_p_svm",  createExtractorPyramidTpLbp(),   Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 21: runtest("mts_svm",      createExtractorMTS(),            Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 22: runtest("mts_hell",     createExtractorMTS(),            Ptr<TextureFeature::Reductor>(),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        //case 24: runtest("mts_e_hell",   createExtractorElasticMTS(),     Ptr<TextureFeature::Reductor>(),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        //case 25: runtest("mts_e_svm",    createExtractorElasticMTS(),     Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 26: runtest("mts_o_svm",    createExtractorOverlapMTS(),     Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 27: runtest("mts_p_svm",    createExtractorPyramidMTS(),     Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 28: runtest("bgc1_hell",    createExtractorBGC1(),           Ptr<TextureFeature::Reductor>(),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 29: runtest("bgc1_svm",     createExtractorBGC1(),           Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                    images,labels,persons, fold); break;
        //case 30: runtest("bgc1_e_svm",   createExtractorElasticBGC1(),    Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 31: runtest("bgc1_o_svm",   createExtractorOverlapBGC1(),    Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 32: runtest("bgc1_p_svm",   createExtractorPyramidBGC1(),    createReductorDct(8000),           createClassifierSVM(),                   images,labels,persons, fold); break;
        case 33: runtest("comb_hell",    createExtractorCombined(),       Ptr<TextureFeature::Reductor>(),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        case 34: runtest("comb_svm",     createExtractorCombined(),       Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 35: runtest("comb_e_svm",   createExtractorElasticCombined(),Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 36: runtest("comb_o_svm",   createExtractorOverlapCombined(),Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 37: runtest("comb_p_svm",   createExtractorPyramidCombined(),Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 38: runtest("gabor_svm",    createExtractorGaborLbp(),       createReductorDct(8000),           createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 39: runtest("dct_cosine",   createExtractorDct(),            Ptr<TextureFeature::Reductor>(),   createClassifierCosine(),                images,labels,persons, fold); break;
        case 40: runtest("dct_svm",      createExtractorDct(),            Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 41: runtest("orb_ham2",     createExtractorORBGrid(),        Ptr<TextureFeature::Reductor>(),   createClassifierNearest(NORM_HAMMING2),  images,labels,persons, fold); break;
        //case 42: runtest("orb_L1",       createExtractorORBGrid(),        Ptr<TextureFeature::Reductor>(),   createClassifierNearest(NORM_L1),        images,labels,persons, fold); break;
        case 41: runtest("sift_L2",      createExtractorSIFTGrid(),       Ptr<TextureFeature::Reductor>(),   createClassifierNearest(NORM_L2),        images,labels,persons, fold); break;
        case 42: runtest("sift_svm",     createExtractorSIFTGrid(),       Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),        images,labels,persons, fold); break;
        case 44: runtest("sift20_dct_svm",createExtractorSIFTGrid(20),    createReductorDct(8000),           createClassifierSVM(),                   images,labels,persons, fold); break;
        case 45: runtest("sift_gftt_svm",createExtractorSIFTGftt(),       Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 46: runtest("grad_svm",     createExtractorGrad(),           Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 47: runtest("grad_gftt_svm",createExtractorGfttGrad(),       Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 48: runtest("gradmag_svm",  createExtractorGfttGradMag(),    Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 49: runtest("eigen",        createExtractorPixels(),         Ptr<TextureFeature::Reductor>(),   createClassifierPCA(),                   images,labels,persons, fold); break;
        case 50: runtest("fisher",       createExtractorPixels(),         Ptr<TextureFeature::Reductor>(),   createClassifierPCA_LDA(),               images,labels,persons, fold); break;
        //case 51: runtest("orb_had_svm",  createExtractorORBGrid(),        createReductorWalshHadamard(),     createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 52: runtest("orb_hell_svm", createExtractorORBGrid(),        createReductorHellinger(),         createClassifierSVM(),                   images,labels,persons, fold); break;
        case 55: runtest("hdlbp_svm",    createExtractorHighDimLbp(),     Ptr<TextureFeature::Reductor>(),   createClassifierSVM(),                   images,labels,persons, fold); break;
        case 56: runtest("hdlbp_whad_svm",createExtractorHighDimLbp(),     createReductorWalshHadamard(12000),          createClassifierSVM(),                   images,labels,persons, fold); break;
        case 57: runtest("hdlbp_hel_svm",createExtractorHighDimLbp(),     createReductorHellinger(),         createClassifierSVM(),                   images,labels,persons, fold); break;
        //case 52: runtest("hdlbp_svm",    createExtractorHighDimLbp(),     Ptr<TextureFeature::Reductor>(),   createClassifierHist(HISTCMP_HELLINGER), images,labels,persons, fold); break;
        }                                                          
    }
    return 0;
}



