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


#include "texturefeature.h"
#include "preprocessor.h"


using TextureFeature::Extractor;
using TextureFeature::Filter;
using TextureFeature::Classifier;

bool debug = false;
RNG rng(getTickCount());

double ct(int64 t)
{
    return double(t) / cv::getTickFrequency();
}

#ifdef _WIN32
 const char SEP = '\\';
#else
 const char SEP = '/';
#endif

//
// imgfolder
//  + pers1
//   + img1
//   + img2
//   + img3
//  + pers2
//   + img1
//   + img2
//   + img3
// ...
// 
// you can pass like 'images/*.png', too!
//
int readdir(String dirpath, std::vector<std::string> &names, std::vector<int> &labels, size_t maxim, int minp=10, int maxp=10)
{

    int r0 = dirpath.find_last_of(SEP)+1;

    vector<String> vec;
    glob(dirpath,vec,true);
    if ( vec.empty())
        return 0;
    std::vector<std::string> tnames;
    std::vector<int> tlabels;
    int nimgs=0;
    int label=-1;
    String last_n="";
    for (size_t i=0; i<vec.size(); i++)
    {
        // extract name from filepath:
        String v = vec[i];
        String v1 = v.substr(r0);
        int r1 = v1.find_last_of(SEP);
        String n = v1.substr(0,r1);
        if (n != last_n)
        {
            if (nimgs < minp) // roll back
            {
                tlabels.clear();
                tnames.clear();
                if (label >= 0) label --;
            }
            else
            {
                labels.insert(labels.end(),tlabels.begin(),(maxp==-1) ? tlabels.end() : tlabels.begin()+std::min(maxp,(int)tlabels.size()));
                names.insert(names.end(),tnames.begin(),(maxp==-1) ? tnames.end() : tnames.begin()+std::min(maxp,(int)tlabels.size()));
                tnames.clear();
                tlabels.clear();
            }
            nimgs = 0;
            last_n = n;
            label ++;
            if (labels.size() >= maxim) break;
        }
        tnames.push_back(v);
        tlabels.push_back(label);
        nimgs ++;
    }
    //for ( int i=0; i<labels.size(); i++)
    //    cerr << labels[i] << " " << names[i] << endl;

    return label;
}


//
// read a 'path <blank> label' list (csv without commas or such)
//
int readtxt(const char *fname, std::vector<std::string> &names, std::vector<int> &labels, size_t maxim)
{
    int maxid=-1;
    std::ifstream in(fname);
    CV_Assert(in.good());
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
void setupPersons(const vector<int> &labels, vector< vector<int> > &persons)
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

int extractDB(const string &path, vector<Mat> &images, Mat &labels, int preproc, int precrop, int maxim, int minp, int maxp, int fixed_size)
{
    // read face db
    vector<string> vec;
    vector<int> vlabels;
    int nsubjects =0;
    int nt = path.find(".txt") ;
    if (nt > 0)
        nsubjects = 1 + readtxt(path.c_str(), vec, vlabels, maxim);
    else
        nsubjects = 1 + readdir(path, vec, vlabels, maxim, minp, maxp);

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

        images.push_back(pre.process(m1));
        labels.push_back(vlabels[i]);
    }
    return nsubjects;
}

int crossfoldData(Ptr<Extractor> ext,
                  Ptr<Filter> fil,
                  Mat & trainFeatures,
                  Mat & trainLabels,
                  Mat & testFeatures,
                  Mat & testLabels,
                  const vector< Mat > &images,
                  const vector< int > &labels,
                  const vector< vector<int> > &persons,
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

            if (!fil.empty())
            {
                fil->filter(feature, feature);
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


double runtest(string name, Ptr<Extractor> ext, Ptr<Filter> fil, Ptr<Classifier> cls, const vector<Mat> &images, const vector<int> &labels, const vector< vector<int> > &persons, size_t fold=10)
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

        fsiz = crossfoldData(ext,fil,trainFeatures,trainLabels,testFeatures,testLabels,images,labels,persons,f,fold);
        trainFeatures = trainFeatures.reshape(1, trainLabels.rows);

        cls->train(trainFeatures, trainLabels);

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
        cout << format("%-23s %-2d %6d %6d %6d %8.3f",name.c_str(), (f+1), fsiz, int(all-neg), int(neg), (1.0-err)) << '\r';
    }


    // evaluate. this is probably all too simple.
    double all = sum(confusion)[0];
    double neg = all - sum(confusion.diag())[0];
    double err = double(neg)/all;
    int64 t1=getTickCount() - t0;
    double t(t1/getTickFrequency());
    cout << format("%-28s %6d %6d %6d %8.3f %8.3f",name.c_str(), fsiz, int(all-neg), int(neg), (1.0-err), t) << endl;
    if (debug) cout << "confusion" << endl << confusion(Range(0,min(20,confusion.rows)), Range(0,min(20,confusion.cols))) << endl;
    return err;
}

double runtest(int ext, int fil, int cls, const vector<Mat> &images, const vector<int> &labels, const vector< vector<int> > &persons, size_t fold=10)
{
    string name = format( "%-8s %-6s %-9s", TextureFeature::EXS[ext], TextureFeature::FILS[fil], TextureFeature::CLS[cls]); 
    //try 
    {
        runtest(name,  
            TextureFeature::createExtractor(ext),  
            TextureFeature::createFilter(fil),
            TextureFeature::createClassifier(cls),
            images,labels,persons, fold); 
    } 
    //catch(...)
    //{
    //    cerr << name << " failed!" << endl;
    //}
    return 0;
}


void printOptions()
{
    cerr << "[extractors]  :"<< endl;
    for (size_t i=0; i<TextureFeature::EXT_MAX; ++i) {  if(i%5==0) cerr << endl; cerr << format("%10s(%2d)",TextureFeature::EXS[i],i); }
    cerr << endl << endl << "[filters] :" << endl;
    for (size_t i=0; i<TextureFeature::FIL_MAX; ++i) {  if(i%5==0) cerr << endl; cerr << format("%10s(%2d)",TextureFeature::FILS[i],i); }
    cerr << endl << endl << "[classifiers] :" << endl;
    for (size_t i=0; i<TextureFeature::CL_MAX; ++i)  {  if(i%5==0) cerr << endl; cerr << format("%10s(%2d)",TextureFeature::CLS[i],i);  }
    //cerr << endl << endl <<  "[preproc] :" << endl;
    //for (size_t i=0; i<TextureFeature::PRE_MAX; ++i) {  if(i%5==0) cerr << endl; cerr << format("%10s(%2d)",TextureFeature::PPS[i],i);  }
    cerr << endl;
}



int main(int argc, const char *argv[])
{
    const char *keys =
            "{ help h usage ? |      | show this message }"
            "{ opts o         |      | show extractor / reductor / classifier options }"
            "{ fold F         |10    | folds for crossvalidation }"
            "{ minp m         |10     | mininal img count per person (when reading folders) }"
            "{ maxp M         |10    | maximal img count per person (-1==read_all)}"
            "{ maxim I        |500   | maximal img count overall }"
            "{ ext e          |7     | extractor  enum }"
            "{ fil f          |0     | filter   enum }"
            "{ cls c          |16     | classifier enum }"
            "{ all a          |false | run a hardcoded list of tests }"
            "{ pre P          |0     | preprocessing }"
            "{ crop C         |0     | crop outer pixels }"
            "{ path p         |lfw3d_9000\\*.jpg|\n    path to dataset,\n    txtfile or directory with 1 subdir per person\n   (trailing slash or wildcard)}";
            //"{ path p         |e:\\MEDIA\\faces\\Aberdeen\\*.jpg|\n    path to dataset,\n    txtfile or directory with 1 subdir per person\n   (trailing slash or wildcard)}";
 
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }
    if (parser.has("opts"))
    {
        printOptions();
        return -1;
    }
    int all = parser.has("all");
    int ext = parser.get<int>("ext");
    int fil = parser.get<int>("fil");
    int cls = parser.get<int>("cls");
    int pre = parser.get<int>("pre");
    int crp = parser.get<int>("crop");
    int fold = parser.get<int>("fold");
    int minp = parser.get<int>("minp");
    int maxp = parser.get<int>("maxp");
    int maxim = parser.get<int>("maxim");

    std::string db_path = parser.get<String>("path");

    // load data:
    Mat labels;
    vector<Mat> images;
    extractDB(db_path, images, labels, pre, crp, maxim, minp, maxp, 90);

    // per person id lookup
    vector< vector<int> > persons;
    setupPersons( labels, persons );
    fold = std::min(fold,int(images.size()/persons.size()));

    // some diagnostics:
    String dbs = db_path.substr(0,db_path.find_last_of('.')) + ":";
    const char *pp[] = { "no preproc", "eqhist", "clahe", "retina", "tan-triggs", "logscale",0 };
    if (all)
        cout << "-------------------------------------------------------------------" << endl;
    cout << format("%-24s",dbs.c_str()) << fold  << " fold, " << persons.size() << " classes, " << images.size() << " images, " << pp[pre] << endl;
    if (all)
    {
        cout << "-------------------------------------------------------------------" << endl;
        cout << "[extra] [redu] [class]     [f_bytes]  [hit]  [miss]  [acc]   [time]" << endl;
    }

    if ( ! all )
    {
        runtest(ext, fil, cls, images, labels, persons, fold);
    }
    else
    {
        int tests[] = {
            TextureFeature::EXT_Pixels, TextureFeature::FIL_NONE,  TextureFeature::CL_NORM_L2,
            TextureFeature::EXT_Pixels, TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_POL,
            TextureFeature::EXT_Pixels, TextureFeature::FIL_NONE,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_Dct,    TextureFeature::FIL_NONE,  TextureFeature::CL_COSINE,
            TextureFeature::EXT_Dct,    TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_POL,
            TextureFeature::EXT_Lbp,    TextureFeature::FIL_NONE,  TextureFeature::CL_HIST_HELL,
            TextureFeature::EXT_Lbp,    TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_POL,
            TextureFeature::EXT_Lbp,    TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_HEL,
            TextureFeature::EXT_Lbp,    TextureFeature::FIL_DCT8,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_LBP_P,  TextureFeature::FIL_DCT8,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_MTS_P,  TextureFeature::FIL_NONE,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_MTS_P,  TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_POL,
            TextureFeature::EXT_MTS,    TextureFeature::FIL_HELL,  TextureFeature::CL_SVM_INT2,
            TextureFeature::EXT_COMB_G, TextureFeature::FIL_NONE,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_COMB_G, TextureFeature::FIL_HELL,  TextureFeature::CL_SVM_INT2,
            TextureFeature::EXT_COMB_P, TextureFeature::FIL_NONE,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_COMB_P, TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_POL,
            TextureFeature::EXT_COMB_P, TextureFeature::FIL_HELL,  TextureFeature::CL_SVM_INT2,
            TextureFeature::EXT_TPLBP_P, TextureFeature::FIL_DCT8, TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_TPLBP_G, TextureFeature::FIL_HELL, TextureFeature::CL_SVM_INT2,
            TextureFeature::EXT_FPLbp,   TextureFeature::FIL_NONE, TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_FPLBP_P, TextureFeature::FIL_NONE, TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_FPLBP_P, TextureFeature::FIL_NONE, TextureFeature::CL_SVM_POL,
            TextureFeature::EXT_FPLBP_P, TextureFeature::FIL_HELL, TextureFeature::CL_SVM_INT2,
            TextureFeature::EXT_FPLBP_P, TextureFeature::FIL_HELL, TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_BGC1_P,  TextureFeature::FIL_WHAD, TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_HDLBP,  TextureFeature::FIL_HELL,  TextureFeature::CL_SVM_INT2,
            TextureFeature::EXT_HDLBP,  TextureFeature::FIL_WHAD,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_Sift,   TextureFeature::FIL_DCT12, TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_Sift,   TextureFeature::FIL_NONE,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_Sift,   TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_HEL,
            TextureFeature::EXT_Sift,   TextureFeature::FIL_HELL,  TextureFeature::CL_SVM_INT2,
            TextureFeature::EXT_Sift_G, TextureFeature::FIL_DCT8,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_Grad_P, TextureFeature::FIL_NONE,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_Grad_P, TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_HEL,
            TextureFeature::EXT_GradMag,TextureFeature::FIL_NONE,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_GradMag_P,TextureFeature::FIL_WHAD,  TextureFeature::CL_PCA_LDA,
            TextureFeature::EXT_GradMag_P,TextureFeature::FIL_NONE,  TextureFeature::CL_SVM_HEL,
            TextureFeature::EXT_GradMag_P,TextureFeature::FIL_DCT8,  TextureFeature::CL_SVM_INT2,

            -1,-1,-1
        };
        for (int i=0; tests[i]>-1; i+=3)
            runtest(tests[i], tests[i+1], tests[i+2], images, labels, persons, fold);
    }
    return 0;
}



