//
// run test db against precompiled trainingset
//

//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;
using namespace cv;

#include "factory.h"

double ct(int64 t)
{
    return double(t) / cv::getTickFrequency();
}

//
// path label
//
int readtxt( const char * fname, std::vector<std::string> & names, std::vector<int> & labels, size_t maxim  ) 
{
    int maxid=-1;
    std::ifstream in(fname);
    while( in.good() && !in.eof() )
    {
        std::string path ;
        in >> path;
        names.push_back(path);

        int label;
        in >> label;
        labels.push_back(label);

        maxid=std::max(maxid,label);
        if ( labels.size() >= maxim ) 
            break;
    }
    return maxid;
}


const char *rec_names[] = {
    "fisher",
    "eigen",
    "lbph",
    "lbph2_u",
    "minmax",
    "norml2"
};


//
// run test db against precompiled trainingset
//
// face3read test.txt 4
// face      db       method 
//
int main(int argc, const char *argv[]) 
{
    vector<Mat> images;
    vector<int> labels;
    vector<string> vec;

    std::string imgpath("my.txt");
    if ( argc>1 ) imgpath=argv[1];

    size_t method = 4;
    if ( argc>2 ) method=atoi(argv[2]);

    size_t maxim = 400;
    if ( argc>3 ) maxim=atoi(argv[3]);

    if ( argc==1 )
    {
        cerr << argv[0] << " " << imgpath << " " << method << endl;
    }

    // read test face db
    int nsubjects = 1 + readtxt(imgpath.c_str(),vec,labels, maxim);

    int64 t0 = cv::getTickCount();
    // read the images
    vector<int> clabels;
    for ( size_t i=0; i<vec.size(); i++ )
    {
        Mat mm = imread(vec[i],0);
        if ( mm.empty() )
        {
            continue;
        }
        Mat m2;
        resize(mm,m2,Size(90,90));
        cv::equalizeHist( m2, mm );
        images.push_back(mm);
        clabels.push_back(labels[i]);
    }
    cout << images.size() << " images " << endl;
    labels = clabels;

    int64 t1 = cv::getTickCount();

    Ptr<FaceRecognizer> model;

    int rec = method;
    if ( rec == 0 ) {
        model = createFisherFaceRecognizer();
    } else if ( rec == 1 ) {
        model = createEigenFaceRecognizer();
    } else if ( rec == 2 ) {
        model = createLBPHFaceRecognizer(1,8,8,8,DBL_MAX);
    } else if ( rec == 3 ) {
        model = createLBPHFaceRecognizer2(1,8,8,8,DBL_MAX,true);
    } else if ( rec == 4 ) {
        model = createMinMaxFaceRecognizer(10,4);
    } else {
        model = createLinearFaceRecognizer(NORM_L2);
    }


    if ( rec==4 )
        model->load(format("%s.png",rec_names[rec]));
    else
        model->load(format("%s.yml",rec_names[rec]));

    int64 t2 = cv::getTickCount();

    double error = 0.0;
    // test model
    int ntests = images.size();
    int misses = 0;
    for ( int i=0; i<ntests; i++ )
    {
        double dist = 0;
        int predicted = -1;
        model->predict(images[i],predicted,dist);
        int missed = ( predicted != labels[i] );
        misses += missed;
    }

    error = double(misses)/ntests;

    int64 t3 = cv::getTickCount();
    double me = error;
    cout << format(" %-12s %-10.3f %-10.3f (%-3.3f %3.3f %3.3f)",rec_names[rec], me, 1.0-me, ct(t1-t0), ct(t2-t1),ct(t3-t2)) << endl;
    return 0;
}
