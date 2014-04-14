//
// run k fold crossvalidation train/test on  person db
//

#include <opencv2/highgui.hpp>
#include "opencv2/contrib.hpp"
#include <opencv2/core/utility.hpp>
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;
using namespace cv;


#define DPFX ""
//#ifdef _DEBUG 
// #define DPFX "_d"
//#else
// #define DPFX "_r"
//#endif //_DEBUG 
//

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


double ct(int64 t)
{
    return double(t) / cv::getTickFrequency();
}


const char *rec_names[] = {
    "fisher",
    "eigen",
    "lbph",
    "lbph2_u",
    "minmax",
    "lbp_comb",
    "lbp_var",
    "ltph",
    "clbpdist",
    "wld",
    "mom",
    "zernike",
    "norml2"
};

void setupPersons( const vector<int> & labels, vector<vector<int>> & persons )
{
    // find out which index belongs to which person
    //
    persons.resize(1);
    int prvid=0;
    for ( size_t j=0; j<labels.size(); j++ )
    {
        int id = labels[j];
        if (prvid!=id)
        {
            persons.push_back(vector<int>());
            prvid=id;
        }
        persons.back().push_back(j);
    }
}

#include "factory.h"

//
// for each fold, take alternating n/fold items for test, the other for training
//
Ptr<FaceRecognizer> runtest( int rec, const vector<Mat>& images, const vector<int>& labels, const vector<vector<int>>& persons, size_t fold=10, bool verbose=false ) 
{
    int64 t0 = cv::getTickCount();
    Ptr<FaceRecognizer> model;

    switch ( rec ) {
        case 0: model = createFisherFaceRecognizer(); break;
        case 1: model = createEigenFaceRecognizer(); break;
        case 2: model = createLBPHFaceRecognizer(1,8,8,8,DBL_MAX); break;
        case 3: model = createLBPHFaceRecognizer2(1,8,8,8,DBL_MAX,true); break;
        case 4: model = createMinMaxFaceRecognizer(10,4); break;
        case 5: model = createCombinedLBPHFaceRecognizer(8,8,DBL_MAX); break;
        case 6: model = createVarLBPFaceRecognizer(8,8,DBL_MAX); break;
        case 7: model = createLTPHFaceRecognizer(25,8,8,DBL_MAX); break;
        case 8: model = createClbpDistFaceRecognizer(DBL_MAX); break;
        case 9: model = createWLDFaceRecognizer(8,8,DBL_MAX); break;
        case 10: model = createMomFaceRecognizer(8,10); break;
        case 11: model = createZernikeFaceRecognizer(2,7); break;
        default: model = createLinearFaceRecognizer(NORM_L2); break;
    }
    int64 t1 = cv::getTickCount();
    //
    // do nfold sliding window train & test runs
    //
    int64 dt1=0, dt2=0;
    double meanSqrError = 0.0;
    Mat confusion = Mat::zeros(persons.size(),persons.size(),CV_32S);
    for ( size_t f=0; f<fold; f++ )
    {
        vector<Mat> trainImages;
        vector<int> trainLabels;
        vector<Mat> testImages;
        vector<int> testLabels;

        // split train/test set per person:
        for ( size_t j=0; j<persons.size(); j++ )
        {
            size_t n_per_person = persons[j].size();
            if ( n_per_person < fold ) continue;
            int r = -1;
            if ( fold != 0 )
                r = n_per_person/fold;
            for ( size_t n=0; n<n_per_person; n++ )
            {
                int index = persons[j][n];
                if ( (fold>1) && (n >= f*r) && (n <= (f+1)*r) ) // sliding window per fold
                {
                    testImages.push_back(images[index]);
                    testLabels.push_back(labels[index]);
                }
                else
                {
                    trainImages.push_back(images[index]);
                    trainLabels.push_back(labels[index]);
                }
            }
        }

        model->train(trainImages, trainLabels);

        int64 t2 = cv::getTickCount();

        // test model
        int ntests = testImages.size();
        int misses = 0;
        for ( int i=0; i<ntests; i++ )
        {
            int label = testLabels[i];

            // test positive img:
            double dist = DBL_MAX;
            int predicted = -1;
            model->predict(testImages[i],predicted,dist);
            if ( verbose )
                confusion.at<int>(label,predicted) += 1;
            misses += (predicted != label);
        }
        double rror = double(misses)/ntests;
        meanSqrError += rror * rror;


        cout << '.';
        int64 t3 = cv::getTickCount();
        dt1 += t2-t1;
        dt2 += t3-t2;

        if ( verbose )
            cout << format(" %-12s %-6.3f (%3d %3d) %3d (%6.3f %6.3f %6.3f)",rec_names[rec], rror,trainImages.size(), testImages.size(), misses, ct(t1-t0),ct(t2-t1), ct(t3-t2) ) << endl;
    }
    double me = sqrt(meanSqrError) / fold;
    //if ( verbose )
    //   cerr << confusion << endl;
    cout << format(" %-12s %-10.3f %-10.3f (%6.3f %6.3f",rec_names[rec], me, (1.0-me), ct(dt1), ct(dt2)) ;
    return model;
}


//
// tried tan_triggs instead of equalizeHist, but no cigar.
// taken from : https://github.com/bytefish/opencv/blob/master/misc/tan_triggs.cpp
//
Mat tan_triggs_preprocessing(InputArray src,
        float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
        int sigma1 = 2) {

    // Convert to floating point:
    Mat X = src.getMat();
    X.convertTo(X, CV_32FC1);
    // Start preprocessing:
    Mat I;
    pow(X, gamma, I);
    // Calculate the DOG Image:
    {
        Mat gaussian0, gaussian1;
        // Kernel Size:
        int kernel_sz0 = (3*sigma0);
        int kernel_sz1 = (3*sigma1);
        // Make them odd for OpenCV:
        kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
        kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
        GaussianBlur(I, gaussian0, Size(kernel_sz0,kernel_sz0), sigma0, sigma0, BORDER_CONSTANT);
        GaussianBlur(I, gaussian1, Size(kernel_sz1,kernel_sz1), sigma1, sigma1, BORDER_CONSTANT);
        subtract(gaussian0, gaussian1, I);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(abs(I), alpha, tmp);
            meanI = mean(tmp).val[0];

        }
        I = I / pow(meanI, 1.0/alpha);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(min(abs(I), tau), alpha, tmp);
            meanI = mean(tmp).val[0];
        }
        I = I / pow(meanI, 1.0/alpha);
    }

    // Squash into the tanh:
    {
        for(int r = 0; r < I.rows; r++) {
            for(int c = 0; c < I.cols; c++) {
                I.at<float>(r,c) = tanh(I.at<float>(r,c) / tau);
            }
        }
        I = tau * I;
    }
    return I;
}



//
//
// face att.txt 5     5      1         1
// face db      fold  reco  verbose  preprocessing
//
// special: fold==1 will train on all images , save them, and ignore the test step
// special: reco==0 will run *all* recognizers available on a given db
//
int main(int argc, const char *argv[]) 
{
    vector<Mat> images;
    vector<int> labels;
    vector<string> vec;

    std::string imgpath("att.txt");
    if ( argc>1 ) imgpath=argv[1];

    size_t fold = 5;
    if ( argc>2 ) fold=atoi(argv[2]);

    int rec = 11;
    if ( argc>3 ) rec=atoi(argv[3]);

    bool verbose = true;
    if ( argc>4 ) verbose=(argv[4][0]=='1');

    bool save = (fold==1);

    int preproc = 1; // 0-none 1-eqhist 2-tantriggs 2-clahe
    if ( argc>5 ) preproc=atoi(argv[5]);

    // read face db
    size_t maxim  = 400;
    int nsubjects = 1 + readtxt(imgpath.c_str(),vec,labels, maxim);


    if ( argc==1 )
    {
        cerr << argv[0] << " " << imgpath << " " << fold << " " << rec << endl;
    }

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(50); // default:40 (!!)

    // read the images, correct labels if empty image skipped
    int load_flag = preproc==-1 ? 1 : 0;
    int skipped = 0;
    vector<int> clabels; 
    for ( size_t i=0; i<vec.size(); i++ )
    {
        Mat mm = imread(vec[i],load_flag);
        if ( mm.empty() )
        {
            skipped ++;
            continue;
        }
        Mat m2;
        resize(mm,m2,Size(90,90));
        switch(preproc) 
        {
            default:
            case 0: mm = m2; break;
            case 1: cv::equalizeHist( m2, mm ); break;
            case 2: cv::normalize( tan_triggs_preprocessing(m2), mm, 0, 255, NORM_MINMAX, CV_8UC1); break;
            case 3: clahe->apply(m2,mm); break;
        }            
        images.push_back(mm);
        clabels.push_back(labels[i]);
        //if ( i%33==0) imshow("i",mm), waitKey(0);
    }
    labels = clabels;

    // per person id lookup
    vector<vector<int>> persons;
    setupPersons( labels, persons );

    fold = std::min(fold,images.size()/nsubjects);

    cout << fold  << " fold, " ;
    cout << nsubjects  << " subjects, " ;
    cout << images.size() << " images, ";
    cout << images.size()/nsubjects << " per person ";
    if ( skipped ) cout << "(" << skipped << " images skipped)";
    cout << endl;

    int n=sizeof(rec_names)/sizeof(char*);
    if ( rec > 0 )
    {
        n = rec+1;
    }
    for ( ; rec<n; rec++ ) 
    {
        Ptr<FaceRecognizer> model = runtest(rec,images,labels,persons, fold,verbose);
        if ( save ) {
            int64 t0 = getTickCount();
            if ( rec==4 )
                model->save(format("%s.png",rec_names[rec]));
            else
                model->save(format("%s.yml",rec_names[rec]));
            int64 t1 = getTickCount();
            cout << format(" %6.3f",ct(t1-t0));
        }
        cout << ")" << endl;
    }
    return 0;
}
