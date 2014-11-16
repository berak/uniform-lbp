#include "extractDB.h"

#include <opencv2/core.hpp>
#include <opencv2/bioinspired.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
using namespace cv;


#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
using namespace std;


#include "TextureFeature.h"


//
// taken from : https://github.com/bytefish/opencv/blob/master/misc/tan_triggs.cpp
//
Mat tan_triggs_preprocessing(InputArray src, float alpha=0.1, float tau=10.0, float gamma=0.2, int sigma0=1, int sigma1=2) 
{
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
// read a 'path <blank> label' list
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


//
// imglists per person.
//  no really, you can't just draw a random probability set from a set of multiple classes and call it a day ...
//
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

int extractDB(const string &txtfile, vector<Mat> & images, Mat & labels, int preproc, int maxim, int fixed_size)
{
    // read face db
    vector<string> vec;
    vector<int> vlabels; 
    int nsubjects = 1 + readtxt(txtfile.c_str(), vec, vlabels, maxim);

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(50); 

    Ptr<bioinspired::Retina> retina = bioinspired::createRetina(Size(fixed_size,fixed_size));


    //
    // read the images, 
    //   correct labels if empty images are skipped
    //   also apply preprocessing,
    //
    int load_flag = preproc==-1 ? 1 : 0;
    int skipped = 0;
    for ( size_t i=0; i<vec.size(); i++ )
    {
        Mat mm = imread(vec[i], load_flag);
        if ( mm.empty() )
        {
            skipped ++;
            continue;
        }
        Mat m2;
        resize(mm, m2, Size(fixed_size,fixed_size));
        switch(preproc) 
        {
            default:
            case 0: mm = m2; break;
            case 1: cv::equalizeHist( m2, mm ); break;
            case 2: cv::normalize( tan_triggs_preprocessing(m2), mm, 0, 255, NORM_MINMAX, CV_8UC1); break;
            case 3: clahe->apply(m2, mm); break;
            case 4: retina->run(m2); retina->getParvo(mm); break;
        }            
        images.push_back(mm);
        labels.push_back(vlabels[i]);
        //if ( i%33==0) imshow("i",mm), waitKey(0);
    }
    return nsubjects;
}