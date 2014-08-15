#include "extractDB.h"

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
#include "tan_triggs.h"


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

int extractDB(const string &txtfile, vector<Mat> & images, Mat & labels, int preproc, int maxim)
{
    // read face db
    int fixed_size = 90;
    vector<string> vec;
    vector<int> vlabels; 
    int nsubjects = 1 + readtxt(txtfile.c_str(), vec, vlabels, maxim);

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
            case 3: 
            {
                Ptr<CLAHE> clahe = createCLAHE();
                clahe->setClipLimit(50); 
                clahe->apply(m2, mm); 
                break;
            }
        }            
        images.push_back(mm);
        labels.push_back(vlabels[i]);
        //if ( i%33==0) imshow("i",mm), waitKey(0);
    }
    return nsubjects;
}