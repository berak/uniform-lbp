#include "SpatialHistogramReco.h"
#include <iostream>



class CombinedLBPH : public SpatialHistogramReco
{
protected:
    virtual void oper(const Mat & src, Mat & hist) const ;

    virtual double distance(const Mat & hist_a, Mat & hist_b) const {
        return cv::norm(hist_a,hist_b,NORM_HAMMING2); 
    }

public:

    CombinedLBPH( int gridx=8, int gridy=8, double threshold = DBL_MAX) 
        : SpatialHistogramReco(gridx,gridy,threshold,8+8+16+16+16,CV_8U)
    {}

};



//------------------------------------------------------------------------------
// CombinedLBPH
//------------------------------------------------------------------------------

//
// per patch, write a combined (consecutive) histogram of 4 features:
// 16 bins for CSLBP as in [A Completed Modeling of Local Binary Pattern Operator for Texture Classification](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CLBP.pdf)
// 16 bins for diagonal outer connectors
// 8 bins for the maximal neighbour id
// 8 bins for the central intensity value
// so, that's 48 bytes per patch, 3072 per nose with 8x8 patches
//
//


void CombinedLBPH::oper(const Mat & src, Mat & hist) const {
    // calculate patterns
    // hardcoded 1 pixel radius
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            //
            // 7 0 1
            // 6 c 2
            // 5 4 3
            //
            uchar c   = src.at<uchar>(i,j);
            uchar n[8]= {
                src.at<uchar>(i-1,j),
                src.at<uchar>(i-1,j+1),
                src.at<uchar>(i,j+1),
                src.at<uchar>(i+1,j+1),
                src.at<uchar>(i+1,j),
                src.at<uchar>(i+1,j-1),
                src.at<uchar>(i,j-1),
                src.at<uchar>(i-1,j-1) 
            };
            unsigned code = 0;
            unsigned offset = 0;           
            // save 4 bits ( 1 for each of 4 possible diagonals )
            //  _\|/_
            //   /|\
            // this is the "central symmetric LBP" idea, from 
            // Zhenhua Guo, Lei Zhang, Member, IEEE, and David Zhang*, Fellow, IEEE ,
            // A Completed Modeling of Local Binary Pattern Operator for Texture Classification             
            //
            code = 0;
            code |= (n[0]>n[4]) << 0;
            code |= (n[1]>n[5]) << 1;
            code |= (n[2]>n[6]) << 2;
            code |= (n[3]>n[7]) << 3;
            hist.at<uchar>(code + offset) += 1;
            offset += 16;

            // save 4 bits ( 1 for each of 4 possible diagonal edges )
            //    / \
            //    \ /
            //
            code = 0;
            code |= (n[0]<n[2]) << 0;
            code |= (n[2]<n[4]) << 1;
            code |= (n[4]<n[6]) << 2;
            code |= (n[6]<n[0]) << 3;
            hist.at<uchar>(code + offset) += 1;
            offset += 16;

            //// save 4 bits ( 1 for each of 4 possible outer edges )
            ////  _ _
            //// |   |
            //// |_ _|
            ////
            code = 0;
            code |= (n[7]<n[1]) << 0;
            code |= (n[1]<n[3]) << 1;
            code |= (n[3]<n[5]) << 2;
            code |= (n[5]<n[7]) << 3;
            hist.at<uchar>(code + offset) += 1;
            offset += 16;

            // save 3 bits of max value ( it's index )
            code = 0;
            int m=-1;
            for ( int k=0; k<8; k++ )
            {
                if (n[k] > m) { m=n[k]; code=k; }
            }
            hist.at<uchar>(code + offset) += 1;
            offset += 8;
           
            // save 3 bits of center value:
            code = ((c >> 5) & 0x07); 
            hist.at<uchar>(code + offset) += 1;
            offset += 8;
        }
    }
}

Ptr<FaceRecognizer> createCombinedLBPHFaceRecognizer(int grid_x, int grid_y, double threshold)
{
    return makePtr<CombinedLBPH>(grid_x, grid_y, threshold);
}
