#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/utility.hpp"

 #include <dlib/image_processing.h>
 #include <dlib/opencv/cv_image.h>

using namespace cv;

#include <vector>
using std::vector;

#include <iostream>
using std::cerr;
using std::endl;
struct LandMarks
{
    dlib::shape_predictor sp;

    LandMarks()
    {   
        dlib::deserialize("data/shape_predictor_68_face_landmarks.dat") >> sp;
    }

    int extract(const Mat &img, vector<Point> &kp) const
    {
        dlib::rectangle rec(0,0,img.cols,img.rows);
        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(img), rec);

        int idx[] = {17,26, 19,24, 21,22, 36,45, 39,42, 38,43, 31,35, 51,33, 48,54, 57,27, 0};
        for(int k=0; (k<40) && (idx[k]>0); k++)
            kp.push_back(Point(shape.part(idx[k]).x(), shape.part(idx[k]).y()));
        return (int)kp.size();
    }
};

struct FeatureLbp
{
    int operator() (const Mat &I, Mat &fI) const
    {
        Mat_<uchar> feature(I.size(),0);
        Mat_<uchar> img(I);
        const int m=1;
        for (int r=m; r<img.rows-m; r++)
        {
            for (int c=m; c<img.cols-m; c++)
            {
                uchar v = 0;
                uchar cen = img(r,c);
                v |= (img(r-1,c  ) > cen) << 0;
                v |= (img(r-1,c+1) > cen) << 1;
                v |= (img(r  ,c+1) > cen) << 2;
                v |= (img(r+1,c+1) > cen) << 3;
                v |= (img(r+1,c  ) > cen) << 4;
                v |= (img(r+1,c-1) > cen) << 5;
                v |= (img(r  ,c-1) > cen) << 6;
                v |= (img(r-1,c-1) > cen) << 7;
                feature(r,c) = v;
            }
        }
        fI = feature;
        return 256;
    }
};

struct FeatureFPLbp
{
    int operator () (const Mat &img, Mat &features) const
    {
        Mat_<uchar> I(img);
        Mat_<uchar> fI(I.size(), 0);
        const int border=2;
        for (int r=border; r<I.rows-border; r++)
        {
            for (int c=border; c<I.cols-border; c++)
            {
                uchar v = 0;
                v |= ((I(r  ,c+1) - I(r+2,c+2)) > (I(r  ,c-1) - I(r-2,c-2))) * 1;
                v |= ((I(r+1,c+1) - I(r+2,c  )) > (I(r-1,c-1) - I(r-2,c  ))) * 2;
                v |= ((I(r+1,c  ) - I(r+2,c-2)) > (I(r-1,c  ) - I(r-2,c+2))) * 4;
                v |= ((I(r+1,c-1) - I(r  ,c-2)) > (I(r-1,c+1) - I(r  ,c+2))) * 8;
                fI(r,c) = v;
            }
        }
        features = fI;
        return 16;
    }
};

static void hist_patch(const Mat_<uchar> &fI, Mat &histo, int histSize=256)
{
    Mat_<float> h(1, histSize, 0.0f);
    for (int i=0; i<fI.rows; i++)
    {
        for (int j=0; j<fI.cols; j++)
        {
            int v = int(fI(i,j));
            h( v ) += 1.0f;
        }
    }
    histo.push_back(h.reshape(1,1));
}


static void hist_patch_uniform(const Mat_<uchar> &fI, Mat &histo, int histSize=256)
{
    static int uniform[256] =
    {   // the well known original uniform2 pattern
        0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
        14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
        58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
        58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
        58,58,58,50,51,52,58,53,54,55,56,57
    };

    Mat_<float> h(1, 60, 0.0f);
    for (int i=0; i<fI.rows; i++)
    {
        for (int j=0; j<fI.cols; j++)
        {
            int v = int(fI(i,j));
            h( uniform[v] ) += 1.0f;
        }
    }
    histo.push_back(h.reshape(1,1));
}


struct HighDimLbp
{
    FeatureLbp lbp;
    LandMarks land;

    int extract(const Mat &img, int k, Mat &trainData) 
    {
        int gr=10; // 10 used in paper
        vector<Point> kp;
        land.extract(img,kp);

        Mat histo;
        float scale[] = {0.75f, 1.06f, 1.5f, 2.2f, 3.0f}; // http://bcsiriuschen.github.io/High-Dimensional-LBP/
        float offsets_16[] = {
            -1.5f,-1.5f, -0.5f,-1.5f, 0.5f,-1.5f, 1.5f,-1.5f,
            -1.5f,-0.5f, -0.5f,-0.5f, 0.5f,-0.5f, 1.5f,-0.5f,
            -1.5f, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 0.5f,
            -1.5f, 1.5f, -0.5f, 1.5f, 0.5f, 1.5f, 1.5f, 1.5f
        };
        for (int i=0; i<5; i++)
        {
            float s = scale[i];
            int noff = 16;
            float *off = offsets_16;
            Mat f1,f2,imgs;
            resize(img,imgs,Size(),s,s);
            int histSize = lbp(imgs,f1);

            //for (size_t k=0; k<kp.size(); k++)
            {
                Mat h;
                Point2f pt(kp[k]);
                for (int o=0; o<noff; o++)
                {
                    Mat patch;
                    getRectSubPix(f1, Size(gr,gr), Point2f(pt.x*s + off[o*2]*gr, pt.y*s + off[o*2+1]*gr), patch);
                    hist_patch_uniform(patch, h, histSize);
                }
                trainData.push_back(h.reshape(1,1));
            }
        }
        return 1; 
    }
    //void train(int K=16*16*5/2) // fplbp / 2
    void train(FileStorage &fs, Mat &trainData, int K=58*16*5/4)  // lbpu   / 8
    {
        Mat td;
        normalize(trainData,td);
        td = td.reshape(1,trainData.rows/5);
        PCA p(td,Mat(),cv::PCA::DATA_AS_ROW,K);
        fs << "{:" ;
        p.write(fs);
        fs << "}";
    }
};


struct HighDimPcaSift
{
    Ptr<Feature2D> sift;
    LandMarks land;

    HighDimPcaSift()
        : sift(xfeatures2d::SIFT::create())
    {}

    int extract(const Mat &img, int k, Mat &trainData) 
    {
        int gr=5; // 10 used in paper
        vector<Point> pt;
        land.extract(img,pt);

        Mat histo;
        float scale[] = {0.75f, 1.06f, 1.5f, 2.2f, 3.0f}; // http://bcsiriuschen.github.io/High-Dimensional-LBP/
        float offsets_16[] = {
            -1.5f,-1.5f, -0.5f,-1.5f, 0.5f,-1.5f, 1.5f,-1.5f,
            -1.5f,-0.5f, -0.5f,-0.5f, 0.5f,-0.5f, 1.5f,-0.5f,
            -1.5f, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 0.5f,
            -1.5f, 1.5f, -0.5f, 1.5f, 0.5f, 1.5f, 1.5f, 1.5f
        };
        float offsets_9[] = {
            -1.f,-1.f, 0.f,-1.f, 1.f,-1.f,
            -1.f, 0.f, 0.f, 0.f, 1.f, 0.f,
            -1.f, 1.f, 0.f, 1.f, 1.f, 1.f,
        };
        int noff = 16;
        float *off = offsets_16;

        vector<KeyPoint> kp;
        Mat h;
        for (int o=0; o<noff; o++)
        {
            kp.push_back(KeyPoint(pt[k].x + off[o*2]*gr, pt[k].y + off[o*2+1]*gr,gr*2));
        }
        sift->compute(img,kp,h);
        Mat h2;
        for (size_t j=0; j<kp.size(); j++)
        {
            Mat hx = h.row(j).t();
            hx.push_back(float(kp[j].pt.x/img.cols - 0.5));
            hx.push_back(float(kp[j].pt.y/img.rows - 0.5));
            h2.push_back(hx.reshape(1,1));
        }
        trainData.push_back(h2);
        return 1; 
    }


    void train(FileStorage &fs, Mat &trainData, int K=20)  
    {
        Mat td = trainData;
        PCA p(td,Mat(),cv::PCA::DATA_AS_ROW,K);
        fs << "{:" ;
        p.write(fs);
        fs << "}";
    }
};


int main()
{
    //HighDimLbp hd;
    HighDimPcaSift hd;
    std::vector<String> fns;
    glob("lfw-deepfunneled/*.jpg",fns,true);
    if ( fns.empty())
        return 0;
    vector<Mat> images;
    RNG rn(getTickCount());
    FileStorage fs("data/hd_pcasift_20.xml.gz", FileStorage::WRITE);
    fs << "hd_pcasift" << "[";

    for (size_t k=0; k<20; k++)
    {
        Mat trainData;
        for (size_t i=0; i<2500; i++)
        {
            int id=rn.uniform(0,fns.size());
            Mat im = imread(fns[id],0);
            Mat i2 = im(Rect(80,80,90,90));
            hd.extract(i2,k,trainData);
            cerr << "extracted    " << i << '\r';
        }
        hd.train(fs, trainData);
        cerr << "trained " << k << '\n';
   }
    fs << "]";
    fs.release();
    cerr << "\n";
    return 0;
}
