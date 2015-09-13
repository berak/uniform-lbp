#include "landmarks.h"

//#define HAVE_ELASTIC
//#define HAVE_FACEX

#ifdef HAVE_FACEX

#include "util/faceX/face_x.h"

//
// https://github.com/delphifirst/FaceX/
//
struct LandMarks : Landmarks
{
    FaceX face_x;
    LandMarks() : face_x("util/faceX/model.xml.gz") {}

    virtual int extract(const cv::Mat &img, std::vector<cv::Point> &pt) const
    {
        // originally: 51 landmarks.
        static int lut[20] = {
            0,2,4, 5,7,9,  // eyebrows
            19,22, 25,28,  // eyecorners
            11,13,14,      // nose
            16,18,31,37,42,38,49 // mouth
        };
        std::vector<cv::Point2d> landmarks = face_x.Alignment(img, cv::Rect(0,0,img.cols,img.rows));
        pt.clear();
        for (size_t i=0; i<20; ++i)
        {
            pt.push_back(cv::Point(landmarks[lut[i]]));
        }
  //      cv::Mat viz; cv::cvtColor(img,viz,cv::COLOR_GRAY2BGR);
  //      for (size_t i=0; i<20; ++i)
  //      {
  //          cv::Point2d p = landmarks[lut[i]];
  //          cv::circle(viz,p,2,cv::Scalar(0,0,200),1);
  //          int w=12,h=12; // half size
  //          cv::rectangle(viz,cv::Rect(p.x-w,p.y-h,2*w,2*h),cv::Scalar(0,200,0),1);
  //      }
  //      cv::imshow("viz",viz);
  //      cv::waitKey(200);
        return int(pt.size());
    }
};


#elif defined(HAVE_DLIB) // 20 pt subset of dlib's landmarks

//
// use dlib's implementation for facial landmarks,
// if not present, fall back to a precalculated
// 'one-size-fits-all' set of points(based on the mean lfw image)
//
//
// 20 assorted keypoints extracted from the 68 dlib facial landmarks, based on the
//    Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf
//
 #include <dlib/image_processing.h>
 #include <dlib/opencv/cv_image.h>

struct LandMarks : Landmarks
{
    dlib::shape_predictor sp;

    int offset;
    LandMarks(int off=0)
        : offset(off)
    {   // it's only 95mb...
        dlib::deserialize("data/shape_predictor_68_face_landmarks.dat") >> sp;
    }

    inline int crop(int v,int M) const {return (v<offset?offset:(v>M-offset?M-offset:v)); }
    int extract(const cv::Mat &img, std::vector<cv::Point> &kp) const
    {
        dlib::rectangle rec(0,0,img.cols,img.rows);
        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(img), rec);

        int idx[] = {17,26, 19,24, 21,22, 36,45, 39,42, 38,43, 31,35, 51,33, 48,54, 57,27, 0};

        for(int k=0; (k<40) && (idx[k]>0); k++)
            kp.push_back(cv::Point(crop(shape.part(idx[k]).x(),img.cols),
                                   crop(shape.part(idx[k]).y(),img.rows)));
        return (int)kp.size();
    }
};


#elif defined(HAVE_ELASTIC) // pre-trained discriminant parts based landmarks
#include "util/elastic/elasticparts.h"
struct LandMarks : Landmarks
{
    cv::Ptr<ElasticParts> elastic;

    LandMarks(int off=0)
    {
        elastic = ElasticParts::createDiscriminative();
        elastic->read("data/disc.xml.gz");
        //elastic = ElasticParts::createGenerative();
        //elastic->read("data/parts.xml.gz");
    }

    int extract(const cv::Mat &img, std::vector<cv::Point> &kp) const
    {
        elastic->getPoints(img, kp);
        return (int)kp.size();
    }
};


#else // fixed manmade landmarks
struct LandMarks : Landmarks
{
    LandMarks(int off=0) {}
    int extract(const cv::Mat &img, std::vector<cv::Point> &kp) const
    {
        kp.push_back(cv::Point(15,19));    kp.push_back(cv::Point(75,19));
        kp.push_back(cv::Point(29,20));    kp.push_back(cv::Point(61,20));
        kp.push_back(cv::Point(36,24));    kp.push_back(cv::Point(54,24));
        kp.push_back(cv::Point(38,35));    kp.push_back(cv::Point(52,35));
        kp.push_back(cv::Point(30,39));    kp.push_back(cv::Point(60,39));
        kp.push_back(cv::Point(19,39));    kp.push_back(cv::Point(71,39));
        kp.push_back(cv::Point(12,38));    kp.push_back(cv::Point(77,38));
        kp.push_back(cv::Point(40,64));    kp.push_back(cv::Point(50,64));
        kp.push_back(cv::Point(31,75));    kp.push_back(cv::Point(59,75));
        kp.push_back(cv::Point(32,49));    kp.push_back(cv::Point(59,49));

        if (img.size() != cv::Size(90,90))
        {
            float scale_x=float(img.cols)/90;
            float scale_y=float(img.rows)/90;
            for (size_t i=0; i<kp.size(); i++)
            {
                kp[i].x *= scale_x;
                kp[i].y *= scale_y;
            }
        }
        return (int)kp.size();
    }
};
#endif

//
// factory
//
cv::Ptr<Landmarks> createLandmarks() { return cv::makePtr<LandMarks>(); }

