#include "landmarks.h"


#define HAVE_FACEX
#ifdef HAVE_FACEX

#include "util/faceX/face_x.h"

struct LandMarks : Landmarks
{
    FaceX face_x;
	LandMarks() : face_x("util/faceX/model.xml.gz") {}
		
    virtual int extract(const cv::Mat &img, std::vector<cv::Point> &pt) const
	{
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
		return int(pt.size());
	}
};

//
// use dlib's implementation for facial landmarks,
// if not present, fall back to a precalculated
// 'one-size-fits-all' set of points(based on the mean lfw image)
//

#elif HAVE_DLIB // 20 pt subset of dlib's landmarks
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
        //int idx[] = {18,25, 20,24, 21,22, 27,29, 31,35, 38,43, 51, 0};
        for(int k=0; (k<40) && (idx[k]>0); k++)
            kp.push_back(cv::Point(crop(shape.part(idx[k]).x(),img.cols), crop(shape.part(idx[k]).y(),img.rows)));
            //kp.push_back(Point(shape.part(idx[k]).x(), shape.part(idx[k]).y()));
        //dlib::point p1 = shape.part(31) + (shape.part(39) - shape.part(31)) * 0.5; // left of nose
        //dlib::point p2 = shape.part(35) + (shape.part(42) - shape.part(35)) * 0.5;
        //dlib::point p3 = shape.part(36) + (shape.part(39) - shape.part(36)) * 0.5; // left eye center
        //dlib::point p4 = shape.part(42) + (shape.part(45) - shape.part(42)) * 0.5; // right eye center
        //dlib::point p5 = shape.part(31) + (shape.part(48) - shape.part(31)) * 0.5; // betw.mouth&nose
        //dlib::point p6 = shape.part(35) + (shape.part(54) - shape.part(35)) * 0.5; // betw.mouth&nose
        //kp.push_back(Point(p1.x(), p1.y()));
        //kp.push_back(Point(p2.x(), p2.y()));
        //kp.push_back(Point(p3.x(), p3.y()));
        //kp.push_back(Point(p4.x(), p4.y()));
        //kp.push_back(Point(p5.x(), p5.y()));
        //kp.push_back(Point(p6.x(), p6.y()));

        return (int)kp.size();
    }
};
#elif HAVE_ELASTIC // pre-trained discriminant parts based landmarks
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

        //kp.push_back(Point(15,19));    kp.push_back(Point(75,19));
        //kp.push_back(Point(29,20));    kp.push_back(Point(61,20));
        //kp.push_back(Point(36,24));    kp.push_back(Point(54,24));
        //kp.push_back(Point(38,35));    kp.push_back(Point(52,35));
        //kp.push_back(Point(30,39));    kp.push_back(Point(60,39));
        //kp.push_back(Point(19,39));    kp.push_back(Point(71,39));
        //kp.push_back(Point(8 ,38));    kp.push_back(Point(82,38));
        //kp.push_back(Point(40,64));    kp.push_back(Point(50,64));
        //kp.push_back(Point(31,75));    kp.push_back(Point(59,75));
        //kp.push_back(Point(27,81));    kp.push_back(Point(63,81));
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

cv::Ptr<Landmarks> createLandmarks() { return cv::makePtr<LandMarks>(); }


//
// 64 hardcoded, precalculated gftt keypoints from the 90x90 cropped mean lfw2 img
//
void gftt64(const cv::Mat &img, std::vector<cv::KeyPoint> &kp)
{
    using namespace cv;
    static const float kpsize = 16.0f;
    kp.push_back(cv::KeyPoint(14.f, 33.f, kpsize));        kp.push_back(cv::KeyPoint(29.f, 77.f, kpsize));        kp.push_back(cv::KeyPoint(55.f, 60.f, kpsize));
    kp.push_back(cv::KeyPoint(63.f, 76.f, kpsize));        kp.push_back(cv::KeyPoint(76.f, 32.f, kpsize));        kp.push_back(cv::KeyPoint(35.f, 60.f, kpsize));
    kp.push_back(cv::KeyPoint(69.f, 21.f, kpsize));        kp.push_back(cv::KeyPoint(45.f, 30.f, kpsize));        kp.push_back(cv::KeyPoint(27.f, 31.f, kpsize));
    kp.push_back(cv::KeyPoint(64.f, 26.f, kpsize));        kp.push_back(cv::KeyPoint(21.f, 22.f, kpsize));        kp.push_back(cv::KeyPoint(25.f, 27.f, kpsize));
    kp.push_back(cv::KeyPoint(69.f, 31.f, kpsize));        kp.push_back(cv::KeyPoint(54.f, 81.f, kpsize));        kp.push_back(cv::KeyPoint(62.f, 30.f, kpsize));
    kp.push_back(cv::KeyPoint(20.f, 32.f, kpsize));        kp.push_back(cv::KeyPoint(52.f, 33.f, kpsize));        kp.push_back(cv::KeyPoint(37.f, 32.f, kpsize));
    kp.push_back(cv::KeyPoint(38.f, 81.f, kpsize));        kp.push_back(cv::KeyPoint(36.f, 82.f, kpsize));        kp.push_back(cv::KeyPoint(32.f, 31.f, kpsize));
    kp.push_back(cv::KeyPoint(78.f, 17.f, kpsize));        kp.push_back(cv::KeyPoint(59.f, 24.f, kpsize));        kp.push_back(cv::KeyPoint(30.f, 24.f, kpsize));
    kp.push_back(cv::KeyPoint(11.f, 18.f, kpsize));        kp.push_back(cv::KeyPoint(13.f, 17.f, kpsize));        kp.push_back(cv::KeyPoint(56.f, 30.f, kpsize));
    kp.push_back(cv::KeyPoint(73.f, 15.f, kpsize));        kp.push_back(cv::KeyPoint(19.f, 15.f, kpsize));        kp.push_back(cv::KeyPoint(57.f, 53.f, kpsize));
    kp.push_back(cv::KeyPoint(33.f, 54.f, kpsize));        kp.push_back(cv::KeyPoint(34.f, 52.f, kpsize));        kp.push_back(cv::KeyPoint(49.f, 25.f, kpsize));
    kp.push_back(cv::KeyPoint(66.f, 33.f, kpsize));        kp.push_back(cv::KeyPoint(55.f, 49.f, kpsize));        kp.push_back(cv::KeyPoint(61.f, 33.f, kpsize));
    kp.push_back(cv::KeyPoint(39.f, 29.f, kpsize));        kp.push_back(cv::KeyPoint(60.f, 46.f, kpsize));        kp.push_back(cv::KeyPoint(40.f, 26.f, kpsize));
    kp.push_back(cv::KeyPoint(41.f, 76.f, kpsize));        kp.push_back(cv::KeyPoint(50.f, 76.f, kpsize));        kp.push_back(cv::KeyPoint(53.f, 41.f, kpsize));
    kp.push_back(cv::KeyPoint(44.f, 23.f, kpsize));        kp.push_back(cv::KeyPoint(29.f, 60.f, kpsize));        kp.push_back(cv::KeyPoint(54.f, 54.f, kpsize));
    kp.push_back(cv::KeyPoint(30.f, 47.f, kpsize));        kp.push_back(cv::KeyPoint(45.f, 50.f, kpsize));        kp.push_back(cv::KeyPoint(83.f, 35.f, kpsize));
    kp.push_back(cv::KeyPoint(36.f, 54.f, kpsize));        kp.push_back(cv::KeyPoint(13.f, 46.f, kpsize));        kp.push_back(cv::KeyPoint(36.f, 44.f, kpsize));
    kp.push_back(cv::KeyPoint(83.f, 38.f, kpsize));        kp.push_back(cv::KeyPoint(49.f, 53.f, kpsize));        kp.push_back(cv::KeyPoint(33.f, 83.f, kpsize));
    kp.push_back(cv::KeyPoint(17.f, 88.f, kpsize));        kp.push_back(cv::KeyPoint(31.f, 63.f, kpsize));        kp.push_back(cv::KeyPoint(13.f, 27.f, kpsize));
    kp.push_back(cv::KeyPoint(50.f, 62.f, kpsize));        kp.push_back(cv::KeyPoint(11.f, 43.f, kpsize));        kp.push_back(cv::KeyPoint(45.f, 55.f, kpsize));
    kp.push_back(cv::KeyPoint(45.f, 56.f, kpsize));        kp.push_back(cv::KeyPoint(79.f, 43.f, kpsize));        kp.push_back(cv::KeyPoint(74.f, 88.f, kpsize));
    kp.push_back(cv::KeyPoint(41.f, 62.f, kpsize));
    if (img.size() != cv::Size(90,90))
    {
        float scale_x=float(img.cols)/90;
        float scale_y=float(img.rows)/90;
        for (size_t i=0; i<kp.size(); i++)
        {
            kp[i].pt.x *= scale_x;
            kp[i].pt.y *= scale_y;
        }
    }
}




