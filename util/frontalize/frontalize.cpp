#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/core/core_c.h" // shame, but needed for using dlib
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
//using namespace dlib; // ** don't ever try to**

using namespace cv; // this has to go later than the dlib includes

#include <iostream>
#include <vector>
using namespace std;

#include "frontalizer.h"


//
// please see:
//  "Effective Face Frontalization in Unconstrained Images"
//      Tal Hassner, Shai Harel, Eran Paz1 , Roee Enbar
//          The open University of Israel
//
struct FrontalizerImpl : public Frontalizer
{
    const dlib::shape_predictor &sp;
    const bool DEBUG_IMAGES;
    const double symBlend;
    const int symThresh;
    const int crop;

    Mat mdl;
    Mat_<double> eyemask;
    vector<Point3d> pts3d;

    FrontalizerImpl(const dlib::shape_predictor &sp, int crop, int symThreshold, double symBlend, bool debug_images)
        : sp(sp)
        , crop(crop)
        , symThresh(symThreshold)
        , symBlend(symBlend)
        , DEBUG_IMAGES(debug_images)
    {
        // model is rotated 90° already, but still in col-major, right hand coords
        FileStorage fs("data/mdl.yml.gz", FileStorage::READ);
        fs["mdl"] >> mdl;
        fs["eyemask"] >> eyemask;
        blur(eyemask,eyemask,Size(4,4));

        //// if you want to see the 3d model ..
        //Mat ch[3];
        //split(mdl, ch);
        //Mat_<double> depth;
        //normalize(ch[1], depth, -100);
        //imshow("head1", depth);

        // get 2d reference points from image
        vector<Point2d> pts2d;
        Mat meanI = imread("data/reference_320_320.png", 0);
        getkp2d(meanI, pts2d, Rect(80,80, 160,160));


        // get 3d reference points from model
        for(size_t k=0; k<pts2d.size(); k++)
        {
            Vec3d pm = mdl.at<Vec3d>(int(pts2d[k].y), int(pts2d[k].x));
            Point3d p(pm[0], pm[2], -pm[1]);
            pts3d.push_back(p);
        }
    }

    //
    // mostly stolen from Roy Shilkrot's HeadPosePnP
    //
    Mat pnp(const Size &s, vector<Point2d> &pts2d) const
    {
        // camMatrix based on img size
        int max_d = std::max(s.width,s.height);
	    Mat camMatrix = (Mat_<double>(3,3) <<
            max_d,   0, s.width/2.0,
			0,	 max_d, s.height/2.0,
			0,   0,	    1.0);

        // 2d -> 3d correspondence
        Mat rvec,tvec;
        solvePnP(pts3d, pts2d, camMatrix, Mat(1,4,CV_64F,0.0), rvec, tvec, false, SOLVEPNP_EPNP);
        cerr << "rot " << rvec.t() *180/CV_PI << endl;
        cerr << "tra " << tvec.t() << endl;
        // get 3d rot mat
	    Mat rotM(3, 3, CV_64F);
	    Rodrigues(rvec, rotM);

        // push tvec to transposed Mat
        Mat rotMT = rotM.t();
        rotMT.push_back(tvec.reshape(1, 1));

        // transpose back, and multiply
        return camMatrix * rotMT.t();
    }

    //! expects grayscale img
    void getkp2d(const Mat &I, vector<Point2d> &pts2d, const Rect &r) const
    {
        dlib::rectangle rec(r.x, r.y, r.x+r.width, r.y+r.height);
        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(I), rec);

        for(size_t k=0; k<shape.num_parts(); k++)
        {
            Point2d p(shape.part(k).x(), shape.part(k).y());
            pts2d.push_back(p);
        }
    }

    inline Mat1d project_vec(const Mat &KP, const Vec3d & p) const
    {
	    return  KP * (Mat1d)Matx41d(p[0], p[2], -p[1], 1.0); // swizzle to left-handed coords (from matlab's)
    }
    inline Mat1d project_vec(const Mat &KP, int row, int col) const
    {
	    return  project_vec(KP, mdl.at<Vec3d>(row, col));
    }

    //
    // thanks again, Haris. i wouldn't be anywhere without your mind here.
    //
    Mat project3d(const Mat & test) const
    {
        int mid = mdl.cols/2;
        int midi = test.cols/2;
        Rect R(mid-crop/2,mid-crop/2,crop,crop);
        Rect Ri(midi-crop/2,midi-crop/2,crop,crop);

        // get landmarks
        vector<Point2d> pts2d;
        getkp2d(test, pts2d, Ri);
        //cerr << "nose :" << pts2d[30].x << endl;

        // get pose mat for our landmarks
        Mat KP = pnp(test.size(), pts2d);

        // project img to head, count occlusions
        Mat_<uchar> test2(mdl.size(),127);
        Mat_<uchar> counts(mdl.size(),0);
	    for (int i=R.y; i<R.y+R.height; i++)
        {
	        for (int j=R.x; j<R.x+R.width; j++)
            {
                Mat1d p = project_vec(KP, i, j);
		        int x = int(p(0) / p(2));
		        int y = int(p(1) / p(2));
                if (y < 0 || y > test.rows - 1) continue;
                if (x < 0 || x > test.cols - 1) continue;
                // stare hard at the coord transformation ;)
                test2(i, j) = test.at<uchar>(y, x);
                // each point used more than once is occluded
                counts(y, x) ++;
	        }
        }

        // project the occlusion counts in the same way
        Mat_<uchar> counts1(mdl.size(),0);
	    for (int i=R.y; i<R.y+R.height; i++)
        {
	        for (int j=R.x; j<R.x+R.width; j++)
            {
                Mat1d p = project_vec(KP, i, j);
		        int x = int(p(0) / p(2));
		        int y = int(p(1) / p(2));
                if (y < 0 || y > test.rows - 1) continue;
                if (x < 0 || x > test.cols - 1) continue;
                counts1(i, j) = counts(y, x);
	        }
        }
        blur(counts1, counts1, Size(9,9));
        counts1 -= eyemask;
        counts1 -= eyemask;

        // count occlusions in left & right half
        Rect left (0,  0,mid,counts1.rows);
        Rect right(mid,0,mid,counts1.rows);
        double sleft=sum(counts1(left))[0];
        double sright=sum(counts1(right))[0];

        // fix occlusions with soft symmetry
        Mat_<double> weights;
        Mat_<uchar> sym = test2.clone();
        if (abs(sleft-sright)>symThresh)
        {
            // make weights
            counts1.convertTo(weights,CV_64F);

            Point p,P;
            double m,M;
            minMaxLoc(weights,&m,&M,&p,&P);

            double *wp = weights.ptr<double>();
            for (size_t i=0; i<weights.total(); ++i)
                wp[i] = (1.0 - 1.0 / exp(symBlend+(wp[i]/M)));
            // cerr << weights(Rect(mid,mid,6,6)) << endl;

            for (int i=R.y; i<R.y+R.height; i++)
            {
                if (sleft-sright>symThresh) // left side needs fixing
                {
                    for (int j=R.x; j<mid; j++)
                    {
                        int k = mdl.cols-j-1;
                        sym(i,j) = test2(i,j) * (1-weights(i,j)) + test2(i,k) * (weights(i,j));
                    }
                }
                if (sright-sleft>symThresh) // right side needs fixing
                {
                    for (int j=mid; j<R.x+R.width; j++)
                    {
                        int k = mdl.cols-j-1;
                        sym(i,j) = test2(i,j) * (1-weights(i,j)) + test2(i,k) * (weights(i,j));
                    }
                }
            }
        }

        if (DEBUG_IMAGES)
        {
            cerr << (sleft-sright) << "\t" << (abs(sleft-sright)>symThresh) << endl;
            imshow("proj",test2);
            if (abs(sleft-sright)>symThresh)
                imshow("weights", weights);
            Mat t = test.clone();
            rectangle(t,Ri,Scalar(255));
            for (size_t i=0; i<pts2d.size(); i++)
                circle(t, pts2d[i], 1, Scalar(0));
            imshow("test3",t);
        }

        Mat gray;
        sym.convertTo(gray,CV_8U);

        return sym(R);
    }

    //
    //! 2d eye-alignment
    //
    Mat align2d(const Mat &img) const
    {
        Mat test;
        resize(img, test, Size(250,250), INTER_CUBIC);

        // get landmarks
        vector<Point2d> pts2d;
        getkp2d(test, pts2d, Rect(0, 0, test.cols, test.rows));

        Point2d eye_l = (pts2d[37] + pts2d[38] + pts2d[40] + pts2d[41]) * 0.25; // left eye center
        Point2d eye_r = (pts2d[43] + pts2d[44] + pts2d[46] + pts2d[47]) * 0.25; // right eye center

        double eyeXdis = eye_r.x - eye_l.x;
        double eyeYdis = eye_r.y - eye_l.y;
        double angle   = atan(eyeYdis/eyeXdis);
        double degree  = angle*180/CV_PI;
        double scale   = 44.0 / eyeXdis; // scale to lfw eye distance

        Mat res;
        Point2f center(test.cols/2, test.rows/2);
        Mat rot = getRotationMatrix2D(center, degree, scale);
        cerr << rot << endl;
        //rot.at<float>(1,2) += eye_l.y - 
        warpAffine(test, res, rot, Size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(127));

        if (DEBUG_IMAGES)
        {
            Mat t = test.clone();
            for (size_t i=0; i<pts2d.size(); i++)
                circle(t, pts2d[i], 1, Scalar(0));
            circle(t, eye_l, 3, Scalar(255));
            circle(t, eye_r, 3, Scalar(255));
            imshow("test2",t);
            imshow("testr",res);
        }
        return res;
    }

    static Ptr<Frontalizer> create(const dlib::shape_predictor &sp, int crop, int symThreshold, double symBlend, bool write)
    {
        return makePtr<FrontalizerImpl>(sp, crop, symThreshold, symBlend, write);
    }

};




//#define FRONTALIZER_STANDALONE
#ifdef FRONTALIZER_STANDALONE

int main(int argc, const char *argv[])
{
    const char *keys =
            "{ help h usage ? |      | show this message }"
            "{ write w        |true | (over)write images (else just show them) }"
            "{ facedet f      |true | do a 2d face detection/crop(first) }"
            "{ align2d a      |true | do a 2d eye alignment(first) }"
            "{ project3d P    |true  | do 3d projection }"
            "{ crop c         |110   | crop size }"
            "{ sym s          |9000  | threshold for soft sym }"
            "{ blend b        |0.7   | blend factor for soft sym }"
            "{ path p         |lfw-deepfunneled/*.jpg| path to data folder}"
            "{ cascade C      |E:\\code\\opencv\\data\\haarcascades\\|\n     path to haarcascades folder}"
            "{ dlibpath d     |data/shape_predictor_68_face_landmarks.dat|\n     path to dlib landmarks model}";


    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="")
    {
        parser.printMessage();
        return -1;
    }
    string dlib_path = parser.get<String>("dlibpath");
    string casc_path = parser.get<String>("cascade");
    int crop = parser.get<int>("crop");
    int sym = parser.get<int>("sym");
    double blend = parser.get<double>("blend");
    bool write = parser.get<bool>("write");
    bool facedet = parser.get<bool>("facedet");
    bool align2d = parser.get<bool>("align2d");
    bool project3d = parser.get<bool>("project3d");

    dlib::shape_predictor sp;
    dlib::deserialize(dlib_path) >> sp;

    FrontalizerImpl front(sp,crop,sym,blend,!write);
    CascadeClassifier casc(casc_path + "haarcascade_frontalface_alt.xml");
    CascadeClassifier cascp(casc_path + "haarcascade_profileface.xml");
    //
    // !!!
    // if write is enabled,
    // please run this on a **copy** of your img folder,
    //  since this will just replace the images
    //  with the frontalized version !
    // !!!
    //
    if (! write)
    {
        namedWindow("orig", 0);
        namedWindow("front", 0);
    }

    vector<String> str;
    glob(path, str, true);
    for (size_t i=0; i<str.size(); i++)
    {
        cerr << str[i] << endl;
        Mat in = imread(str[i], 0);
        if (! write)
        {
            imshow("orig", in);
        }
        if (facedet && !casc.empty())
        {
            vector<Rect> rects;
            casc.detectMultiScale(in, rects, 1.3, 4);
            if (rects.size() > 0)
            {
                cerr << "frontal " << rects[0] << endl;
                in = in(rects[0]);
            }
            else
            {
                cascp.detectMultiScale(in, rects, 1.3, 4);
                if (rects.size() > 0)
                {
                    in = in(rects[0]);
                }
                else
                {
                    flip(in,in,1);
                    cascp.detectMultiScale(in, rects, 1.3, 4);
                    if (rects.size() > 0)
                    {
                        in = in(rects[0]);
                    }
                }
            }
        }

        if (align2d)
            in = front.align2d(in);

        Mat out = in;
        if (project3d)
           out = front.project3d(in);

        if (! write)
        {
            imshow("front", out);
            if (waitKey() == 27) break;
        }
        else
        {
            imwrite(str[i], out);
        }
    }
    return 0;
}

#endif // FRONTALIZER_STANDALONE
