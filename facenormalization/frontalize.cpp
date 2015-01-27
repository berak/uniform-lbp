//
// please see:
//  "Effective Face Frontalization in Unconstrained Images"
//      Tal Hassner, Shai Harel, Eran Paz1 , Roee Enbar
//          The open University of Israel
//


#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "opencv2/core/core_c.h" // shame, but needed for using dlib
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
//using namespace dlib; // ** don't ever try to**

using namespace cv; // this has to go later than the dlib includes

#include <iostream>
#include <vector>
using namespace std;

#include "../Profile.h"

const bool WRITE_IMAGES = 0;

//
//@brief  apply a one-size-fits-all 3d model transformation (POSIT style)
//
struct Frontalizer
{
    dlib::shape_predictor sp;
    vector<Point3d> pts3d;
    Mat mdl;
    Mat_<double> eyemask;
    Mat_<uchar>  mask;

    Frontalizer()
    {
        dlib::deserialize("D:/Temp/dlib-18.10/examples/shape_predictor_68_face_landmarks.dat") >> sp;

        // get 2d reference points from image
        vector<Point2d> pts2d;
        Mat meanI = imread("reference_320_320.png", 0);
        getkp(meanI, pts2d, 80, 160);

        // model is rotated 90° already, but still in col-major, right hand coords
        FileStorage fs("mdl.yml.gz", FileStorage::READ);
        fs["mdl"] >> mdl;
        fs["eyemask"] >> eyemask;

        // make depth masks
        Mat ch[4];
        split(mdl, ch);
        Mat_<double> depth;
        normalize(ch[1], depth, -100);
        mask = depth > 0;
        //imshow("head1", depth);
        //imshow("mask", mask);

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
    Mat pnp(const Size &s, vector<Point2d> &ip)
    {
        PROFILE
        // camMatrix based on img size
        int max_d = std::max(s.width,s.height);
	    Mat camMatrix = (Mat_<double>(3,3) <<
            max_d,   0, s.width/2.0,
			0,	 max_d, s.height/2.0,
			0,   0,	    1.0);

        // 2d -> 3d correspondence
        Mat rvec,tvec;
        solvePnP(pts3d, ip, camMatrix, Mat(1,4,CV_64F,0.0), rvec, tvec, false, SOLVEPNP_EPNP);

        // get 3d rot mat
	    Mat rotM(3, 3, CV_64F);
	    Rodrigues(rvec, rotM);

        // push tvec to transposed Mat
        Mat rotMT = rotM.t();
        rotMT.push_back(tvec.reshape(1, 1));

        // transpose back, and multiply
        return camMatrix * rotMT.t();
    }

    // expects grayscale img
    void getkp(const Mat &I, vector<Point2d> &pts2d, int off=0, int siz=0)
    {
        PROFILE
        dlib::rectangle rec(0, 0, I.cols, I.rows);
        if (off)
            rec = dlib::rectangle(off, off, off+siz, off+siz);

        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(I), rec);

        //int idx[] = {17,26, 19,24, 21,22, 36,45, 39,42, 38,43, 31,35, 51,33, 48,54, 57,27, 0};
        ////int idx[] = {18,25, 20,24, 21,22, 27,29, 31,35, 38,43, 51, 0};
        //for(int k=0; (k<40) && (idx[k]>0); k++)
        //    pts2d.push_back(Point2f(shape.part(idx[k]).x(), shape.part(idx[k]).y()));

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
    Mat project(const Mat & test, int kpoff, int kpsiz)
    {
        PROFILE

        // get landmarks
        vector<Point2d> pts2d;
        getkp(test, pts2d, kpoff, kpsiz);
        //cerr << "nose :" << pts2d[30].x << endl;
        // pose mat for our landmarks
        Mat KP = pnp(test.size(), pts2d);

        //
        //  " remap all points by defining 3d corresponding points for entire reference image
        //    .. this is the trick. 
        //    just go over their paper.. you will easily get it. need just 20 lines of opencv code... :) . 
        //      But will require full effort for at least 3 days from this understanding... "
        //
        //      ( yea, right, .. (facepalm) )
        //

        int crop=110;
        int mid = mdl.cols/2;
        Rect R(mid-crop/2,mid-crop/2,crop,crop);

        int offy = (mdl.rows-test.rows)/2;
        int offx = (mdl.cols-test.cols)/2;
        Mat_<uchar> test2(mdl.size(),0);
        Mat_<uchar> counts(mdl.size(),0);
	    for (int i=R.y; i<R.y+R.height; i++)
        {
            PROFILEX("proj_1");
	        for (int j=R.x; j<R.x+R.width; j++)
            {
                Mat1d p = project_vec(KP, i, j);
		        int x = int(p(0) / p(2));
		        int y = int(p(1) / p(2));
                if (y < 0 || y > test.rows - 1) continue;
                if (x < 0 || x > test.cols - 1) continue;
                // stare hard at the coord transformation ;)
                test2(i, j) = test.at<uchar>(y, x);
                counts(y, x) ++; // each point used more than once is occluded
	        }
        }

        //imshow("proj",test2&mask);

        // project the occlusion counts in the same way
        Mat_<uchar> counts1(mdl.size(),0);
	    for (int i=R.y; i<R.y+R.height; i++)
        {
            PROFILEX("proj_2");
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
        counts1 -= eyemask;
        blur(counts1 & mask, counts1, Size(9,9));

        // count occlusions in left & right half
        Rect left (0,  0,mid,counts1.rows);
        Rect right(mid,0,mid,counts1.rows);
        double sleft=sum(counts1(left))[0];
        double sright=sum(counts1(right))[0];
        //cerr << sleft << "\t" << sright << "\t" << (sleft-sright) << "\t" << (sleft+sright) << endl;
        // fix occlusions with soft symmetry
        Mat_<uchar> sym = test2.clone();
        if (abs(sleft-sright)>8000)
        {
            PROFILEX("proj_3");
            // make weights
            Mat_<double> weights;
            counts1.convertTo(weights,CV_64F);

            Point p,P;
            double m,M;
            minMaxLoc(weights,&m,&M,&p,&P);

            double *wp = weights.ptr<double>();
            for (size_t i=0; i<weights.total(); ++i)
                wp[i] = (1.0 - 1.0 / exp(0.7+(wp[i]/M)));
            // cerr << weights(Rect(mid,mid,6,6)) << endl;
            if (! WRITE_IMAGES)
                imshow("weights", weights);

            for (int i=R.y; i<R.y+R.height; i++)
            {
                if (sleft-sright>8000) // left side needs fixing
                {                
                    for (int j=R.x; j<mid; j++)
                    {
                        int k = mdl.cols-j-1;
                        sym(i,j) = test2(i,j) * (1-weights(i,j)) + test2(i,k) * (weights(i,j));
                    }
                }
                if (sright-sleft>8000) // right side needs fixing
                {
                    for (int j=mid; j<R.x+R.width; j++)
                    {
                        int k = mdl.cols-j-1;
                        sym(i,j) = test2(i,j) * (1-weights(i,j)) + test2(i,k) * (weights(i,j));
                    }
                }
            }
        }

        for (size_t i=0; i<pts2d.size(); i++)
            circle(test, pts2d[i], 1, Scalar(0));

        Mat gray;
        sym.convertTo(gray,CV_8U);

        if (! WRITE_IMAGES)
            imshow("sym", sym & mask);

        return sym(R);
    }
};


int main(int argc, const char *argv[])
{
    Frontalizer front;
    vector<String> str;
    //
    // !!!
    // please run this on a **copy** of your img folder,
    //  since this will just replace the images
    //  with the frontalized version !
    // !!!
    //
    //glob("e:/MEDIA/faces/tv/*.png",str,true);
    //glob("e:/MEDIA/faces/fem/*.png",str,true);
    //glob("e:/MEDIA/faces/sheffield",str,true);
    //glob("img/*.jpg", str, true);
    //glob("../lfw3d_b/*.jpg", str, true);
    glob("../lfw-deepfunneled/*.jpg", str, true);
    for (size_t i=0; i<str.size(); i++)
    {
        cerr << str[i] << endl;
        Mat in = imread(str[i], 0);
        Mat out = front.project(in, 80, 90); //lfw offsets
        //Mat out = front.project(in, 0, 0); //sheff
        if (! WRITE_IMAGES)
        {
            imshow("orig", in);
            imshow("rota", out);
            if (waitKey() == 27) break;
        } 
        else
        {
            imwrite(str[i], out);
        }
    }
    return 0;
}
