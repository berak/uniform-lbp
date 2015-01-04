//
//
// instead of using/parsing a csv file,
// we save each person to a seperate directory,
// and later parse the glob() output.
//
// use : online [capture id or path] [img_path] [cascade_path]
//
//

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;
using namespace cv;

#include "TextureFeature.h"
#include "Preprocessor.h"

#ifdef __linux__
const char SEP = '/';
#else
const char SEP = '\\';
#endif

double ct(int64 t)
{
    return double(t) / cv::getTickFrequency();
}


//
// 3 state online face db
//
enum
{
    NEUTRAL = 0,
    CAPTURE = 1,
    PREDICT = 2
};

//
// default face size, [80..160]
//
enum
{
    FIXED_FACE = 90
};



class FaceRec
{
    Preprocessor pre;

    Ptr<TextureFeature::Extractor>  extractor;
    Ptr<TextureFeature::Classifier> classifier;

    map<int,String> persons;

public:
    FaceRec()
        : pre(3,0,FIXED_FACE)
        , extractor(createExtractorFPLbp())
        , classifier(createClassifierHist(HISTCMP_HELLINGER))
        //, classifier(createClassifierPCA_LDA())
    {}

    int train(const String &imgdir)
    {
        string last_n("");
        int label(-1);

        Mat features;
        Mat labels;

        vector<String> vec;
        glob(imgdir,vec,true);
        if ( vec.empty())
            return 0;

        for (size_t i=0; i<vec.size(); i++)
        {
            String v = vec[i];
            int r1 = v.find_last_of(SEP);
            String v2 = v.substr(0,r1);
            int r2 = v2.find_last_of(SEP);
            String n = v2.substr(r2+1);
            if (n!=last_n)
            {
                last_n=n;
                label++;
            }
            persons[label] = n;
            Mat img=imread(vec[i],0);
            img = pre.process(img);

            Mat feature;
            extractor->extract(img,feature);
            features.push_back(feature.reshape(1,1));
            labels.push_back(label);
        }
        return classifier->train(features,labels);
    }

    String predict(const Mat & img)
    {
        Size sz(FIXED_FACE,FIXED_FACE);
        Mat im2;
        if (img.size() != sz)
            resize(img,im2,sz);
        else im2 = img;

        Mat feature;
        extractor->extract(pre.process(im2), feature);

        Mat_<float> result;
        classifier->predict(feature, result);
        int id = int(result(0));
        if (id < 0)
            return "";

        // unfortunately, some classifiers do not return a proper distance value.
        float conf = 1.0f - result(1);
        return format("%s : %2.3f", persons[id].c_str(), conf);
    }
};

int main(int argc, const char *argv[])
{
    theRNG().state = getTickCount();

    string cp("0");
    if ( argc>1 ) cp=argv[1];

    string imgpath("persons");
    if ( argc>2 ) imgpath=argv[2];

    std::string cascade_path("E:\\code\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
    if ( argc>3 ) cascade_path=argv[3];

    if ( argc==1 )
    {
        cerr << "please use : " << endl << argv[0] << " [capture id or path] [img_path] [cascade_path]" << endl;
        cerr << argv[0] << " " << cp << " " << imgpath <<  " " << cascade_path << endl << endl;
    }
    cerr << "press 'c' to record new persons," << endl;
    cerr << "      space, to stop recording. (then input a name)." << endl;
    cerr << "      'p' to predict," << endl;
    cerr << "      'n' for neutral," << endl;
    cerr << "      esc to quit." << endl;

    namedWindow("reco");

    // read haarcascade
    cv::CascadeClassifier cascade;
    bool clod = cascade.load(cascade_path);
    cerr << "cascade: " << clod  << endl;;


    VideoCapture cap;
    if ( cp=="0" ) cap.open(0);
    else if ( cp=="1" ) cap.open(1);
    else cap.open(cp);
    cerr << "capture(" << cp << ") : " << cap.isOpened() << endl;


    FaceRec reco;
    reco.train(imgpath);

    vector<Mat> images;
    String caption = "";
    int showCaption = 0;
    int frameNo = 0;
    int state = NEUTRAL;
    Scalar color;
    while(cap.isOpened() && clod)
    {
        Mat frame;
        cap >> frame;
        if ( frame.empty() )
            break;

        if ( state == PREDICT )
            color=Scalar(10,160,10);
        else if ( state == CAPTURE )
            color=Scalar(10,10,160);
        else
            color=Scalar(130,30,30);

        if ( state == PREDICT || state == CAPTURE )
        {
            Mat gray;
            cvtColor(frame,gray,COLOR_RGB2GRAY);

            std::vector<cv::Rect> faces;
            cascade.detectMultiScale( gray, faces, 1.2, 3,
                CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH  ,
                Size(40, 40),Size(300,300) );

            if ( faces.size() )
            {
                Rect roi = faces[0];
                if ( state == PREDICT )
                {
                    caption = reco.predict(gray(roi));
                    //cerr << "predict: " << caption << endl;
                    if (!caption.empty()) showCaption = 20;
                }
                if ( state == CAPTURE )
                {
                    if ( frameNo % 3 == 0 )
                    {
                        Mat m;
                        resize(gray(roi), m, Size(FIXED_FACE,FIXED_FACE));
                        images.push_back( m );
                        cerr << ".";
                    }
                }
                else
                {
                    if ( caption != "" && showCaption>0)
                    {
                        putText(frame,caption,Point(roi.x,roi.y+roi.width+13),FONT_HERSHEY_PLAIN,1.1,color,2);
                        showCaption--;
                    }
                }
                rectangle(frame, roi,color);
            }
        }
        circle(frame, Point(10,10),6, color*0.6, -1, LINE_AA);
        circle(frame, Point(10,10),5, color*0.8, -1, LINE_AA);
        circle(frame, Point(10,10),3, color*1.2, -1, LINE_AA);
        imshow("reco",frame);
        int k = waitKey(30);
        if ( k==27 ) break;
        if ( k=='p' ) state=PREDICT;
        if (( k=='n' ) || (k == ' '))
        {
            if ( (state == CAPTURE) && (!images.empty()) )
            {
                cerr << endl << "please enter a name(empty to abort) :" << endl;
                char n[200];
                gets(n);
                if (n[0]!=0 && images.size()>0)
                {
                    String path = imgpath;
                    path += SEP;
                    path += n;
                    String cmdline = "mkdir ";
                    cmdline += path;
                    system(cmdline.c_str());
                    for (size_t i=0; i<images.size(); i++)
                    {
                        imwrite(format("%s/%6d.png",path.c_str(), theRNG().next()), images[i]);
                    }
                    reco.train(imgpath);
                }
            }
            state=NEUTRAL;
        }
        if ( k=='c' )
        {
            if ( state != CAPTURE )
            {
                state=CAPTURE;
                images.clear();
            }
        }
        frameNo++;
    }
    return 0;
}
