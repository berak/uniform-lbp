//
// 3 state online face db
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/ml/ml.hpp>
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;
using namespace cv;

#include "factory.h"


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
    "combined",
    "norml2"
};

//
// online 0 att.txt 5 400 1
// online cap db reco nimages verbose
//
//
enum {
    NEUTRAL = 0,
    CAPTURE = 1,
    PREDICT = 2
};

int findName(const map<string,int> & persons, const string & n)
{
    map<string,int>::const_iterator it=persons.find(n);
    if ( it != persons.end() )
    {
        return it->second;
    }
    return -1;
}
string findId(const map<string,int> & persons, int id)
{
    map<string,int>::const_iterator it = persons.begin();
    for ( ; it != persons.end(); ++it )
    {
        if (id == it->second )
            return it->first;
    }
    return "";
}
void dump(const map<string,int> & persons)
{
    map<string,int>::const_iterator it = persons.begin();
    for ( ; it != persons.end(); ++it )
    {
        cerr << it->first << ":" << it->second << " ";
    }
    cerr << endl;
}

Mat preprocess( const Mat & mm )
{
    Mat m2,m3;
    resize(mm,m2,Size(90,90));
    cv::equalizeHist( m2, m3 );
    return m3;
}

int main(int argc, const char *argv[]) 
{
    vector<Mat> images;
    vector<int> labels;
    map<string,int> persons;
    vector<string> vec;
    int state = NEUTRAL;

    string cp("0");
    if ( argc>1 ) cp=argv[1];

    string imgpath("-");
    if ( argc>2 ) imgpath=argv[2];

    int rec = 5;
    if ( argc>3 ) rec=atoi(argv[3]);

    std::string cascade_path("E:\\code\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
    if ( argc>4 ) cascade_path=argv[4];

    if ( argc==1 )
    {
        cerr << argv[0] << " " << imgpath <<  " " << cascade_path << endl;
    }

    namedWindow("reco");

    // read haarcascade
	cv::CascadeClassifier cascade;
    bool clod = cascade.load(cascade_path);
    cerr << clod  << " cascade, " ;



    Ptr<FaceRecognizer> model;
    if ( rec == 0 ) {
        model = createFisherFaceRecognizer();
    } else if ( rec == 1 ) {
        model = createEigenFaceRecognizer();
    } else if ( rec == 2 ) {
        model = createLBPHFaceRecognizer(1,8,8,8,DBL_MAX);
    } else if ( rec == 3 ) {
        model = createLBPHFaceRecognizer2(1,8,8,8,DBL_MAX,true);
    } else if ( rec == 4 ) {
        model = createMinMaxFaceRecognizer(10,4);
    } else if ( rec == 5 ) {
        model = createCombinedLBPHFaceRecognizer(8,8,DBL_MAX);
    } else {
        model = createLinearFaceRecognizer(NORM_L2);
    }

    // read previous state 
    FileStorage fs("zz13.yml",FileStorage::READ);
    if ( fs.isOpened() )
    {
        model->load(fs);
        FileNode pers = fs["persons"];
        FileNodeIterator it = pers.begin();
        for( ; it != pers.end(); ++it )
        {
            string s = (*it).name();

            int id;
            (*it) >> id;

            persons[s] = id;
        }
        fs.release();
    }
    // read face db
    size_t maxim = 400;
    int nsubjects = -1;
    if ( imgpath != "-" )
        nsubjects = 1 + readtxt(imgpath.c_str(),vec,labels, maxim);

    cerr << nsubjects  << " subjects, " ;
    // read the images, correct labels if empty image skipped
    int skipped = 0;
    vector<int> clabels;
    for ( size_t i=0; i<vec.size(); i++ )
    {
        Mat mm = imread(vec[i],0);
        if ( mm.empty() )
        {
            skipped ++;
            continue;
        }
        mm = preprocess(mm);
        images.push_back(mm);
        clabels.push_back(labels[i]);
        string n(vec[i]);
        int e = n.find_last_of("\\") - 1;
        int s = n.find_last_of("\\",e );
        string name(n.substr(s+1,e-s));
        persons[name] = labels[i];
    }
    labels = clabels;
    cerr << images.size() << " images ";
    if ( ! images.empty() )
    {
        cerr << ", " << images.size()/nsubjects << " per person ";
        if ( skipped ) cerr << "(" << skipped << " images skipped)";
        cerr << endl;

        model->train(images, labels);
        cerr << "trained " << rec_names[rec];
    }
    cerr << endl;

    VideoCapture cap;
    if ( cp=="0" ) cap.open(0);
    else if ( cp=="1" ) cap.open(1);
    else cap.open(cp);
    cerr << "capture " << cp << " : " << cap.isOpened() << endl;
    //cap.set(CAP_PROP_SETTINGS,1);
    int frameNo = 0;
    while(cap.isOpened())
    {
        Mat frame;
        cap >> frame;
        if ( frame.empty() ) 
            break;
        if ( state == PREDICT || state == CAPTURE )
        {
            Mat gray;
            cvtColor(frame,gray,COLOR_RGB2GRAY);
        	while ( gray.rows > 512 )
            {
                pyrDown(gray,gray);
            }

            std::vector<cv::Rect> faces;
    	    cascade.detectMultiScale( gray, faces, 1.2, 3, 
		        CASCADE_FIND_BIGGEST_OBJECT	| CASCADE_DO_ROUGH_SEARCH  ,
    		    Size(20, 20) );

	        if ( faces.size() )
	        {
		        Rect roi = faces[0];
                string n = "";
                if ( state == PREDICT )
                {
                    double dist = 0;
                    int predicted = -1;
                    try {
                        Mat m = gray(roi);
                        m = preprocess(m);
                        model->predict( m,predicted,dist );
                    } catch(const Exception & e ) {
                        cerr << e.what() << endl;
                        state = NEUTRAL; 
                    }
                    if ( predicted != -1 )
                    {
                        n = findId(persons,predicted);
                    }
                    cerr << "predict " << predicted << " " << dist << " " << n << endl;
                }
                if ( state == CAPTURE )
                {
                    if ( frameNo % 3 == 0 )
                    {
                        Mat m = gray(roi);
                        m = preprocess(m);
                        images.push_back( m );
                        //labels.push_back( cap_id );
                        cerr << ".";
                        frame = m;
                    }
                }
                else
                {
                    rectangle(frame, roi,Scalar(0,200,0));
                    if ( n != "" )
                    {
                        putText(frame,n,Point(roi.x,roi.y+roi.width+13),FONT_HERSHEY_PLAIN,1.1,Scalar(60,160,0),2);
                    }
                }
	        }
        }

        imshow("reco",frame);
        int k = waitKey(30);
        if ( k==27 ) break;
        if ( k=='d' ) { dump(persons); };
        if ( k=='p' ) { state=PREDICT; cerr << "state " << state << endl; };
        if (( k=='n' ) || (k == ' ')) 
        { 
            if ( (state == CAPTURE) && (!images.empty()) )
            {
                cerr << endl << "please enter a name(empty to abort)" << endl;
                char n[200];
                gets(n);
                if ( n[0] != 0 )
                {
                    int cap_id = persons.size() + 1;
                    if ( persons.find(n) != persons.end() )
                    {
                        cap_id = persons[n];
                    }
                    persons[n] = cap_id;
                    labels = vector<int>(images.size(),cap_id);
                    model->update(images,labels);
                    cerr << "state " << state << " added " << n << " " << cap_id <<  " : " << images.size() << " images." << endl; 
                    FileStorage fs("zz13.yml",FileStorage::WRITE);
                    model->save(fs);
                    fs << "persons" << "{";
                    map<string,int>::const_iterator it = persons.begin();
                    for ( ; it != persons.end(); ++it )
                    {
                        fs << it->first << it->second;
                    }
                    fs << "}";
                    fs.release();
                }
            }
            state=NEUTRAL; 
            cerr << "state " << state << endl; 
        }
        if ( k=='c' ) 
        { 
            if ( state != CAPTURE )
            {
                state=CAPTURE; 
                cerr << "state " << state << endl; 
                images.clear();
                labels.clear();
                //cap_id = persons.size() + 1;
            }
        }
        frameNo++;
    }
    return 0;
}
