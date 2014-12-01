
#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/datasets/fr_lfw.hpp"


#if 1
 #include "profile.h"
#else
 #define PROFILE ;
 #define PROFILEX(s) ;
#endif

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <set>

using namespace std;
using namespace cv;
using namespace cv::datasets;

map<string, int> people;

int getLabel(const string &imagePath)
{   
    size_t pos = imagePath.find('/');
    string curr = imagePath.substr(0, pos);
    map<string, int>::iterator it = people.find(curr);
    if (people.end() == it)
    {
        people.insert(make_pair(curr, (int)people.size()));
        it = people.find(curr);
    }
    return (*it).second;
}

RNG rng(getTickCount());


struct Entry
{
    Rect r;
    double eq, ne;
    int n;
    Entry() : eq(0),ne(1), n(1) 
    {
        r.x = rng.uniform(0,85);
        r.y = rng.uniform(0,85);
        r.width  = rng.uniform(4,8+(90-r.x)/4);
        r.height = rng.uniform(4,8+(90-r.y)/4);
        r &= Rect(0,0,90,90);
        //r.width  = rng.uniform(4,90-r.x);
        //r.height = rng.uniform(4,90-r.y);
    }
    void test(const Mat &a,const Mat &b, bool same)
    {
        double dist = norm(a(r), b(r), NORM_L2);
        if (same) eq += dist;
        else      ne += dist;
        n++;
    }
    inline double score() const
    {
        return (ne-eq)*(ne-eq) / (n*n);
    }
    inline void reset() 
    {
        eq = ne = n = 0;
    }
    String str() 
    {
        return format("%2d %2d %2d %2d %6d %9.2f %9.2f  %9.6f",r.x,r.y,r.width,r.height,r.area(),eq/n,ne/n, score()); 
    }
};

struct EntrySort
{
    bool operator () (const Entry &a, const Entry &b) const
    {
        return a.score() > b.score();
    }
};


int prune_dups(vector<Entry> &e, int clip=500)
{
    PROFILE;
    int pruned=0;
    for (vector<Entry>::iterator it=e.begin(); it!=e.end(); )
    {
        vector<Entry>::iterator hit = e.end();
        for (vector<Entry>::iterator et=e.begin(); et!=e.end(); et++ )
        {
            int inner = (it->r & et->r).area();
            int outer = (it->r | et->r).area();
            if ( outer-inner <= clip )
            {
                hit =et;
                break;
            }
        }
        if ( hit != e.end() && it->score() < hit->score() )
        {
            //cerr << "hit it " << it->str() << "\n  keep  " << hit->str() << endl;
            pruned++;
            it = e.erase(it);
        }
        else it++;
    }
    return pruned;
}

void prune(vector<Entry> &e)
{
    PROFILE;
    std::sort(e.begin(), e.end(), EntrySort());
    size_t n = e.size() / 2;
    e.resize(n);
    cerr << "pruned " << prune_dups(e) << endl;
    for (size_t i=0; i<e.size(); i++)
        e[i].reset();
    while (e.size() < 64)
        e.push_back(Entry());
}

int main(int argc, const char *argv[])
{   
    PROFILE;
    string path("lfw-deepfunneled/");

    vector<Entry> e;
    for (size_t i=0; i<64; i++)
    {
        Entry x;
        //cerr << x.str() << endl;
        e.push_back(x);
    }

    // load dataset
    Ptr<FR_lfw> dataset = FR_lfw::create();
    dataset->load(path);
    unsigned int numSplits = dataset->getNumSplits();

    double best=0;
    for (unsigned int k=0; k<512; ++k)
    {
        PROFILEX("generations");
        for (unsigned int j=0; j<numSplits; ++j)
        //int j = 0;
        {  
            PROFILEX("splits");
            //if ( j%3 == 0)
            //    prune(e);
            vector < Ptr<Object> > &curr = dataset->getTest(j);
            for (unsigned int i=0; i<curr.size(); ++i)
            {
                FR_lfwObj *example = static_cast<FR_lfwObj *>(curr[i].get());
                Mat img1 = imread(path+example->image1, IMREAD_GRAYSCALE);
                Mat img2 = imread(path+example->image2, IMREAD_GRAYSCALE);
                Mat roi1(img1,Rect(80,80,90,90));
                Mat roi2(img2,Rect(80,80,90,90));
                for (size_t k=0; k<e.size(); k++)
                {
                    e[k].test( roi1,roi2, example->same );
                }
            }
        }
        cerr << "--------------"<<k<<"----------------" << endl;
        std::sort(e.begin(), e.end(), EntrySort());
        double s=0;
        for (size_t i=0; i<e.size()/2; i++)
        {
            cerr << e[i].str() << endl;
            double es = e[i].score();
            s += es;
        }
        double ak = ((s)/(e.size()/2));
        best = std::max(best, ak);
        cerr << "gen(" << k << ") avg score " << ak << "/" << best << endl;

        prune(e);
        vector < Ptr<Object> > &curr = dataset->getTest(k%numSplits);
        FR_lfwObj *example = static_cast<FR_lfwObj *>(curr[0].get());
        //Mat img1 = imread(path+example->image1);
        //for ( size_t i=0; i<32; i++ )
        //{
        //    Mat roi(img1,Rect(80,80,90,90));
        //    rectangle(roi,e[i].r,Scalar(rng.uniform(0,200),rng.uniform(0,200),rng.uniform(0,200)));
        //}
        //imshow("voila!",img1);
        //waitKey(1);
    }
    cerr << ";)" << endl;
    //waitKey();

    return 0;
}
