#ifndef __profile_onboard__
#define __profile_onboard__

//#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
using namespace cv;
#include <iostream>
using namespace std;

#define get_ticks cv::getTickCount
#define get_freq  cv::getTickFrequency

struct Profile
{
    double dt(int64 t) { return double(t*1000/freq)/1000.0; }
    static double freq;
    cv::String name;
    int64 t; // accumulated time
    int64 c; // function calls
    double d_tc;
    double d_t;

    Profile(cv::String name) 
        : name(name)
        , t(0) 
        , c(0)
        , d_tc(0)
        , d_t(0)
    {}   
    ~Profile() 
    {
        fprintf(stderr, "%-24s %8u ",name.c_str(),c);
        fprintf(stderr, "%13.6f ",d_tc); 
        fprintf(stderr, "%13.6f ",d_t);
        fprintf(stderr, "%14u",t);
        fprintf(stderr, "\n");
    }

    void tick(int64 delta)
    {
        if (delta <= 0)  return;
        t += delta;
        c ++;
        d_t  = dt(delta);
        d_tc = d_t/c;
    }


    struct Scope
    {
        Profile & p;
        int64 t;

        Scope(Profile & p) 
            : p(p) 
            , t(get_ticks()) 
        {}

        ~Scope() 
        { 
            p.tick(get_ticks() - t);
        }
    }; 
};
double Profile::freq = get_freq();

#define PROFILEX(s) static Profile _a_rose(s); Profile::Scope _is_a_rose_is(_a_rose);
#define PROFILE PROFILEX(__FUNCTION__)


#endif // __profile_onboard__

