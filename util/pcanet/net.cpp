#include <opencv2/opencv.hpp>
using namespace cv;


#include <cstdio>
#include <iostream>
using namespace std;

#include "net.h"


namespace util
{
//
// matlab like helpers:
//
static cv::Mat bsxfun_times(cv::Mat &bhist, int numFilters)
{
    int row = bhist.rows;
    int col = bhist.cols;

    vector<float> sum(col,0);

    for (int i = 0; i<row; i++)
    {
        const float *pb = bhist.ptr<float>(i);
        for (int j = 0; j<col; j++)
        {
            sum[j] += pb[j];
        }
    }

    float p = pow(2.0f, numFilters);
    for (int i = 0; i<col; i++)
    {
        sum[i] = p / sum[i];
    }

    for (int i = 0; i<row; i++)
    {
        float *pb = bhist.ptr<float>(i);
        for (int j=0; j<col; j++)
        {
            pb[j] *= sum[j];
        }
    }

    return bhist;
}



static cv::Mat hist(const cv::Mat &mat, int range)
{
    cv::Mat mt = mat.t();
    cv::Mat hist = cv::Mat::zeros(mt.rows, range + 1, CV_32F);

    for (int i=0; i<mt.rows; i++)
    {
        const float *m = mt.ptr<float>(i);
        float *h = hist.ptr<float>(i);

        for (int j=0; j<mt.cols; j++)
        {
            h[int(m[j])] += 1;
        }
    }

    return hist.t();
}


static cv::Mat im2col(const cv::Mat &images, const vector<int> &blockSize, const vector<int> &stepSize)
{
    const int ROW_DIM = 0;
    const int COL_DIM = 1;
    int row_diff = images.rows - blockSize[ROW_DIM];
    int col_diff = images.cols - blockSize[COL_DIM];
    int r_row = blockSize[ROW_DIM] * blockSize[COL_DIM];
    int r_col = (row_diff / stepSize[ROW_DIM] + 1) * (col_diff / stepSize[COL_DIM] + 1);
    cv::Mat outBlocks(r_col, r_row, images.type());

    int blocknum = 0;
    for (int j=0; j<=col_diff; j+=stepSize[COL_DIM])
    {
        for (int i=0; i<=row_diff; i+=stepSize[ROW_DIM])
        {
            float *p_out = outBlocks.ptr<float>(blocknum);

            for (int m=0; m<blockSize[ROW_DIM]; m++)
            {
                const float *p_in = images.ptr<float>(i + m);

                for (int l=0; l<blockSize[COL_DIM]; l++)
                {
                    p_out[blockSize[ROW_DIM] * l + m] = p_in[j + l];
                }
            }
            blocknum++;
        }
    }

    return outBlocks.t();
}


static cv::Mat patchImage(const cv::Mat &image, int patchSize, bool reduceMean=false)
{
    vector<int> blockSize(2, patchSize);
    vector<int> stepSize(2, 1);
    cv::Mat temp = im2col(image, blockSize, stepSize);
    if (! reduceMean)
        return temp;

    cv::Mat mean;
    cv::reduce(temp, mean, 0, cv::REDUCE_AVG);
    cv::Mat res;
    for (int i=0; i<temp.rows; i++)
    {
        cv::Mat temp2 = (temp.row(i) - mean.row(0));
        res.push_back(temp2.row(0));
    }
    return res;
}


static void randomIndex(cv::Mat_<int> &randIdx)
{
    for (size_t i=0; i<randIdx.total(); i++)
        randIdx(i) = i;
    cv::randShuffle(randIdx);
}


static void randomFat(const vector<cv::Mat> &input, cv::Mat &fat)
{
    cv::Mat_<int> randIdx(1, input.size());
    randomIndex(randIdx);

    for (size_t i=0; i<input.size(); i++)
    {
        int idx = randIdx(i);
        fat.push_back(input[idx].reshape(1,1));
    }
}

} // namespace util




//
// internal interface type:
//
struct Stage
{
    virtual bool process(const vector<Mat> &input, vector<Mat> &output) const = 0;
    virtual bool train(const vector<Mat> &input) { return false; }

    virtual bool save(FileStorage &fs) const { return false; }
    virtual bool load(const FileNode &fn) { return false; }

    virtual String type() const = 0;
    virtual String info() const { return type(); }
};



//
// base impl class already holds 100% of the testing code,
//   (training is left to subclasses)
//
struct FilterBank : Stage
{
    int patchSize, numFilters;
    Mat filters; // holds all filters, row-aligned.

    FilterBank() {}
    FilterBank(int patchSize, int numFilters)
        : patchSize(patchSize), numFilters(numFilters)
    {}


    inline
    cv::Mat filter(int f) const
    {
        return filters.row(f).reshape(1, patchSize);
    }


    Mat correlate(const Mat &src, const Mat &f, bool full_convolution=false) const
    {
        Point anchor(-1, -1);
        Mat dst, fil;
        if (full_convolution)
        {
            cv::flip(f, fil, -1);
            anchor = Point(f.cols - f.cols/2 - 1, f.rows - f.rows/2 - 1);
        }
        else
        {
            fil = f;
        }
        filter2D(src, dst, CV_32F, fil, anchor);
        cv::normalize(dst, dst, 1); // is this the general case ?
        return dst;
    }


    virtual bool process(const vector<Mat> &input, vector<Mat> &output) const
    {
        for (size_t i=0; i<input.size(); i++)
        {
            for (int j=0; j<numFilters; j++)
            {
                Mat o = correlate(input[i], filter(j), false);
                output.push_back(o);
            }
        }
        return true;
    }


    //
    // serialize & back
    //
    bool save(FileStorage &fs) const
    {
        fs << "PatchSize"  << patchSize;
        fs << "NumFilters" << numFilters;
        fs << "Filter"     << filters;
        return true;
    }
    bool load(const FileNode &fn)
    {
        fn["PatchSize"]    >> patchSize;
        fn["NumFilters"]   >> numFilters;
        fn["Filter"]       >> filters;
        return true;
    }


    //
    // visualization code.
    //
    inline
    void append(Mat &a, const Mat &b) const
    {
        if (a.empty()) a=b; else hconcat(a,b,a);
    }
    void filterVisAdd(const cv::Mat &fil, cv::Mat &res) const
    {
        cv::Mat r;//=fil;
        normalize(fil,r,255,0);
        r.convertTo(r,CV_8U,2,24);
        cv::Mat rb;
        cv::copyMakeBorder(r, rb,1,1,1,1, cv::BORDER_CONSTANT);
        append(res,rb);
    }
    void filterVis(cv::Mat &draw) const
    {
        cv::Mat res;
        for (int j=0; j<filters.rows; j++)
        {
            filterVisAdd(filters.row(j).clone().reshape(1, patchSize), res);
        }
        resize(res, draw, draw.size(), 0,0, INTER_NEAREST);
    }
    void filterVis(cv::String win) const
    {
        Mat draw(32,32*8,CV_8U,Scalar(0));
        filterVis(draw);
        imshow(win,draw);
        waitKey(1);
    }
    void filterVis(cv::String win, const vector<cv::Mat> &_filters) const
    {
        Mat draw(32,32*8,CV_8U,Scalar(0));
        cv::Mat res;
        for (size_t j=0; j<_filters.size(); j++)
        {
            filterVisAdd(_filters[j], res);
        }
        resize(res, draw, draw.size(), 0,0, INTER_NEAREST);
        imshow(win,draw);
        waitKey(1);
    }


    virtual String type() const { return "FilterBank"; }
    virtual String info() const { return type() + format("[%d,%d]", patchSize, numFilters); }
};



struct Learner : FilterBank
{
    int ngens;

    Learner()
        : ngens(1800)
    {}
    Learner(int patchSize, int numFilters, int ngens=1800)
        : FilterBank(patchSize, numFilters)
        , ngens(ngens)
    {}

    Mat normalize(const Mat &inp) const
    {
        cv::Scalar me,sd;
        cv::meanStdDev(inp, me, sd);
        cv::Mat im(inp - me[0]);
        im /= sd[0];
        return im;
    }

    virtual bool process(const vector<Mat> &input, vector<Mat> &output) const
    {
        for (size_t i=0; i<input.size(); i++)
        {
            Mat inp = normalize(input[i]);
            for (int j=0; j<numFilters; j++)
            {
                Mat o = correlate(inp, filter(j), false);
                output.push_back(o);
            }
        }
        return true;
    }

    virtual bool train(const vector<Mat> &images)
    {
        Mat grads(numFilters, patchSize*patchSize, CV_32F, 0.0f);
        filters = cv::Mat(numFilters, patchSize*patchSize, CV_32F);
        randu(filters,-1,1);
        namedWindow("fil",0);
        namedWindow("grad",0);
        for (int g=0; g<ngens; g++)
        {
            // random sample:
            int idx = theRNG().uniform(0, images.size());
            Mat im = normalize(images[idx]);
            //  reconstruct:
            Mat recon(im.size(), CV_32F, 0.0f);
            for (int i=0; i<numFilters; ++i)
            {
                Mat r = correlate(im, filter(i), true); // forward
                r = correlate(r, filter(i), true);       // backward
                accumulate(r, recon);
            }
            recon /= double(filters.rows);
            // update grads and filters:
            Mat residual = im - recon;
            cv::normalize(residual, residual);
            resize(residual,residual,Size(2*patchSize,2*patchSize));
            Mat vfil,vgrad;
            for (int f=0; f<filters.rows; ++f)
            {
                Mat &g = grads.row(f);
                g -= 0.095 * correlate(filter(f), residual, false).reshape(1,1);
                filters.row(f) += g * 0.0025f;
                append(vfil, filter(f));
                append(vgrad, g.reshape(1,patchSize));
            }
            imshow("fil",vfil);
            imshow("grad",vgrad);
            waitKey(5);
            cerr << "gen " << g << '\r';
        }
        return false;
    }
    virtual String type() const { return "Learner"; }
    virtual String info() const { return type() + format("[%d,%d,%d]", patchSize, numFilters, ngens); }
};



struct Oszillator : FilterBank
{
    float freq;

    Oszillator() {}
    Oszillator(int patchSize, int numFilters, float freq)
        : FilterBank(patchSize, numFilters)
        , freq(freq)
    {}
    bool save(FileStorage &fs) const
    {
        fs << "Freq" << freq;
        return FilterBank::save(fs);
    }
    bool load(const FileNode &fn)
    {
        fn["Freq"]   >> freq;
        return FilterBank::load(fn);
    }
    virtual String info() const { return type() + format("[%d,%d,%2.2f]", patchSize, numFilters, freq); }
};



struct GaborProjection : Oszillator
{
    float invert;

    GaborProjection() {}
    GaborProjection(int patchSize, int numFilters, float freq=2.1f, float invert=-1.0f)
        : Oszillator(patchSize, numFilters, freq)
        , invert(invert)
    {}

    virtual bool train(const vector<Mat> &images)
    {
        int N = numFilters;
        cv::Mat proj;
        for (int i=0; i<N; i++)
        {
            double sigma  = 12.0;;
            double theta  = double(freq*(7-i)) * CV_PI;
            double lambda = double(1.3*(10-i)) * CV_PI/4;// * 180.0;//180.0 - theta;//45.0;// * 180.0 / CV_PI;
            double gamma  = (i+5);//6.0;// * 180.0 / CV_PI;
            double psi = CV_PI;//90.0;
            cv::Mat gab = cv::getGaborKernel(cv::Size(patchSize,patchSize),sigma,theta,lambda,gamma,psi);
            gab.convertTo(gab,CV_32F,invert);
            proj.push_back(gab.reshape(1,1));
        }
        filters = proj;
        return true;
    }
    virtual String type() const { return "Gabor"; }
};



struct WaveProjection : Oszillator
{
    WaveProjection() {}
    WaveProjection(int patchSize, int numFilters,float freq=1.5f)
        : Oszillator(patchSize, numFilters, freq)
    {}
    virtual bool train(const vector<Mat> &images)
    {
        int N = numFilters, K = patchSize*patchSize;
        cv::Mat proj(N, K, CV_32F);

        float t = freq * 0.17f;
        float off = 1.0f + 1.0f/float(freq+1);
        for (int i=0; i<N; i++)
        {
            float *fp = proj.ptr<float>(i);
            for (int j=0; j<K; j++)
            {
                *fp++ = sin(off + (t*float(j+3)*float(i+3)));
            }
        }
        filters = proj;
        return true;
    }
    virtual String type() const { return "Wave"; }
};



//
// 2 of those, followed by a Hashing stage, and you got PCANet.
//
//paper: http://arxiv.org/pdf/1404.3606v2.pdf
//original code from: https://github.com/ldpe2g/pcanet
//
struct PcaProjection : FilterBank
{
    PcaProjection() {}
    PcaProjection(int patchSize, int numFilters)
        : FilterBank(patchSize, numFilters)
    {}

    virtual bool train(const vector<Mat> &images)
    {
        cv::Mat_<int> randIdx(1, images.size());
        util::randomIndex(randIdx);

        int size = patchSize * patchSize;
        cv::Mat Rx = cv::Mat::zeros(size, size, images[0].type());

        for (size_t j=0; j<images.size(); j++)
        {
            cv::Mat temp = util::patchImage(images[randIdx(j)], patchSize);
            Rx = Rx + temp * temp.t();
        }
        int count = images[0].cols * images.size();
        Rx = Rx / (double)(count);

        cv::Mat evals, evecs;
        eigen(Rx, evals, evecs);

        for (int i = 0; i<numFilters; i++)
        {
            filters.push_back(evecs.row(i));
        }
        return true;
    }

    virtual String type() const { return "Pca"; }
    virtual String info() const { return type() + format("[%d,%d]", patchSize, numFilters); }
};



//
// hash input vectors of recent stage to a single feature
//
struct Hashing : Stage
{
    int numFilters; // of prev. stage
    int histBlockSize; // no overlap for now
    vector<int> blockSize;

    Hashing() {}
    Hashing(int numFilters, int histBlockSize)
        : numFilters(numFilters)
        , histBlockSize(histBlockSize)
        , blockSize(2, histBlockSize)
    {}

    virtual bool process(const vector<Mat> &input, vector<Mat> &output) const
    {
        cv::Mat bhist;
        int numImg = input.size() / numFilters;
        for (int i=0; i<numImg; i++)
        {
            cv::Mat T(input[0].size(), input[0].type(), cv::Scalar(0));
            for (int j=0; j<numFilters; j++)
            {
                cv::Mat temp;
                threshold(input[numFilters * i + j], temp, 0.0, double(1<<j), 0);
                T += temp;
            }
            cv::Mat t2 = util::im2col(T, blockSize, blockSize);
            t2 = util::hist(t2, (int)(pow(2.0, numFilters)) - 1);
            t2 = util::bsxfun_times(t2, numFilters);
            if (i == 0) bhist = t2;
            else hconcat(bhist, t2, bhist);
        }
        if (!bhist.empty())
        {
            output.push_back(bhist.reshape(1,1));
            return true;
        }
        output.push_back(input[0]);
        return false;
    }

    bool save(FileStorage &fs) const
    {
        fs << "NumFilters" << numFilters;
        fs << "BlockSize"  << histBlockSize;
        return true;
    }

    bool load(const FileNode &fn)
    {
        fn["NumFilters"]  >> numFilters;
        fn["BlockSize"]   >> histBlockSize;
        blockSize = std::vector<int>(2,histBlockSize);
        return true;
    }

    virtual String type() const { return "Hashing"; }
    virtual String info() const { return type() + format("[%d,%d]",  numFilters, histBlockSize); }
};



struct Network : public PNet
{
    vector< Ptr<Stage> > layers;

    int addStage(Ptr<Stage> s)
    {
        layers.push_back(s);
        return layers.size();
    }

    //
    // used to deserialize (e.g. from file),
    // basically anything is a filterbank,
    //    (additionally params are only saved for later reconstruction)
    //
    Ptr<Stage> addStage(const String &name)
    {
        Ptr<Stage> s;
        if (name=="FilterBank")  s = makePtr<FilterBank>();
        if (name=="Pca")  s = makePtr<PcaProjection>();
        if (name=="Wave")  s = makePtr<WaveProjection>();
        if (name=="Gabor")  s = makePtr<GaborProjection>();
        if (name=="Learner")  s = makePtr<Learner>();
        if (name=="Hashing") s = makePtr<Hashing>();

        CV_Assert(! s.empty());
        addStage(s);
        return s;
    }

    bool train(const vector<Mat> &images)
    {
        vector<cv::Mat>feat(images), post;

        for (size_t i=0; i<layers.size() - 1; i++)
        {
            // train & propagate to next stage
            layers[i]->train(feat);
            layers[i]->process(feat, post);

            //dbg/vis output:
            stageViz(post,layers[i]->type() + format("_%d",i), 10);
            cerr << "\t" << i << "\t" << layers[i]->type() << "\t" << feat.size() << "\t" << post.size() << endl;

            // swap prev/cur set
            feat.clear();
            cv::swap(feat,post);
        }
        return true;
    }

    virtual Mat extract(const Mat &img) const
    {
        Mat im;
        img.convertTo(im,CV_32F);
        vector<cv::Mat>feat(1,im), post;

        for (size_t i=0; i<layers.size(); i++)
        {
            layers[i]->process(feat, post);
            // stageViz(post,layers[i]->type() + "_post", 100);
            // cerr << "\t" << i << "\t" << layers[i]->type() << "\t" << feat.size() << "\t" << post.size() << endl;
            feat.clear();
            cv::swap(feat,post);
        }
        return feat.back();
    }

    bool save(const cv::String &fn) const
    {
        cv::FileStorage fs(fn, cv::FileStorage::WRITE);
        fs << "Stages" << "[" ;
        for (size_t i=0; i<layers.size(); i++)
        {
            fs << "{:" ;
            fs << "Type" << layers[i]->type();
            layers[i]->save(fs);
            fs << "}";
        }
        fs << "]";
        fs.release();
        return true;
    }

    virtual bool load(const cv::String &fn)
    {
        cv::FileStorage fs(fn, cv::FileStorage::READ);
        cv::FileNode no = fs["Stages"];
        for (cv::FileNodeIterator it=no.begin(); it!=no.end(); ++it)
        {
            const cv::FileNode &n = *it;
            cv::String type;
            n["Type"] >> type;
            cv::Ptr<Stage> stage = addStage(type);
            stage->load(n);
        }
        fs.release();
        return true;
    }

    cv::Mat filterVis() const
    {
        int nStages = layers.size() - 1;
        cv::Mat draw(400,400,CV_8U,Scalar(0));
        int step = draw.rows / nStages;
        for (int i=0; i<nStages; i++)
        {
            Rect r(0, i*step, draw.cols, step);
            Ptr<FilterBank> fb = layers[i].dynamicCast<FilterBank>();
            if (!fb.empty())
                fb->filterVis(draw(r));
        }
        return draw;
    }

    void prepVis(const cv::Mat &img, cv::Mat &res) const
    {
        cv::Mat r;
        cv::resize(img,r,Size(50,50));
        cv::normalize(r,r,255,0);
        r.convertTo(r,CV_8U,3,32);
        cv::Mat rb;
        cv::copyMakeBorder(r,rb,1,1,1,1,cv::BORDER_CONSTANT);
        if (res.empty()) res=rb;
        else cv::hconcat(res,rb,res);
    }

    void imgVis(const vector<cv::Mat> &img, cv::Mat &draw, int n=8) const
    {
        cv::Mat res;
        n = (n < int(img.size())) ? n : int(img.size());
        for (int j=0; j<n; j++)
        {
            prepVis(img[j],res);
        }
        res.copyTo(draw);
    }

    void stageViz(const vector<cv::Mat> &img, const cv::String &title, int delay) const
    {
        cv::Mat draw(32,15*32,CV_8U,cv::Scalar(0));
        imgVis(img,draw,15);
        imshow(title,draw);
        waitKey(delay);
    }

    cv::String ingo()
    {
        cv::String s="";
        for (size_t j=0; j<layers.size(); j++)
        {
            s += layers[j]->info() + "\r\n";
        }
        return s;
    }
};



//
// global factory method
//
cv::Ptr<PNet> loadNet(const String &fn)
{
    cv::Ptr<Network> pn = makePtr<Network>();
    pn->load(fn);
    cerr << pn->ingo() << endl;
    return pn;
}





//
// -------------8<------------------------------------------------------------------------------
//
#ifdef TRAIN_STANDALONE

int main()
{
    // String path("e:/code/vlc/faces/*.png");
    String path("e:/code/opencv_p/face3/data/funneled/");
    vector<String> fn;
    glob(path,fn);

    vector<Mat> images;
    for (size_t i=0; i<600; ++i)
    {
        size_t idx = theRNG().uniform(0,fn.size());
        Mat im = imread(fn[idx],0);
        im.convertTo(im,CV_32F);
        images.push_back(im);
    }

    namedWindow("filters", 0);
    Network net;
    int nFilters=7;
    if (1)
    {
        cerr << "train " << images.size() << endl;
        net.addStage(makePtr<PcaProjection>(7, nFilters));
        // net.addStage(makePtr<GaborProjection>(9, 5, 0.373f, -1.0f)); // gabor kernels need to be odd
        //net.addStage(makePtr<Learner>(11, 6));
        net.addStage(makePtr<Learner>(11, nFilters));
        // net.addStage(makePtr<Learner>(7, 4));
        // net.addStage(makePtr<GaborProjection>(9, 5, 0.373f, -1.0f)); // gabor kernels need to be odd
        net.addStage(makePtr<Hashing>(nFilters, 18));
        net.train(images);
        net.save("my.xml");
    }
    else
    {
        net.load("my.xml");
    }
    cerr << net.ingo() << endl;

    Mat v = net.filterVis();
    imshow("filters", v);
    imwrite("filters.png",v);
    int Z=23;
    cerr << "test" << endl;
    Mat im = imread(fn[Z],0);
    Mat res = net.extract(im);
    imshow(String("0 ")+fn[Z], im); cerr << fn[Z] << endl;
    Z=41;
    im = imread(fn[Z],0);
    Mat res1 = net.extract(im);
    imshow(String("1 ")+fn[Z], im); cerr << fn[Z] << endl;
    Z=42;
    im = imread(fn[Z],0);
    Mat res2 = net.extract(im);
    imshow(String("2 ")+fn[Z], im); cerr << fn[Z] << endl;

    cerr << "d 02 " << norm(res,res2) << endl;
    cerr << "d 01 " << norm(res,res1) << endl;
    cerr << "d 12 " << norm(res1,res2) << endl;
    cerr << "d xx " << 1.0 - norm(res1,res2)/norm(res,res2) << endl;

    waitKey();
    return 0;
}

#endif // TRAIN_STANDALONE
