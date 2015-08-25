#include <opencv2/opencv.hpp>
using namespace cv;


#include <cstdio>
#include <iostream>
using namespace std;

#include "net.h"



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


struct FilterBank : Stage
{
    int patchSize, numFilters, threadnum;
    bool doFlip;
    Mat filters;

    FilterBank() : doFlip(true){}
    FilterBank(int patchSize, int numFilters, int threadnum=1, bool doFlip=true)
        : patchSize(patchSize), numFilters(numFilters), threadnum(threadnum), doFlip(doFlip)
    {}

    cv::Mat filter(int f) const
    {
        return filters.row(f).reshape(1, patchSize);
    }

    virtual bool process(const vector<Mat> &input, vector<Mat> &output) const
    {
        for (size_t i=0; i<input.size(); i++)
        {
            for (int j=0; j<numFilters; j++)
            {
                const cv::Mat &tf = filter(j);
                cv::Mat temp;
                filter2D(input[i], temp, CV_32F, tf);
                output.push_back(temp);
            }
        }
        return true;
    }

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

    void filterVisAdd(const cv::Mat &fil, cv::Mat &res) const
    {
        cv::Mat r;//=fil;
        normalize(fil,r,255,0);
        r.convertTo(r,CV_8U,2,24);
        cv::Mat rb;
        cv::copyMakeBorder(r,rb,1,1,1,1,cv::BORDER_CONSTANT);
        if (res.empty()) res=rb;
        else cv::hconcat(res,rb,res);
    }

    void filterVis(cv::Mat &draw) const
    {
        cv::Mat res;
        for (int j=0; j<filters.rows; j++)
        {
            filterVisAdd(filters.row(j).clone().reshape(1,patchSize), res);
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
        fs << "Freq"   << freq;
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
        int N = numFilters, K = patchSize*patchSize;
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
////
//// WIP ...
////
//
//
//
//static double meanval(vector<double> &v, double cur)
//{
//    v.pop_back();
//    v.insert(v.begin(),cur);
//    double s = cv::sum(cv::abs(cv::Mat(v)))[0];
//    return s / v.size();
//}
//static void correlate_valid(const Mat &src, const Mat &kernel, Mat &dst)
//{
//    //cv::Mat tmp_dst;
//    cv::filter2D(src, dst, -1, kernel);
//    // incr deals with even-sized filters
//    //unsigned int incr = (kernel.rows%2==0) ? 1 : 0;
//    //dst = tmp_dst(cv::Range(kernel.rows/2,src.cols-kernel.rows/2+incr),
//    //              cv::Range(kernel.rows/2,src.cols-kernel.rows/2+incr));
//}
//
//static void convolve_full(const cv::Mat& src,const cv::Mat& kernel,cv::Mat& dst)
//{
//    cv::Mat flipped_kernel(kernel.rows,kernel.cols,CV_32FC1);
//    cv::flip(kernel,flipped_kernel,-1);
//
//    cv::Mat framed_src(src.rows+2*(kernel.rows-1),2*(src.rows+kernel.rows-1),CV_32FC1);
//    cv::copyMakeBorder(src,framed_src,kernel.rows-1,kernel.rows-1,kernel.cols-1,kernel.cols-1,cv::BORDER_REFLECT101);
//
//    correlate_valid(framed_src,flipped_kernel,dst);
//}
//struct ConvLearn : FilterBank
//{
//
//    ConvLearn() {}
//    ConvLearn(int patchSize, int numFilters)
//        : FilterBank(patchSize, numFilters)
//    {}
//
//    virtual bool train(const vector<Mat> &images)
//    {
//        const int maxIterations = 3000;
//        const int numGDCoeffSteps = 8;
//        const float eta_coeffs  = 0.00133f;  // Step size for the gradient descent on the feature maps
//        const float eta_filters = 0.193f;   // Step size for the gradient descent on the filters
//        const float lambda_learn = 0.1f;    // Regularization parameter
//        const float learn_err = 5.0f;        // stop distance
//        const float xi_filters = 30000.0f;
//        bool penalize_similar_filters = true;
//        vector<double> _means(20); // debug
//
//        filters = cv::Mat(numFilters, patchSize*patchSize, CV_32F, 0.0f);
//        //randu(filters, -1.0f, 1.0f);
//        randu(filters, 0.0f, 1.0f);
//        cv::Mat fold = filters.clone();
//
//        // Create the vector of filter gradients that will be used throughout the optimization process
//        std::vector<cv::Mat> filters_gradient;
//        for (int i=0; i<numFilters; ++i)
//        {
//            cv::Mat tmp_gradient(patchSize,patchSize,CV_32FC1,cv::Scalar(0));
//            filters_gradient.push_back(tmp_gradient);
//        }
//
//        std::vector<cv::Mat> feature_maps(numFilters);
//        int iter=0;
//        for (; iter<maxIterations; ++iter)
//        {
//            cv::Mat sample = images[theRNG().uniform(0,images.size())];
//            cv::Range ry(patchSize, sample.rows - patchSize);
//            cv::Range rx(patchSize, sample.cols - patchSize);
//            cv::Mat sample_roi = sample(ry,rx);
//            cv::Mat bin;// = sample_roi *0.05;
//            normalize(sample_roi, bin, 25.0f);
//            // compute feature maps:
//            for (int f=0; f<numFilters; ++f)
//            {
//                filters_gradient[f] = 0;
//                correlate_valid(bin, filter(f), feature_maps[f]);
//            }
//            // Refine feature maps
//            for (int j=0; j<numGDCoeffSteps; ++j)
//            {
//                cv::Mat reconstruction = compute_reconstruction(feature_maps);
//                cv::Mat residual = bin - reconstruction;
//                cv::Mat fm_gradient;
//                for (int f=0; f<numFilters; ++f)
//                {
//                    convolve_full(residual, filter(f), fm_gradient);
//                    ISTA(feature_maps[f], fm_gradient, eta_coeffs, lambda_learn);
//                }
//            }
//
//            // compute filter gradients:
//            cv::Mat reconstruction = compute_reconstruction(feature_maps);
//            cv::Mat residual = bin - reconstruction;
//
//            cv::Mat gram_matrix;
//            if (penalize_similar_filters)
//            {
//                gram_matrix = cv::Mat(numFilters, numFilters, CV_32FC1, cv::Scalar(0));
//                for (int r=0; r<numFilters-1; ++r)
//                {
//                    float *Mr = gram_matrix.ptr< float >(r);
//                    for (int c=r+1; c<numFilters;++c)
//                    {
//                        float *Mc = gram_matrix.ptr< float >(c);
//                        Mr[c] = (float)filters.row(r).dot(filters.row(c));
//                        Mc[r] = Mr[c];
//                    }
//                }
//            }
//
//            //cerr << format("%4d : ",iter);
//            for (int f=0; f<numFilters; ++f)
//            {
//                cv::Mat &fm = feature_maps[f];
//                int x = patchSize + fm.cols/4 + theRNG().uniform(0, patchSize);
//                int y = patchSize + fm.rows/4 + theRNG().uniform(0, patchSize);
//                //int x = theRNG().uniform(patchSize, fm.cols-patchSize*2);
//                //int y = theRNG().uniform(patchSize, fm.rows-patchSize*2);
//
//                cv::Mat fm_roi = fm(cv::Range(x, x + patchSize), cv::Range(y, y + patchSize));
//                //cv::Mat fm_roi = fm(cv::Range(1, fm_size - 1), cv::Range(1, fm_size - 1));
//                cv::Mat grad;
//                correlate_valid(fm_roi, residual, grad);
//
//                if (penalize_similar_filters)
//                {
//                    cv::Mat fbprod_penalty(patchSize, patchSize, CV_32FC1, cv::Scalar(0));
//                    float *Mr = gram_matrix.ptr<float>(f);
//                    for (int c=0; c<numFilters; ++c)
//                    {
//                        if (c != f)
//                        {
//                            fbprod_penalty += Mr[c] * filter(c);
//                        }
//                    }
//                    grad -= fbprod_penalty * xi_filters;
//                    //cerr << format("%3.1f ",norm(fbprod_penalty * xi_filters));
//                }
//                normalize(grad, grad, 1, 0, NORM_L1);
//                filters_gradient[f] += grad * eta_filters;
//            }
//            //cerr << endl;
//
//            // update the filters:
//            for (int f=0; f<numFilters; ++f)
//            {
//                Mat &fil = filter(f);
//                fil += filters_gradient[f];
//                cv::normalize(fil,fil);
//            }
//            //cerr << endl;
//            double dist = sum(residual)[0];
//            double mv = meanval(_means, dist);
//            if (mv < learn_err)
//            {
//                cerr << iter << "final residual " << dist << "\tmean " << mv << endl;
//                break;
//            }
//            if (iter%_means.size() == 0)
//            {
//                double fd = norm(fold,filters);
//                cerr << iter << "\tresidual " << dist << "\tmean " << mv<< "\tfd " << fd << endl;
//            }
//
//            //store(iter);
//            imshow("bin",bin);
//            filterVis("fil");
//            filterVis("grad",filters_gradient);
//            filterVis("maps",feature_maps);
//        }
//        store(iter);
//        return true;
//    }
//
//    void store(int iter)
//    {
//        cv::FileStorage fs(cv::format("conv_%d.xml",iter), cv::FileStorage::WRITE);
//        FilterBank::save(fs);
//        fs.release();
//    }
//
//    void ISTA(cv::Mat &src, const cv::Mat &gradient, float GD_step, float reg_param) const
//    {
//        for (int r=0; r<src.rows; ++r)
//        {
//            float *M_src = src.ptr<float>(r);
//            const float *M_grad = gradient.ptr<float>(r);
//            for (int c=0; c<src.cols; ++c)
//            {
//                // x - \eta \grad(g)
//                M_src[c] += GD_step*M_grad[c];
//
//                // Thresholding step
//                if (M_src[c] >= reg_param)
//                {
//                    M_src[c] -= reg_param;
//                }
//                else
//                {
//                    if (M_src[c] <= -reg_param)
//                        M_src[c] += reg_param;
//                    else
//                        M_src[c] = 0;
//                }
//            }
//        }
//    }
//
    //cv::Mat compute_reconstruction(const vector<cv::Mat> &feature_maps) const
    //{
    //    cv::Mat reconstruction;//(sample_size, sample_size, CV_32FC1, cv::Scalar(0));
    //    cv::Mat tmp_reconstruction;
    //    //cv::Mat tmp_reconstruction_roi;
    //    for (int f=0; f<numFilters; ++f)
    //    {
    //        correlate_valid(feature_maps[f], filter(f), tmp_reconstruction);
    //        //tmp_reconstruction_roi = tmp_reconstruction(cv::Range(1, tmp_reconstruction.rows - 1),
    //        //                                            cv::Range(1, tmp_reconstruction.cols - 1));
    //        if (reconstruction.empty())
    //            reconstruction = tmp_reconstruction;
    //        else
    //            reconstruction += tmp_reconstruction;
    //    }
    //    reconstruction /= ((patchSize) * (patchSize));
    //    return(reconstruction);
    //}
//    virtual String type() const { return "Conv"; }
//    virtual String info() const { return type() + format("[%d,%d]", patchSize, numFilters); }
//};
//

//
// 2 of those, followed by a Hashing stage, and you got PCANet.
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
        randomIndex(randIdx);

        int size = patchSize * patchSize;
        cv::Mat Rx = cv::Mat::zeros(size, size, images[0].type());

        for (size_t j=0; j<images.size(); j++)
        {
            cv::Mat temp = patchImage(images[randIdx(j)], patchSize);
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
// hash input vector to a single feature
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
            cv::Mat t2 = im2col(T, blockSize, blockSize);
            t2 = hist(t2, (int)(pow(2.0, numFilters)) - 1);
            t2 = bsxfun_times(t2, numFilters);
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

    Ptr<Stage> addStage(const String &name)
    {
        Ptr<Stage> s;
        if (name=="FilterBank")  s = makePtr<FilterBank>();
        //if (name=="Random")  s = makePtr<RandomProjection>();
        if (name=="Gabor")  s = makePtr<GaborProjection>();
        if (name=="Pca")  s = makePtr<PcaProjection>();
        if (name=="Wave")  s = makePtr<WaveProjection>();
        //if (name=="Conv")  s = makePtr<ConvLearn>();
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
            layers[i]->train(feat);
            layers[i]->process(feat,post);

            stageViz(post,layers[i]->type() + format("_%d",i), 10);
            cerr << "\t" << i << "\t" << layers[i]->type() << "\t" << feat.size() << "\t" << post.size() << endl;

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

cv::Ptr<PNet> loadNet(const String &fn)
{
    cv::Ptr<Network> pn = makePtr<Network>();
    pn->load(fn);
    cerr << pn->ingo() << endl;
    return pn;
}


#ifdef TRAIN_STANDALONE

int main()
{
    // String path("e:/code/vlc/faces/*.png");
    String path("e:/code/opencv_p/face3/data/funneled/");
    vector<String> fn;
    glob(path,fn);

    vector<Mat> images;
    for (size_t i=0; i<300; ++i)
    {
        size_t idx = theRNG().uniform(0,fn.size());
        Mat im = imread(fn[idx],0);
        im.convertTo(im,CV_32F);
        images.push_back(im);
    }

    Network net;
    if (1)
    {
        cerr << "train " << images.size() << endl;
        net.addStage(makePtr<PcaProjection>(7, 5));
        net.addStage(makePtr<GaborProjection>(9, 5, 0.373f, -1.0f)); // gabor kernels need to be odd
        //net.addStage(makePtr<ConvLearn>(11, 5));
        net.addStage(makePtr<Hashing>(5, 18));
        net.train(images);
        net.save("my.xml");
    }
    else
    {
        net.load("my.xml");
    }
    cerr << net.ingo() << endl;

    namedWindow("filters", 0);
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
