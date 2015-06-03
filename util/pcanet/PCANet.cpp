#include "PCANet.h"

#if 1
 #include "../../profile.h"
#else
 #define PROFILE
 #define PROFILEX
#endif

using std::cerr;
using std::cout;
using std::endl;



cv::Mat bsxfun_times(cv::Mat &bhist, int numFilters)
{
    PROFILE;
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


cv::Mat hist(const cv::Mat &mat, int range)
{
    PROFILE;
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


cv::Mat im2col(const cv::Mat &images, const vector<int> &blockSize, const vector<int> &stepSize)
{
    PROFILE;
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


cv::Mat reduceMean(const cv::Mat &image, int patchSize)
{
    PROFILE;
    vector<int> blockSize(2, patchSize);
    vector<int> stepSize(2, 1);
    cv::Mat temp = im2col(image, blockSize, stepSize);
    return temp;
    //cv::Mat mean;
    //{
    //    PROFILEX("reduce");
    //    cv::reduce(temp, mean, 0, cv::REDUCE_AVG);
    //}
    ////cv::Scalar m,s; cv::meanStdDev(temp, m, s);
    ////cv::Mat white = temp;
    ////white -= m[0];
    ////white /= s[0];
    //cv::Mat res;
    //for (int i=0; i<temp.rows; i++)
    //{
    //    PROFILEX("reduce_sub");
    //    cv::Mat temp2 = (temp.row(i) - mean.row(0));
    //    res.push_back(temp2.row(0));
    //}
    //return res;
}


//
// only used for pca-training
//
cv::Mat pcaFilterBank(const vector<cv::Mat> &images, int patchSize, int numFilters)
{
    PROFILE;
    int64 e1 = cv::getTickCount();
    
    cv::Mat_<int> randIdx(1, images.size());
    for (size_t i=0; i<images.size(); i++)
    {
        randIdx(i) = i;
    }
    cv::randShuffle(randIdx);

    int size = patchSize * patchSize;
    cv::Mat Rx = cv::Mat::zeros(size, size, images[0].type());

    for (size_t j=0; j<images.size(); j++)
    {
        cv::Mat temp = reduceMean(images[randIdx(j)], patchSize);
        Rx = Rx + temp * temp.t();
    }
    int count = images[0].cols * images.size();
    Rx = Rx / (double)(count);

    cv::Mat evals, evecs;
    eigen(Rx, evals, evecs);

    cv::Mat filters;
    for (int i = 0; i<numFilters; i++)
    {
        filters.push_back(evecs.row(i));
    }
    int64 e2 = cv::getTickCount();
    double time = (e2 - e1) / cv::getTickFrequency();
    cout << "\n Filter  time: " << time << " " << filters.size() << " " << images.size() << " " << images[0].size() << endl;
    return filters;
}


//
// swapping the im2col -> gemm approach for filter2D()
//   gives like a 30 times speedup, yet looses some bits of accuracy.
//
void pcaStage(const vector<cv::Mat> &inImg, vector<cv::Mat> &outImg, int patchSize, int numFilters, const cv::Mat &filters, int threadnum)
{
    PROFILE;
    int img_length = inImg.size();
    int mag = (patchSize - 1) / 2;

    for (int i=0; i<img_length; i++)
    {
        const cv::Mat &img = inImg[i];
        for (int j=0; j<numFilters; j++)
        {
            PROFILEX("pca_mult");
            cv::Mat tf = filters.row(j).reshape(1,patchSize);
            cv::Mat temp;
            filter2D(img, temp, CV_32F, tf);
            outImg.push_back(temp);
        }
        //{
        //    PROFILEX("pca_border");
        //    cv::copyMakeBorder(inImg[i], img, mag, mag, mag, mag, cv::BORDER_CONSTANT, cv::Scalar(0));
        //}
        //cv::Mat temp3 = reduceMean(img, patchSize);
        //for (int j=0; j<numFilters; j++)
        //{
        //    PROFILEX("pca_mult");
        //    cv::Mat temp = filters.row(j) * temp3;
        //    temp = temp.reshape(0, inImg[i].cols);
        //    outImg.push_back(temp.t());
        //}
    }
}


cv::Mat PCANet::extract(const cv::Mat &img) const
{
    PROFILEX("extract");

    vector<cv::Mat>feat(1,img), post;

    for (int i=0; i<numStages; i++)
    {
        pcaStage(feat, post, patchSize, stages[i].numFilters, stages[i].filters, 1);
        feat.clear();
        cv::swap(feat,post);
    }

    cv::Mat hashing = hashingHist(feat);

    if ((!projVecPCA.empty()) && (!projVecLDA.empty()))
    {
        cv::Mat lowTFeatTes = hashing * projVecPCA.t();
        hashing = lowTFeatTes * projVecLDA;
    } 
    return hashing;
}


cv::Mat PCANet::hashingHist(const vector<cv::Mat> &images) const
{
    PROFILE;
    int length = images.size();
    int end = numStages - 1;
    int filtersLast = stages[end].numFilters;
    int numImgin0 = length / filtersLast;

    vector<double> map_weights;
    for (int i = filtersLast - 1; i >= 0; i--)
    {
        map_weights.push_back(pow(2.0, (double)i));
    }

    vector<int> ro_BlockSize,histBlockSize;
    double rate = 1.0 - blockOverLapRatio;
    for (size_t i=0; i<stages.size(); i++)
    {
        histBlockSize.push_back(stages[i].histBlockSize);
        ro_BlockSize.push_back(cvRound(stages[i].histBlockSize * rate));
    }

    cv::Mat bhist;
    for (int i=0; i<numImgin0; i++)
    {
        cv::Mat T(images[0].rows, images[0].cols, images[0].type(), cv::Scalar(0));
        for (int j=0; j<filtersLast; j++)
        {
            cv::Mat temp;
            threshold(images[filtersLast * i + j], temp, 0.0, map_weights[j], 0);
            T += temp;
        }

        cv::Mat t2 = im2col(T, histBlockSize, ro_BlockSize);
        t2 = hist(t2, (int)(pow(2.0, filtersLast)) - 1);
        t2 = bsxfun_times(t2, filtersLast);

        if (i == 0) bhist = t2;
        else hconcat(bhist, t2, bhist);
    }

    return bhist.reshape(1,1); 
}


cv::Mat PCANet::trainPCA(vector<cv::Mat> &images, bool extract_feature)
{
    PROFILE;
    assert(stages.size() == numStages);
    int64 e1 = cv::getTickCount();

    vector<cv::Mat>feat(images), post;
    for (int i=0; i<(numStages - 1); i++)
    {
        stages[i].filters = pcaFilterBank(feat, patchSize, stages[i].numFilters);
        pcaStage(feat, post, patchSize, stages[i].numFilters, stages[i].filters, 1);
        feat.clear();
        cv::swap(feat,post);
    }
    Stage &lastStage = stages[numStages - 1];
    lastStage.filters = pcaFilterBank(feat, patchSize, lastStage.numFilters);

    int64 e2 = cv::getTickCount();
    double time = (e2 - e1) / cv::getTickFrequency();
    cout << "\n Train     time: " << time << endl;

    cv::Mat features;
    if (extract_feature)
    {
        size_t feat0Size = images.size();
        images.clear();

        vector<cv::Mat>::const_iterator first = feat.begin();
        vector<cv::Mat>::const_iterator last  = feat.begin();
        for (size_t i=0; i<feat0Size; i++)
        {
            vector<cv::Mat> subInImg(first + i * lastStage.numFilters, last + (i + 1) * lastStage.numFilters);

            vector<cv::Mat> feat2;
            pcaStage(subInImg, feat2, patchSize, lastStage.numFilters, lastStage.filters, 1);

            cv::Mat hashing = hashingHist(feat2);
            features.push_back(hashing);
        }
        cout << "\n Extraction time: " << ((cv::getTickCount() - e2) / cv::getTickFrequency()) << endl;
    }
    return features;
}


cv::Mat PCANet::trainLDA(const cv::Mat &features, const cv::Mat &labels, int dimensionLDA)
{
    PROFILE;

    cv::Mat evals, evecs;
    eigen(cv::Mat(features.t() * features), evals, evecs);

    for (int i=0; i<dimensionLDA; i++)
    {
        projVecPCA.push_back(evecs.row(i));
    }

    cv::LDA lda;
    lda.compute(cv::Mat(features * projVecPCA.t()), labels);
    lda.eigenvectors().convertTo(projVecLDA, CV_32F);

    return features * projVecPCA.t() * projVecLDA;
}


cv::String PCANet::settings() const
{
    cv::String s = _type + cv::format(" %d %d ", numStages, patchSize);
    for (int i=0; i<numStages; i++)
    {
        s += cv::format("[%d %d]", stages[i].numFilters, stages[i].histBlockSize);
    }
    return s;
}


bool PCANet::save(const cv::String &fn) const
{
    cv::FileStorage fs(fn, cv::FileStorage::WRITE);
    fs << "Type"            << _type;
    fs << "NumStages"       << numStages;
    fs << "PatchSize"       << patchSize;
    fs << "BlockOverLapRatio" << blockOverLapRatio;
    fs << "Stages" << "[" ;
    for (int i=0; i<numStages; i++)
    {
        fs << "{:" ;
        fs << "NumFilters"    << stages[i].numFilters;
        fs << "HistBlockSize" << stages[i].histBlockSize;
        fs << "Filter"        << stages[i].filters;
        fs << "}";
    }
    fs << "]";
    fs << "ProjVecPCA" << projVecPCA;
    fs << "ProjVecLDA" << projVecLDA;
    fs.release();
    return true;
}


bool PCANet::load(const cv::String &fn)
{
    cv::FileStorage fs(fn, cv::FileStorage::READ);
    if (! fs.isOpened()) 
    {
        cerr << "PCANet::load : " << fn << " nor found !" << endl;
        return false;
    }
    fs["Type"]            >> _type;
    fs["NumStages"]       >> numStages;
    fs["PatchSize"]       >> patchSize;
    fs["BlockOverLapRatio"] >> blockOverLapRatio;
    cv::FileNode pnodes = fs["Stages"];
    for (cv::FileNodeIterator it=pnodes.begin(); it!=pnodes.end(); ++it)
    {
        const cv::FileNode &n = *it;
        Stage stage;
        n["NumFilters"]    >> stage.numFilters;
        n["HistBlockSize"] >> stage.histBlockSize;
        n["Filter"]        >> stage.filters;
        stages.push_back(stage);
    }
    fs["ProjVecPCA"] >> projVecPCA;
    fs["ProjVecLDA"] >> projVecLDA;
    fs.release();
    return true;
}


cv::Mat PCANet::filterVis() const
{
    int maxFilters=0;
    for (int i=0; i<numStages; i++)
    {
        maxFilters = std::max(stages[i].numFilters, maxFilters);
    }
    cv::Mat fils;
    for (int i=0; i<numStages; i++)
    {
        cv::Mat f; stages[i].filters.convertTo(f,CV_8U,128,128);
        cv::Mat res;        
        for (int j=0; j<maxFilters; j++)
        {
            cv::Mat r;
            if (j<stages[i].numFilters)
                r = f.row(j).clone().reshape(1,patchSize);
            else
                r = cv::Mat(patchSize,patchSize,CV_8U,cv::Scalar(0));
            cv::Mat rb;
            cv::copyMakeBorder(r,rb,1,1,1,1,cv::BORDER_CONSTANT);
            if (j==0) res=rb;
            else cv::hconcat(res,rb,res);
        }
        fils.push_back(res);
    }
    return fils;
}


int PCANet::addStage(int a, int b)
{
    Stage s = {a,b};
    stages.push_back(s);
    numStages++;
    return stages.size();
}


void PCANet::randomProjection()
{
    cv::theRNG().state = 37183927;
    for (int s=0; s<numStages; s++)
    {
        int N = stages[s].numFilters, K = patchSize*patchSize;
        cv::Mat proj(N, K, CV_32F);
        cv::randn(proj, cv::Scalar(0), cv::Scalar(1.0));
        for (int i=0; i<N; i++)
        {
            cv::normalize(proj.row(i), proj.row(i));
        }
        stages[s].filters = proj;
    }
    _type = "Rand";
}

void PCANet::waveProjection(float freq)
{
    for (int s=0; s<numStages; s++)
    {
        int N = stages[s].numFilters, K = patchSize*patchSize;
        cv::Mat proj(N, K, CV_32F);

        float t = freq * 0.17f;
        float off = 1.0f + 1.0f/float(s+1);
        for (int i=0; i<N; i++)
        {
            float *fp = proj.ptr<float>(i);
            for (int j=0; j<K; j++)
            {
                *fp++ = sin(off + (t*float(j+3)*float(i+3)));
            }
        }
        freq *= 0.71f;
        stages[s].filters = proj;
    }
    _type = "Wave";
}

void PCANet::gaborProjection(float freq)
{
    for (int s=0; s<numStages; s++)
    {
        int N = stages[s].numFilters, K = patchSize*patchSize;
        cv::Mat proj;
        for (int i=0; i<N; i++)
        {
            double sigma  = 12.0;;
            double theta  = double(freq*(7-i)) * CV_PI;
            double lambda = double(1.3*(10-i)) * CV_PI/4;// * 180.0;//180.0 - theta;//45.0;// * 180.0 / CV_PI;
            double gamma  = (i+5);//6.0;// * 180.0 / CV_PI;
            double psi = CV_PI;//90.0;
            cv::Mat gab = cv::getGaborKernel(cv::Size(patchSize,patchSize),sigma,theta,lambda,gamma,psi);
            gab.convertTo(gab,CV_32F);
            proj.push_back(gab.reshape(1,1));
        }
        freq *= 3.71f;
        stages[s].filters = proj;
    }
    _type = "Gabor";
}
//
//int checker(cv::Mat &m, int b)
//{
//    for (int i=0; i<m.rows-2*b; i+=2*b)
//    {
//        for (int j=0; j<m.cols-2*b; j+=2*b)
//        {
//            m(cv::Rect(i,j,b,b)) ~= 1;
//        }
//    }
//}
//
//void PCANet::haarProjection(float freq)
//{
//    for (int s=0; s<numStages; s++)
//    {
//        int N = stages[s].numFilters, K = patchSize*patchSize;
//        cv::Mat proj(N, K, CV_32F);
//
//        for (int i=0; i<N; i++)
//        {
//            checker(m(cv::Rect(i*patchSize,0)), patchSize/(N+1));
//        }
//        freq *= 0.71f;
//        stages[s].filters = proj;
//    }
//    _type = "Wave";
//}
