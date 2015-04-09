#include "PCANet.h"
#include "../profile.h"
#include <fstream>

const int ROW_DIM = 0;
const int COL_DIM = 1;


cv::Mat heaviside(const cv::Mat& X)
{
    PROFILE;
    cv::Mat H(X.size(), X.type());
    for (int i=0; i<X.rows; i++)
    {
        const float *x = X.ptr<float>(i);
        float *h = H.ptr<float>(i);
        for (int j=0; j<X.cols; j++)
        {
            h[j] = (x[j] > 0) ? 1.0f : 0.0f;
        }
    }
    return H;
}

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
            sum[j] += pb[j];
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

void getRandom(vector<int> &idx)
{
    PROFILE;
    // fill the index vector
    for (size_t i=0; i<idx.size(); i++)
    {
        idx[i] = i;
    }
    // random shuffle it
    for (size_t i=0; i<idx.size(); i++)
    {
        int ti = cv::theRNG().uniform(0,idx.size());
        int t = idx[i];
        idx[i] = idx[ti];
        idx[ti] = t;
    }
}

cv::Mat pcaFilterBank(const vector<cv::Mat> &images, int patchSize, int numFilters)
{
    PROFILE;
    
    vector<int> randIdx(images.size());
    getRandom(randIdx);

    int size = patchSize * patchSize;
    cv::Mat Rx = cv::Mat::zeros(size, size, images[0].type());

    vector<int> blockSize;
    vector<int> stepSize;

    for (int i=0; i<2; i++)
    {
        blockSize.push_back(patchSize);
        stepSize.push_back(1);
    }

    int cols=0;
    for (size_t j=0; j<images.size(); j++)
    {
        cv::Mat temp = im2col(images[randIdx[j]], blockSize, stepSize);
        cols = temp.cols;

        cv::Mat mean;
        cv::reduce(temp, mean, 0, CV_REDUCE_AVG);

        cv::Mat temp3;
        for (int i=0; i<temp.rows; i++)
        {
            cv::Mat temp2 = (temp.row(i) - mean.row(0));
            temp3.push_back(temp2.row(0));
        }
        Rx = Rx + temp3 * temp3.t();
    }
    Rx = Rx / (double)(images.size() * cols);

    cv::Mat eValuesMat;
    cv::Mat eVectorsMat;

    eigen(Rx, eValuesMat, eVectorsMat);

    cv::Mat filters;
    for (int i = 0; i<numFilters; i++)
    {
        filters.push_back(eVectorsMat.row(i));
    }
    return filters;
}




void calcStage(const vector<cv::Mat> &inImg, vector<cv::Mat> &outImg, int patchSize, int numFilters, const cv::Mat &filters, int threadnum)
{
    PROFILE;
    int img_length = inImg.size();
    int mag = (patchSize - 1) / 2;

    cv::Mat img;

    vector<int> blockSize;
    vector<int> stepSize;

    for (int i=0; i<2; i++)
    {
        blockSize.push_back(patchSize);
        stepSize.push_back(1);
    }

    for (int i=0; i<img_length; i++)
    {
        cv::copyMakeBorder(inImg[i], img, mag, mag, mag, mag, cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::Mat temp = im2col(img, blockSize, stepSize);

        cv::Mat mean;
        cv::reduce(temp, mean, 0, CV_REDUCE_AVG);

        cv::Mat temp3;
        for (int j=0; j<temp.rows; j++)
        {
            cv::Mat temp2 = (temp.row(j) - mean.row(0));
            temp3.push_back(temp2.row(0));
        }
        for (int j=0; j<numFilters; j++)
        {
            temp = filters.row(j) * temp3;
            temp = temp.reshape(0, inImg[i].cols);
            outImg.push_back(temp.t());
        }
    }
}



cv::Mat PCANet::extract(const cv::Mat &img) const
{
    PROFILE;
    vector<cv::Mat>s0(1,img),s1,s2;

    calcStage(s0, s1, patchSize, numFilters[0], filters[0], 1);
    s0.clear();

    calcStage(s1, s2, patchSize, numFilters[1], filters[1], 1);
    cv::Mat hashing = hashingHist(s2);

    if (dimensionLDA > 0)
    {
        cv::Mat lowTFeatTes = hashing * projVecPCA.t();
        hashing = lowTFeatTes * projVecLDA;
    } 
    return hashing;
}


cv::Mat PCANet::trainPCA(vector<cv::Mat>& stage0, bool extract_feature)
{
    PROFILE;
    assert(numFilters.size() == numStages);
    int64 e1 = cv::getTickCount();

    filters.push_back(pcaFilterBank(stage0, patchSize, numFilters[0]));

    vector<cv::Mat> stage1;
    calcStage(stage0, stage1, patchSize, numFilters[0],filters[0], 1);

    filters.push_back(pcaFilterBank(stage1, patchSize, numFilters[1]));

    cv::Mat features;
    if (extract_feature)
    {
        stage0.clear();
        vector<cv::Mat>::const_iterator first = stage1.begin();
        vector<cv::Mat>::const_iterator last  = stage1.begin();
        int end = numStages - 1;

        e1 = cv::getTickCount();
        for (size_t i=0; i<stage0.size(); i++)
        {
            vector<cv::Mat> subInImg(first + i * numFilters[end], last + (i + 1) * numFilters[end]);

            vector<cv::Mat> stage2;
            calcStage(subInImg, stage2, patchSize, numFilters[end], filters[end], 1);

            cv::Mat hashing = hashingHist(stage2);
            features.push_back(hashing);
        }
        int64 e2 = cv::getTickCount();
        double time = (e2 - e1) / cv::getTickFrequency();
        cout << "\n Hashing time: " << time << endl;
    }
    return features;
}


cv::Mat PCANet::hashingHist(const vector<cv::Mat> &images) const
{
    PROFILE;
    int length = images.size();
    int filtersLast = numFilters[numStages - 1];
    int numImgin0 = length / filtersLast;

    vector<double> map_weights;
    for (int i = filtersLast - 1; i >= 0; i--)
    {
        map_weights.push_back(pow(2.0, (double)i));
    }

    vector<int> ro_BlockSize;
    double rate = 1.0 - blkOverLapRatio;
    for (size_t i=0; i<histBlockSize.size(); i++)
    {
        ro_BlockSize.push_back(cvRound(histBlockSize[i] * rate));
    }

    cv::Mat bhist;
    for (int i=0; i<numImgin0; i++)
    {
        cv::Mat T(images[0].rows, images[0].cols, images[0].type(), cv::Scalar(0));
        for (int j=0; j<filtersLast; j++)
        {
            cv::Mat temp = heaviside(images[filtersLast * i + j]);
            temp *= map_weights[j];
            T += temp;
        }

        cv::Mat t2 = im2col(T, histBlockSize, ro_BlockSize);
        t2 = hist(t2, (int)(pow(2.0, filtersLast)) - 1);
        t2 = bsxfun_times(t2, filtersLast);

        if (i == 0) bhist = t2;
        else hconcat(bhist, t2, bhist);
    }

    int rows = bhist.rows;
    int cols = bhist.cols;

    cv::Mat hashed(1, rows * cols, bhist.type());

    float *p_Fe = hashed.ptr<float>(0);
    float *p_Hi;
    for (int i=0; i<rows; i++)
    {
        p_Hi = bhist.ptr<float>(i);
        for (int j=0; j<cols; j++)
        {
            p_Fe[j * rows + i] = p_Hi[j];
        }
    }
    return hashed;
}



cv::Mat PCANet::trainLDA(const cv::Mat &features, const cv::Mat &labels)
{
    PROFILE;

    cv::Mat evec;
    eigen(cv::Mat(features.t() * features), cv::Mat(), evec);

    for (int i=0; i<dimensionLDA; i++)
    {
        projVecPCA.push_back(evec.row(i));
    }

    cv::LDA lda;
    lda.compute(cv::Mat(features * projVecPCA.t()), labels);
    lda.eigenvectors().convertTo(projVecLDA, CV_32F);

    return features * projVecPCA.t() * projVecLDA;
}


cv::String PCANet::settings() const
{
    cv::String s = cv::format("%d %d %d ", dimensionLDA, numStages, patchSize);
    for (int i=0; i<numStages; i++)
    {
        s += cv::format("[%d %d]", numFilters[i], histBlockSize[i]);
    }
    return s;
}

bool PCANet::save(const cv::String &fn) const
{
    cv::FileStorage fs(fn, cv::FileStorage::WRITE);
    fs << "NumStages" << numStages;
    fs << "PatchSize" << patchSize;
    fs << "BlkOverLapRatio" << blkOverLapRatio;
    fs << "Stages" << "[" ;
    for (int i=0; i<numStages; i++)
    {
        fs << "{:" ;
        fs << "NumFilters" << numFilters[i];
        fs << "HistBlockSize" << histBlockSize[i];
        fs << "Filter" << filters[i];
        fs << "}";
    }
    fs << "]";
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
    fs["NumStages"] >> numStages;
    fs["PatchSize"] >> patchSize;
    fs["BlkOverLapRatio"] >> blkOverLapRatio;
    FileNode pnodes = fs["Stages"];
    for (FileNodeIterator it=pnodes.begin(); it!=pnodes.end(); ++it)
    {
        const FileNode &n = *it;

        int nf;
        n["NumFilters"] >> nf;
        numFilters.push_back(nf);

        int hbs;      
        n["HistBlockSize"] >> hbs;
        histBlockSize.push_back(hbs);

        Mat fil;
        n["Filter"] >> fil;
        filters.push_back(fil);
    }
    fs.release();
    return true;
}
