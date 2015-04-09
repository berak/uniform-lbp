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
    int64 e1 = cv::getTickCount();
    
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
    vector<cv::Mat>feat(1,img), post;

    for (int i=0; i<numStages; i++)
    {
        calcStage(feat, post, patchSize, stages[i].numFilters, stages[i].filters, 1);
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


cv::Mat PCANet::trainPCA(vector<cv::Mat>& feat0, bool extract_feature)
{
    PROFILE;
    assert(stages.size() == numStages);
    int64 e1 = cv::getTickCount();

    int end = numStages - 1;
    Stage & lastStage = stages[end];
    vector<cv::Mat>feat(feat0), post;

    for (int i=0; i<end; i++)
    {
        stages[i].filters = pcaFilterBank(feat, patchSize, stages[i].numFilters);
        calcStage(feat, post, patchSize, stages[i].numFilters, stages[i].filters, 1);
        feat.clear();
        cv::swap(feat,post);
    }
    lastStage.filters = pcaFilterBank(feat, patchSize, lastStage.numFilters);
    int64 e2 = cv::getTickCount();
    double time = (e2 - e1) / cv::getTickFrequency();
    cout << "\n Train     time: " << time << endl;

    cv::Mat features;
    if (extract_feature)
    {
        size_t feat0Size = feat0.size();
        feat0.clear();

        vector<cv::Mat>::const_iterator first = feat.begin();
        vector<cv::Mat>::const_iterator last  = feat.begin();
        size_t endFilters = lastStage.numFilters;
        e1 = cv::getTickCount();
        for (size_t i=0; i<feat0Size; i++)
        {
            vector<cv::Mat> subInImg(first + i * lastStage.numFilters, last + (i + 1) * lastStage.numFilters);

            vector<cv::Mat> feat2;
            calcStage(subInImg, feat2, patchSize, lastStage.numFilters, lastStage.filters, 1);

            cv::Mat hashing = hashingHist(feat2);
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
    int end = numStages - 1;
    int filtersLast = stages[end].numFilters;
    int numImgin0 = length / filtersLast;

    vector<double> map_weights;
    for (int i = filtersLast - 1; i >= 0; i--)
    {
        map_weights.push_back(pow(2.0, (double)i));
    }

    vector<int> ro_BlockSize,histBlockSize;
    double rate = 1.0 - blkOverLapRatio;
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
    cv::String s = cv::format("%d %d ", numStages, patchSize);
    for (int i=0; i<numStages; i++)
    {
        s += cv::format("[%d %d]", stages[i].numFilters, stages[i].histBlockSize);
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
        fs << "NumFilters" << stages[i].numFilters;
        fs << "HistBlockSize" << stages[i].histBlockSize;
        fs << "Filter" << stages[i].filters;
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
    fs["NumStages"] >> numStages;
    fs["PatchSize"] >> patchSize;
    fs["BlkOverLapRatio"] >> blkOverLapRatio;
    FileNode pnodes = fs["Stages"];
    for (FileNodeIterator it=pnodes.begin(); it!=pnodes.end(); ++it)
    {
        const FileNode &n = *it;
        Stage stage;
        n["NumFilters"] >> stage.numFilters;
        n["HistBlockSize"] >> stage.histBlockSize;
        n["Filter"] >> stage.filters;
        stages.push_back(stage);
    }
    fs["ProjVecPCA"] >> projVecPCA;
    fs["ProjVecLDA"] >> projVecLDA;
    fs.release();
    return true;
}

int PCANet::addStage(int a, int b)
{
    Stage s = {a,b};
    stages.push_back(s);
    numStages++;
    return stages.size();
}