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
		const float *p_X = X.ptr<float>(i);
		float *p_H = H.ptr<float>(i);

		for (int j=0; j<X.cols; j++)
        {
			if (p_X[j] > 0) p_H[j] = 1;
			else p_H[j] = 0;
		}
	}
	return H;
}

cv::Mat bsxfun_times(cv::Mat& BHist, int NumFilters)
{
    PROFILE;
	float *p_BHist;
	int row = BHist.rows;
	int col = BHist.cols;

	vector<float> sum(col,0);

    for (int i = 0; i<row; i++)
    {
		p_BHist = BHist.ptr<float>(i);
		for (int j = 0; j<col; j++)
			sum[j] += p_BHist[j];
	}
	float p = pow(2.0f, NumFilters);

	for (int i = 0; i<col; i++)
		sum[i] = p / sum[i];

	for (int i = 0; i<row; i++)
    {
		p_BHist = BHist.ptr<float>(i);
		for (int j=0; j<col; j++)
			p_BHist[j] *= sum[j];
	}

	return BHist;
}

cv::Mat hist(const cv::Mat &mat, int range)
{
    PROFILE;
	cv::Mat temp = mat.t();
	int row = temp.rows;
	int col = temp.cols;

    cv::Mat Hist = cv::Mat::zeros(row, range + 1, CV_32F);

	for (int i=0; i<row; i++)
    {
		const float *p_M = temp.ptr<float>(i);
		float *p_H = Hist.ptr<float>(i);

		for (int j=0; j<col; j++)
        {
			p_H[(int)p_M[j]] += 1;
		}
	}

	return Hist.t();
}


cv::Mat im2col(const cv::Mat &InImg, const vector<int> &blockSize, const vector<int> &stepSize)
{
    PROFILE;
	int row_diff = InImg.rows - blockSize[ROW_DIM];
	int col_diff = InImg.cols - blockSize[COL_DIM];
	int r_row = blockSize[ROW_DIM] * blockSize[COL_DIM];
	int r_col = (row_diff / stepSize[ROW_DIM] + 1) * (col_diff / stepSize[COL_DIM] + 1);
	cv::Mat OutBlocks(r_col, r_row, InImg.type());

	int blocknum = 0;
	for (int j=0; j<=col_diff; j+=stepSize[COL_DIM])
    {
		for (int i=0; i<=row_diff; i+=stepSize[ROW_DIM])
        {
			float *p_OutImg = OutBlocks.ptr<float>(blocknum);

			for (int m=0; m<blockSize[ROW_DIM]; m++)
            {
				const float *p_InImg = InImg.ptr<float>(i + m);

				for (int l=0; l<blockSize[COL_DIM]; l++)
                {
					p_OutImg[blockSize[ROW_DIM] * l + m] = p_InImg[j + l];
				}
			}
			blocknum++;
		}
	}

	return OutBlocks.t();
}

void getRandom(vector<int> &idx)
{
    PROFILE;
	for (size_t i=0; i<idx.size(); i++)
    {
		idx[i] = i;
    }
	for (size_t i=0; i<idx.size(); i++)
    {
        int ti = cv::theRNG().uniform(0,idx.size());
        int t = idx[i];
		idx[i] = idx[ti];
        idx[ti] = t;
    }
}

cv::Mat pcaFilterBank(const vector<cv::Mat> &InImg, int PatchSize, int NumFilters)
{
    PROFILE;
	int channels = InImg[0].channels();
	int InImg_Size = InImg.size();

    vector<int> randIdx(InImg_Size);
	getRandom(randIdx);

	int size = channels * PatchSize * PatchSize;
	cv::Mat Rx = cv::Mat::zeros(size, size, InImg[0].type());

	vector<int> blockSize;
	vector<int> stepSize;

	for (int i=0; i<2; i++)
    {
		blockSize.push_back(PatchSize);
		stepSize.push_back(1);
	}

    int cols=0;
	for (int j=0; j<InImg_Size; j++)
    {
        cv::Mat temp = im2col(InImg[randIdx[j]], blockSize, stepSize);
        cols = temp.cols;

        cv::Mat mean;
		cv::reduce(temp, mean, 0, CV_REDUCE_AVG);

        cv::Mat temp3;
		for (int i = 0; i<temp.rows; i++)
        {
			cv::Mat temp2 = (temp.row(i) - mean.row(0));
			temp3.push_back(temp2.row(0));
		}
		Rx = Rx + temp3 * temp3.t();
	}
	Rx = Rx / (double)(InImg_Size * cols);

	cv::Mat eValuesMat;
	cv::Mat eVectorsMat;

	eigen(Rx, eValuesMat, eVectorsMat);

	cv::Mat Filters(0, Rx.cols, Rx.depth());

	for (int i = 0; i<NumFilters; i++)
		Filters.push_back(eVectorsMat.row(i));
	return Filters;
}




void calcStage(const vector<cv::Mat> &InImg, vector<cv::Mat> &OutImg, int PatchSize, int NumFilters, const cv::Mat& Filters, int threadnum)
{
    PROFILE;
	int img_length = InImg.size();
	int mag = (PatchSize - 1) / 2;

    OutImg.clear();

	cv::Mat img;

	vector<int> blockSize;
	vector<int> stepSize;

	for (int i=0; i<2; i++)
    {
		blockSize.push_back(PatchSize);
		stepSize.push_back(1);
	}

	for (int i=0; i<img_length; i++)
    {
		cv::copyMakeBorder(InImg[i], img, mag, mag, mag, mag, cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::Mat temp = im2col(img, blockSize, stepSize);

        cv::Mat mean;
		cv::reduce(temp, mean, 0, CV_REDUCE_AVG);

		cv::Mat temp3;
		for (int j=0; j<temp.rows; j++)
        {
			cv::Mat temp2 = (temp.row(j) - mean.row(0));
			temp3.push_back(temp2.row(0));
		}
	    for (int j=0; j<NumFilters; j++)
        {
		    temp = Filters.row(j) * temp3;
		    temp = temp.reshape(0, InImg[i].cols);
		    OutImg.push_back(temp.t());
	    }
	}
}



cv::Mat PCANet::extract(const cv::Mat &img) const
{
    PROFILE;
    vector<cv::Mat>s0(1,img),s1,s2;

    calcStage(s0, s1, PatchSize, NumFilters[0], Filters[0], 1);
    s0.clear();

    calcStage(s1, s2, PatchSize, NumFilters[1], Filters[1], 1);
    cv::Mat hashing = hashingHist(s2);

    if (dimensionLDA > 0)
    {
	    cv::Mat LowTFeatTes = hashing * ProjVecPCA.t();
	    hashing = LowTFeatTes * ProjVecLDA;
    } 
    return hashing;
}


cv::Mat PCANet::trainPCA(vector<cv::Mat>& imgs, bool extract_feature)
{
    PROFILE;
	assert(NumFilters.size() == NumStages);
	int64 e1 = cv::getTickCount();

    Filters.push_back(pcaFilterBank(imgs, PatchSize, NumFilters[0]));

    vector<cv::Mat> stage;
    calcStage(imgs, stage, PatchSize, NumFilters[0], Filters[0], 1);

    Filters.push_back(pcaFilterBank(stage, PatchSize, NumFilters[1]));

    cv::Mat Features;
    if (extract_feature)
    {
        imgs.clear();
        vector<cv::Mat>::const_iterator first = stage.begin();
		vector<cv::Mat>::const_iterator last  = stage.begin();
    	int end = NumStages - 1;

		e1 = cv::getTickCount();
		for (size_t i=0; i<imgs.size(); i++)
        {
			vector<cv::Mat> subInImg(first + i * NumFilters[end], last + (i + 1) * NumFilters[end]);

            vector<cv::Mat> om;
            calcStage(subInImg, om, PatchSize, NumFilters[end], Filters[end], 1);

            cv::Mat hashing = hashingHist(om);
			Features.push_back(hashing);
		}
		int64 e2 = cv::getTickCount();
		double time = (e2 - e1) / cv::getTickFrequency();
		cout << "\n Hashing time: " << time << endl;
	}
    return Features;
}


cv::Mat PCANet::hashingHist(const vector<cv::Mat> &Imgs) const
{
    PROFILE;
	int length = Imgs.size();
	int numFilters = NumFilters[NumStages - 1];
	int NumImgin0 = length / numFilters;

	vector<double> map_weights;
	for (int i = numFilters - 1; i >= 0; i--)
		map_weights.push_back(pow(2.0, (double)i));

	vector<int> Ro_BlockSize;
	double rate = 1.0 - BlkOverLapRatio;
	for (size_t i = 0; i<HistBlockSize.size(); i++)
		Ro_BlockSize.push_back(cvRound(HistBlockSize[i] * rate));

	cv::Mat BHist;
	for (int i=0; i<NumImgin0; i++)
    {
        cv::Mat T(Imgs[0].rows, Imgs[0].cols, Imgs[0].type(), cv::Scalar(0));
		for (int j=0; j<numFilters; j++)
        {
        	cv::Mat temp = heaviside(Imgs[numFilters * i + j]);
			temp *= map_weights[j];
			T += temp;
		}

        cv::Mat t2 = im2col(T, HistBlockSize, Ro_BlockSize);
		t2 = hist(t2, (int)(pow(2.0, numFilters)) - 1);
		t2 = bsxfun_times(t2, numFilters);

		if (i == 0) BHist = t2;
		else hconcat(BHist, t2, BHist);
	}

	int rows = BHist.rows;
	int cols = BHist.cols;

    cv::Mat hashed(1, rows * cols, BHist.type());

	float *p_Fe = hashed.ptr<float>(0);
	float *p_Hi;
	for (int i=0; i<rows; i++)
    {
		p_Hi = BHist.ptr<float>(i);
		for (int j=0; j<cols; j++)
        {
			p_Fe[j * rows + i] = p_Hi[j];
		}
	}
	return hashed;
}



cv::Mat PCANet::trainLDA(const cv::Mat &Features, const cv::Mat &labels)
{
    PROFILE;
	cv::Mat ValueMMetPCA, VectorMMetPCA;
    eigen(cv::Mat(Features.t() * Features), ValueMMetPCA, VectorMMetPCA);

	for (int i=0; i<dimensionLDA; i++)
		ProjVecPCA.push_back(VectorMMetPCA.row(i));

	cv::LDA lda;
    lda.compute(cv::Mat(Features * ProjVecPCA.t()), labels);
	lda.eigenvectors().convertTo(ProjVecLDA, CV_32F);

	return Features * ProjVecPCA.t() * ProjVecLDA;
}


cv::String PCANet::settings() const
{
    cv::String s = cv::format("%d %d %d ", dimensionLDA, NumStages, PatchSize);
    for (int i=0; i<NumStages; i++)
    {
        s += cv::format("[%d %d]", NumFilters[i], HistBlockSize[i]);
    }
    return s;
}

bool PCANet::save(const cv::String &fn) const
{
    cv::FileStorage fs(fn, cv::FileStorage::WRITE);
    fs << "NumStages" << NumStages;
    fs << "PatchSize" << PatchSize;
    fs << "BlkOverLapRatio" << BlkOverLapRatio;
    fs << "Stages" << "[" ;
    for (int i=0; i<NumStages; i++)
    {
        fs << "{:" ;
        fs << "NumFilters" << NumFilters[i];
        fs << "HistBlockSize" << HistBlockSize[i];
	    fs << "Filter" << Filters[i];
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
    fs["NumStages"] >> NumStages;
    fs["PatchSize"] >> PatchSize;
    fs["BlkOverLapRatio"] >> BlkOverLapRatio;
    FileNode pnodes = fs["Stages"];
    for (FileNodeIterator it=pnodes.begin(); it!=pnodes.end(); ++it)
    {
        const FileNode &n = *it;

        int nf;
        n["NumFilters"] >> nf;
        NumFilters.push_back(nf);

        int hbs;      
        n["HistBlockSize"] >> hbs;
        HistBlockSize.push_back(hbs);

        Mat fil;
	    n["Filter"] >> fil;
        Filters.push_back(fil);
    }
    fs.release();
    return true;
}