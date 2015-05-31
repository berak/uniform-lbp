
namespace  TextureFeature {



struct PNet : public Extractor
{
    struct Stage
    {
        vector<cv::Mat> filters;
        void convolute(const cv::Mat &im_in, vector<cv::Mat> &im_out) const
        {
            for (size_t i=0; i<filters.size(); i++)
            {
                cv::Mat r;
                sepFilter2D(im_in, r, CV_32F, filters[i].row(0), filters[i].col(0));
                //sepFilter2D(im_in, r, CV_32F, filters[i].row(filters[i].cols/2), filters[i].col(filters[i].rows/2));

                //Point anchor(filters[i].cols -filters[i].cols/2 - 1, filters[i].rows - filters[i].rows/2 - 1);
                //Mat filflip;
                //flip(filters[i],filflip,-1);
                //filter2D(im_in, r, CV_32F, filflip, anchor);

                //filter2D(im_in, r, CV_32F, filters[i]);
                im_out.push_back(r);
            }
        }
        void convolute(const vector<cv::Mat> &im_in, vector<cv::Mat> &im_out) const
        {
            for (size_t i=0; i<im_in.size(); i++)
            {
                convolute(im_in[i], im_out);
            }
        }
        void addFilter(int dim, float tx, float ty)
        {
            //double sigma  = 12.0;;
            //double theta  = double(ty) * 30.0;
            //double lambda = double(tx) * 20.0;// * 180.0;//180.0 - theta;//45.0;// * 180.0 / CV_PI;
            //double gamma  = 6.0;// * 180.0 / CV_PI;
            //double psi = CV_PI;//90.0;
            //cv::Mat proj = getGaborKernel(Size(dim,dim),sigma,theta,lambda,gamma,psi);

            cv::Mat proj(dim, dim, CV_32F);
            for (int i=0; i<dim; i++)
            {
                float *fp = proj.ptr<float>(i);
                for (int j=0; j<dim; j++)
                {
                    //*fp++ = sin(tx*float(j+3)) + cos(ty*float(i+3));
                    *fp++ = sin(tx*float(i+3) + (ty*float(j+3)));
                }
            }
            filters.push_back(proj);
        }
        void pooling(const vector<cv::Mat> &in, vector<cv::Mat> &out) const
        {
            size_t nperFilter = in.size() / filters.size();
            for (size_t i=0,k=0; i<nperFilter; i++)
            {
                cv::Mat T(in[0].size(), CV_32F, Scalar(0));
                for (size_t j=0; j<filters.size(); j++)
                {
                    cv::Mat t;
                    threshold(in[k], t, 0.0, float(1<<(1+j)), 0);
                    T += t;
                    k++;
                }
                out.push_back(T);
            }
        }
    };
    vector<Stage> stages;
    int hgrid;
    int hsize;
    

    PNet() 
        : stages(2)
        , hgrid(2)
        , hsize(64) 
    {
        PNet::Stage &s1 = stages[0];
        PNet::Stage &s2 = stages[1];
        s1.addFilter(13, -0.393f, 1.102f);
        s1.addFilter(12, 2.471f, 0.114f);
        s1.addFilter(9, -0.299f, 1.718f);
        s1.addFilter(12, -0.171f, 1.400f);
        s2.addFilter(9, -0.273f, 0.774f);
        s2.addFilter(14, 2.478f, 3.024f);
        s2.addFilter(15, 2.810f, 1.630f);
        s2.addFilter(14, 2.182f, -0.278f);
        s2.addFilter(9, 1.584f, 2.696f);
        s2.addFilter(16, 0.345f, 2.503f);
    }

    PNet(const cv::String &fn)
        : stages(2)
        //, hgrid(2)
        //, hsize(16)
    {
        load(fn);
        hsize = 1 << stages[1].filters.size();
    }
    

    PNet(int nstages, int hgrid=4, int hsize=16) 
        : stages(nstages)
        , hgrid(hgrid)
        , hsize(hsize) 
    {}

    //cv::Mat bsxfun(const cv::Mat &bhist, int numFilters) const
    //{
    //    double p = double(1<<numFilters);
    //    double s = cv::sum(bhist)[0];
    //    return (bhist * (p/s));
    //}


    cv::Mat bsxfun_times(cv::Mat &bhist, int numFilters) const
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
        //cerr << "sum1" << endl << Mat(sum).t() << endl;

        float p = pow(2.0f, numFilters);
        for (int i = 0; i<col; i++)
        {
            sum[i] = p / sum[i];
        }

        //cerr << "sum2" << endl << Mat(sum).t() << endl;
        for (int i = 0; i<row; i++)
        {
            float *pb = bhist.ptr<float>(i);
            for (int j=0; j<col; j++)
            {
                pb[j] *= sum[j];
            }
        }

        //cerr << "sum3" << endl << bhist << endl;
        return bhist;
    }


    cv::Mat bhist(const cv::Mat &in, int numFilters) const
    {
        cv::Mat his;
        float range[] = {0.0f, hsize};
        const float *histRange = {range};
        int sw = in.cols/hgrid;
        int sh = in.rows/hgrid;
        for (int i=0; i<hgrid; i++)
        {
            for (int j=0; j<hgrid; j++)
            {
                cv::Mat h, patch(in, Range(j*sh, (j+1)*sh), Range(i*sw, (i+1)*sw));
                calcHist(&in, 1, 0, cv::Mat(), h, 1, &hsize, &histRange, true, false);
                //normalize(h,h);
                //h = bsxfun(h, numFilters);
                his.push_back(h.reshape(1,1));
            }
        }
       // cerr << "in" << endl << his << endl;
        Mat h = bsxfun_times(Mat(his.t()), numFilters);
       // cerr << "out" << endl << h << endl;
        return h.reshape(1,1);
        //return his.reshape(1,1);
    }


    int extract(const cv::Mat &img, cv::Mat &res) const
    {
        cv::Mat imf;
        img.convertTo(imf,CV_32F);

        Scalar m,s; meanStdDev(imf, m, s);
        Mat white = imf;
        white -= m[0];
        white /= s[0];

        vector<cv::Mat> conv(1,white), ctmp, pool;
        for (size_t i=0; i<stages.size(); i++)
        {
            stages[i].convolute(conv, ctmp);
            cv::swap(conv, ctmp);
            ctmp.clear();
        }

        const Stage &last = stages.back();
        last.pooling(conv, pool);

        cv::Mat r;
        for (size_t i=0; i<pool.size(); i++)
        {
           // cv::Mat c; normalize(pool[i], c, 1, 0, NORM_MINMAX);
            cv::Mat c = pool[i];
            cv::Mat h = bhist(c, last.filters.size());
            r.push_back(h);
        }
        res = r.reshape(1,1);
        return res.total() * res.elemSize();
    }

    bool load(const cv::String &fn)
    {
        cv::FileStorage fs(fn, cv::FileStorage::READ);
        if (! fs.isOpened()) 
        {
            cerr << "PNet::load : " << fn << " nor found !" << endl;
            return false;
        }

        int patchSize;
        fs["PatchSize"]       >> patchSize;

        cv::FileNode pnodes = fs["Stages"];
        int stageNo=0;
        for (cv::FileNodeIterator it=pnodes.begin(); it!=pnodes.end(); ++it)
        {
            const cv::FileNode &n = *it;
            int hg;
            n["HistBlockSize"] >> hg;
            hgrid = 90/hg;

            cv::Mat filters;
            n["Filter"]        >> filters;
            for (int i=0; i<filters.rows; i++)
            {
                stages[stageNo].filters.push_back(filters.row(i).reshape(1,patchSize));
            }
            stageNo ++;
        }
        fs.release();
        return true;
    }



    cv::Mat filterVis() const
    {
        size_t maxFilters=0;
        size_t maxFilterSize=0;
        for (size_t i=0; i<stages.size(); i++)
        {
            maxFilters = std::max(stages[i].filters.size(), maxFilters);
            for (size_t j=0; j<stages[i].filters.size(); j++)
            {
                if ( stages[i].filters[j].cols > int(maxFilterSize))
                    maxFilterSize = size_t(stages[i].filters[j].cols);
            }
        }
        maxFilterSize += 2;
        cv::Mat fils(maxFilterSize*stages.size(), maxFilterSize*maxFilters, CV_8U, Scalar(0));
        for (size_t i=0; i<stages.size(); i++)
        {
            cv::Mat res;
            for (size_t j=0; j<stages[i].filters.size(); j++)
            {
                Rect roi(1+j*maxFilterSize,1+i*maxFilterSize, stages[i].filters[j].cols,stages[i].filters[j].rows);
                stages[i].filters[j].convertTo( fils(roi), CV_8U,128,128);
            }
        }
        return fils;
    }
};

} // TextureFeature

