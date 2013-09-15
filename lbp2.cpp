#include "opencv2/contrib.hpp"
#include <opencv2/core/utility.hpp>
using namespace cv;
//#include <iostream>
#include <limits>

class LBPH : public FaceRecognizer
{
private:
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;

    bool _uniform;
    int  _num_uniforms;
    std::vector<int> _uniform_lookup;

    std::vector<Mat> _histograms;
    Mat _labels;

    // Computes a LBPH model with images in src and
    // corresponding labels in labels, possibly preserving
    // old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);

    void initUniformLookup();

public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes this LBPH Model. The current implementation is rather fixed
    // as it uses the Extended Local Binary Patterns per default.
    //
    // radius, neighbors are used in the local binary patterns creation.
    // grid_x, grid_y control the grid size of the spatial histograms.
    LBPH(int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX,
            bool uniform=false) :
        _grid_x(gridx),
        _grid_y(gridy),
        _radius(radius_),
        _neighbors(neighbors_),
        _threshold(threshold),
        _num_uniforms(0),
        _uniform(uniform) {
            if (uniform) {
                initUniformLookup();
            }
        }

    // Initializes and computes this LBPH Model. The current implementation is
    // rather fixed as it uses the Extended Local Binary Patterns per default.
    //
    // (radius=1), (neighbors=8) are used in the local binary patterns creation.
    // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
    LBPH(InputArrayOfArrays src,
            InputArray labels,
            int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX,
            bool uniform=false) :
                _grid_x(gridx),
                _grid_y(gridy),
                _radius(radius_),
                _neighbors(neighbors_),
                _threshold(threshold),
                _num_uniforms(0),
                _uniform(uniform) {
        if (uniform) {
            initUniformLookup();
        }
        train(src, labels);
    }

    ~LBPH() { }

    // Computes a LBPH model with images in src and
    // corresponding labels in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Updates this LBPH model with images in src and
    // corresponding labels in labels.
    void update(InputArrayOfArrays src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Predicts the label and confidence for a given sample.
    void predict(InputArray _src, int &label, double &dist) const;

    // See FaceRecognizer::load.
    void load(const FileStorage& fs);

    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;

    // Getter functions.
    int neighbors() const { return _neighbors; }
    int radius() const { return _radius; }
    int grid_x() const { return _grid_x; }
    int grid_y() const { return _grid_y; }

    AlgorithmInfo* info() const;
};



//------------------------------------------------------------------------------
// LBPH
//------------------------------------------------------------------------------


//
// a bitmask is 'uniform' if the number of transitions <= 2.
// 
// we precompute the possible values to a lookup(index) table for 
// all possible lbp combinations of n bits(neighbours). 
//
// check, if the 1st bit eq 2nd, 2nd eq 3rd, ..., last eq 1st, 
//   else add a transition for each bit.
//
//   if there's no transition, it's solid
//   1 transition: we've found a solid edge.
//   2 transitions: we've found a line.
//
//   since the radius of the lbp operator is quite small, 
//   we consider any larger number of transitions as noise, 
//   and 'discard' them from our histogram, by assinging all of them 
//   to a single noise bin
//
//   this way, using uniform lbp features boils down to indexing into the lut
//   instead of the original value, and adjusting the sizes for the histograms.
//
bool bit(unsigned b, unsigned i) {
    return ((b & (1 << i)) != 0);
}
void LBPH::initUniformLookup() {
    int numSlots  = 1 << _neighbors;  // 2 ** _neighbours
    _num_uniforms = 0;
    _uniform_lookup = std::vector<int>(numSlots);
    for ( int i=0; i<numSlots; i++ ) {
        int transitions = 0;
        for ( int j=0; j<_neighbors-1; j++ ) {
            transitions += (bit(i,j) != bit(i,j+1));
        }
        transitions += (bit(i,_neighbors-1) != bit(i,0));

        if ( transitions <= 2 ) {
            _uniform_lookup[i] = _num_uniforms++;
        } else {
            _uniform_lookup[i] = -1; // mark all non-uniforms as noise channel
        }
    }

    // now run again through the lut, replace -1 with the 'noise' slot (numUniforms)
    for ( int i=0; i<numSlots; i++ ) {
        if ( _uniform_lookup[i] == -1 ) {
            _uniform_lookup[i] = _num_uniforms;
        }
    }
}


template <typename _Tp> static
void olbp_(InputArray _src, OutputArray _dst,bool uniform, std::vector<int> lookup) {
    // get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2, src.cols-2, CV_8UC1);
    Mat dst = _dst.getMat();
    // zero the result matrix
    dst.setTo(0);
    // calculate patterns
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) >= center) << 7;
            code |= (src.at<_Tp>(i-1,j) >= center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) >= center) << 5;
            code |= (src.at<_Tp>(i,j+1) >= center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) >= center) << 3;
            code |= (src.at<_Tp>(i+1,j) >= center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) >= center) << 1;
            code |= (src.at<_Tp>(i,j-1) >= center) << 0;
            if ( uniform ) {
                code = lookup[ code ];
            }
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
template <typename _Tp> static
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors, bool uniform, std::vector<int> lookup) {
    //get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    Mat dst = _dst.getMat();
    // zero
    dst.setTo(0);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
        float y = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
    if ( uniform ) { // replace content of dst with its resp. lookup value
        for ( size_t i=0; i<dst.total(); i++ ) {
            dst.at<int>(i) = lookup[ dst.at<int>(i) ];
        }
    }
}

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors, bool uniform, std::vector<int> lookup)
{
    int type = src.type();
    switch (type) {
    case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors,uniform,lookup); break;
    case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors,uniform,lookup); break;
    case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors,uniform,lookup); break;
    case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors,uniform,lookup); break;
    case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors,uniform,lookup); break;
    case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors,uniform,lookup); break;
    case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors,uniform,lookup); break;
    default:
        std::string error_msg = format("Using Original Local Binary Patterns for feature extraction only works on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
        CV_Error(Error::StsNotImplemented, error_msg);
        break;
    }
}

static Mat
histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false)
{
    Mat result;
    // Establish the number of bins.
    int histSize = maxVal-minVal+1;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal+1) };
    const float* histRange = { range };
    // calc histogram
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
    // normalize
    if(normed) {
        result /= (int)src.total();
    }
    return result.reshape(1,1);
}

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
{
    Mat src = _src.getMat();
    switch (src.type()) {
        case CV_8SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_8UC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        case CV_16SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_16UC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        case CV_32SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_32FC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        default:
            CV_Error(Error::StsUnmatchedFormats, "This type is not implemented yet."); break;
    }
    return Mat();
}


static Mat spatial_histogram(InputArray _src, int numPatterns,
                             int grid_x, int grid_y, bool /*normed*/)
{
    Mat src = _src.getMat();
    // calculate LBP patch size
    int width = src.cols/grid_x;
    int height = src.rows/grid_y;
    // allocate memory for the spatial histogram
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // return matrix with zeros if no data was given
    if(src.empty())
        return result.reshape(1,1);
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    for(int i = 0; i < grid_y; i++) {
        for(int j = 0; j < grid_x; j++) {
            Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
            Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);
            // copy to the result matrix
            Mat result_row = result.row(resultRowIdx);
            cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return result.reshape(1,1);
}

//------------------------------------------------------------------------------
// wrapper to cv::elbp (extended local binary patterns)
//------------------------------------------------------------------------------

static Mat elbp(InputArray src, int radius, int neighbors, bool uniform, std::vector<int> lookup) {
    Mat dst;
    elbp(src, dst, radius, neighbors, uniform,lookup);
    return dst;
}


// Reads a sequence from a FileNode::SEQ with type _Tp into a result vector.
template<typename _Tp>
inline void readFileNodeList(const FileNode& fn, std::vector<_Tp>& result) {
    if (fn.type() == FileNode::SEQ) {
        for (FileNodeIterator it = fn.begin(); it != fn.end();) {
            _Tp item;
            it >> item;
            result.push_back(item);
        }
    }
}

// Writes the a list of given items to a cv::FileStorage.
template<typename _Tp>
inline void writeFileNodeList(FileStorage& fs, const cv::String& name,
                              const std::vector<_Tp>& items) {
    // typedefs
    typedef typename std::vector<_Tp>::const_iterator constVecIterator;
    // write the elements in item to fs
    fs << name << "[";
    for (constVecIterator it = items.begin(); it != items.end(); ++it) {
        fs << *it;
    }
    fs << "]";
}
void LBPH::load(const FileStorage& fs) {
    fs["radius"] >> _radius;
    fs["neighbors"] >> _neighbors;
    fs["grid_x"] >> _grid_x;
    fs["grid_y"] >> _grid_y;
    //read matrices
    readFileNodeList(fs["histograms"], _histograms);
    fs["labels"] >> _labels;

    fs["uniform"] >> _uniform;

    if (_uniform) {
        initUniformLookup();
    }
}

// See FaceRecognizer::save.
void LBPH::save(FileStorage& fs) const {
    fs << "radius" << _radius;
    fs << "neighbors" << _neighbors;
    fs << "grid_x" << _grid_x;
    fs << "grid_y" << _grid_y;
    // write matrices
    writeFileNodeList(fs, "histograms", _histograms);
    fs << "labels" << _labels;

    fs << "uniform" << _uniform;
}

void LBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels) {
    this->train(_in_src, _in_labels, false);
}

void LBPH::update(InputArrayOfArrays _in_src, InputArray _in_labels) {
    // got no data, just return
    if(_in_src.total() == 0)
        return;

    this->train(_in_src, _in_labels, true);
}

void LBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels, bool preserveData) {
    if(_in_src.kind() != _InputArray::STD_VECTOR_MAT && _in_src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        std::string error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
        CV_Error(Error::StsBadArg, error_message);
    }
    if(_in_src.total() == 0) {
        std::string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(Error::StsUnsupportedFormat, error_message);
    } else if(_in_labels.getMat().type() != CV_32SC1) {
        std::string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
        CV_Error(Error::StsUnsupportedFormat, error_message);
    }
    // get the vector of matrices
    std::vector<Mat> src;
    _in_src.getMatVector(src);
    // get the label matrix
    Mat labels = _in_labels.getMat();
    // check if data is well- aligned
    if(labels.total() != src.size()) {
        std::string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), _labels.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // if this model should be trained without preserving old data, delete old model data
    if(!preserveData) {
        _labels.release();
        _histograms.clear();
    }
    // append labels to _labels matrix
    for(size_t labelIdx = 0; labelIdx < labels.total(); labelIdx++) {
        _labels.push_back(labels.at<int>((int)labelIdx));
    }
    // store the spatial histograms of the original data
    for(size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
        // calculate lbp image
        Mat lbp_image = elbp(src[sampleIdx], _radius, _neighbors,_uniform,_uniform_lookup);
        
        int numPatterns = static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors)));
        if ( _uniform ) {
            numPatterns = _num_uniforms;
        }
        
        // get spatial histogram from this lbp image
        Mat p = spatial_histogram(
                lbp_image, /* lbp_image */
                numPatterns, /* number of possible patterns */
                _grid_x, /* grid size x */
                _grid_y, /* grid size y */
                true);
        // add to templates
        _histograms.push_back(p);
    }
    //std::cout << _histograms.size() << " * " << _histograms[0].total() << " = " << (_histograms.size() * _histograms[0].total()) << " elems." << std::endl;;
}



void LBPH::predict(InputArray _src, int &minClass, double &minDist) const {
    if(_histograms.empty()) {
        // throw error if no data (or simply return -1?)
        std::string error_message = "This LBPH model is not computed yet. Did you call the train method?";
        CV_Error(Error::StsBadArg, error_message);
    }
    Mat src = _src.getMat();
    // get the spatial histogram from input image
    Mat lbp_image = elbp(src, _radius, _neighbors,_uniform, _uniform_lookup);

    int numPatterns = 1 << _neighbors ; 
    if ( _uniform ) {
        numPatterns = _num_uniforms;
    }

    Mat query = spatial_histogram(
            lbp_image, /* lbp_image */
            numPatterns, /* number of possible patterns */
            _grid_x, /* grid size x */
            _grid_y, /* grid size y */
            true /* normed histograms */);
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(size_t sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
        double dist = compareHist(_histograms[sampleIdx], query, HISTCMP_CHISQR);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels.at<int>((int) sampleIdx);
        }
    }
}

int LBPH::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}


Ptr<FaceRecognizer> createLBPHFaceRecognizer2(int radius, int neighbors,
                                             int grid_x, int grid_y, double threshold,bool uniform)
{
    return new LBPH(radius, neighbors, grid_x, grid_y, threshold,uniform);
}


//ach, just return a dummy here
AlgorithmInfo * LBPH::info() const { return 0; }

//#define CV_INIT_ALGORITHM(classname, algname, memberinit) \
//    static ::cv::Algorithm* create##classname() \
//    { \
//        return new classname; \
//    } \
//    \
//    static ::cv::AlgorithmInfo& classname##_info() \
//    { \
//        static ::cv::AlgorithmInfo classname##_info_var(algname, create##classname); \
//        return classname##_info_var; \
//    } \
//    \
//    static ::cv::AlgorithmInfo& classname##_info_auto = classname##_info(); \
//    \
//    ::cv::AlgorithmInfo* classname::info() const \
//    { \
//        static volatile bool initialized = false; \
//        \
//        if( !initialized ) \
//        { \
//            initialized = true; \
//            classname obj; \
//            memberinit; \
//        } \
//        return &classname##_info(); \
//    }
//
//CV_INIT_ALGORITHM(LBPH, "FaceRecognizer.LBPH",
//                  obj.info()->addParam(obj, "radius", obj._radius);
//                  obj.info()->addParam(obj, "neighbors", obj._neighbors);
//                  obj.info()->addParam(obj, "grid_x", obj._grid_x);
//                  obj.info()->addParam(obj, "grid_y", obj._grid_y);
//                  obj.info()->addParam(obj, "threshold", obj._threshold);
//                  obj.info()->addParam(obj, "histograms", obj._histograms, true);
//                  obj.info()->addParam(obj, "labels", obj._labels, true));
//
