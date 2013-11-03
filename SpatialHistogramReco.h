#ifndef __SpatialHistogramReco_onboard__
#define __SpatialHistogramReco_onboard__

#include "opencv2/contrib.hpp"

using namespace cv;
using namespace std;


//
// base class for any SpatialHistogram implementing FaceReco,
//   overide the `virtual void oper(const Mat & src, Mat & hist) const` 
//      method for the per pixel -> (called per tile)histogram mapping job.
//
class SpatialHistogramReco : public FaceRecognizer
{
protected:

    //! the bread-and-butter thing, collect a histogram (per patch)
    virtual void oper(const Mat & src, Mat & hist) const = 0;

    //! choose a distance function for your algo
    virtual double distance(const Mat & hist_a, Mat & hist_b) const = 0;


private:
    int _grid_x;
    int _grid_y;
    double _threshold;

    int step_size;
    int hist_len;
    int hist_type;

    std::vector<Mat> _histograms;
    Mat _labels;

    // Computes a SpatialHistogramReco model with images in src and
    // corresponding labels in labels, possibly preserving
    // old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);

    Mat spatial_histogram(InputArray _src) const ;
    Mat spatial_histogram_overlap(InputArray _src) const ;


public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    SpatialHistogramReco( int gridx=8, int gridy=8,
            double threshold = DBL_MAX, int h_len=255,int h_type=CV_8U, int step_size=0) :
        _grid_x(gridx),
        _grid_y(gridy),
        _threshold(threshold),
        hist_len(h_len),
        hist_type(h_type),
        step_size(step_size)
    {}


    ~SpatialHistogramReco() { }

    // Computes a model with images in src and
    // corresponding labels in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Updates this model with images in src and
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

    AlgorithmInfo* info() const;
};





#endif // __SpatialHistogramReco_onboard__

