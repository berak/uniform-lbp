#ifndef __fr_onboard__
#define __fr_onboard__

#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/objdetect/objdetect.hpp"

#include <string>
#include <vector>
#include <map>

namespace cv
{

    CV_EXPORTS Mat subspaceProject(InputArray W, InputArray mean, InputArray src);
    CV_EXPORTS Mat subspaceReconstruct(InputArray W, InputArray mean, InputArray src);

    class CV_EXPORTS LDA
    {
    public:
        // Initializes a LDA with num_components (default 0) and specifies how
        // samples are aligned (default dataAsRow=true).
        LDA(int num_components = 0) 
            : _num_components(num_components) 
        {}

        // Initializes and performs a Discriminant Analysis with Fisher's
        // Optimization Criterion on given data in src and corresponding labels
        // in labels. If 0 (or less) number of components are given, they are
        // automatically determined for given data in computation.
        LDA(const Mat& src, std::vector<int> labels, int num_components = 0) 
            : _num_components(num_components)
        {
            this->compute(src, labels); //! compute eigenvectors and eigenvalues
        }

        // Initializes and performs a Discriminant Analysis with Fisher's
        // Optimization Criterion on given data in src and corresponding labels
        // in labels. If 0 (or less) number of components are given, they are
        // automatically determined for given data in computation.
        LDA(InputArrayOfArrays src, InputArray labels, int num_components = 0) 
            :  _num_components(num_components)
        {
            this->compute(src, labels); //! compute eigenvectors and eigenvalues
        }

        // Serializes this object to a given filename.
        void save(const std::string& filename) const;

        // Deserializes this object from a given filename.
        void load(const std::string& filename);

        // Serializes this object to a given cv::FileStorage.
        void save(FileStorage& fs) const;

            // Deserializes this object from a given cv::FileStorage.
        void load(const FileStorage& node);

        // Destructor.
        ~LDA() {}

        //! Compute the discriminants for data in src and labels.
        void compute(InputArrayOfArrays src, InputArray labels);

        // Projects samples into the LDA subspace.
        Mat project(InputArray src);

        // Reconstructs projections from the LDA subspace.
        Mat reconstruct(InputArray src);

        // Returns the eigenvectors of this LDA.
        Mat eigenvectors() const { return _eigenvectors; };

        // Returns the eigenvalues of this LDA.
        Mat eigenvalues() const { return _eigenvalues; }

    protected:
        bool _dataAsRow;
        int _num_components;
        Mat _eigenvectors;
        Mat _eigenvalues;

        void lda(InputArrayOfArrays src, InputArray labels);
    };

    class CV_EXPORTS_W FaceRecognizer : public Algorithm
    {
    public:
        //! virtual destructor
        virtual ~FaceRecognizer() {}

        // Trains a FaceRecognizer.
        CV_WRAP virtual void train(InputArrayOfArrays src, InputArray labels) = 0;

        // Updates a FaceRecognizer.
        CV_WRAP void update(InputArrayOfArrays src, InputArray labels);

        // Gets a prediction from a FaceRecognizer.
        virtual int predict(InputArray src) const = 0;

        // Predicts the label and confidence for a given sample.
        CV_WRAP virtual void predict(InputArray src, CV_OUT int &label, CV_OUT double &confidence) const = 0;

        // Serializes this object to a given filename.
        CV_WRAP virtual void save(const std::string& filename) const;

        // Deserializes this object from a given filename.
        CV_WRAP virtual void load(const std::string& filename);

        // Serializes this object to a given cv::FileStorage.
        virtual void save(FileStorage& fs) const = 0;

        // Deserializes this object from a given cv::FileStorage.
        virtual void load(const FileStorage& fs) = 0;

        // Sets additional information as pairs label - info.
        void setLabelsInfo(const std::map<int, std::string>& labelsInfo);

        // Gets string information by label
        std::string getLabelInfo(const int &label);

        // Gets labels by string
        std::vector<int> getLabelsByString(const std::string& str);
    };

    CV_EXPORTS_W Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
    CV_EXPORTS_W Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
    CV_EXPORTS_W Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8,
                                                            int grid_x=8, int grid_y=8, double threshold = DBL_MAX);


}

#endif // __fr_onboard__
