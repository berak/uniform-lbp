/*
The MIT License(MIT)

Copyright(c) 2015 Yang Cao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//
// code from https://github.com/delphifirst/FaceX/
// (compressed to a single unit)
//

#ifndef FACE_X_H_
#define FACE_X_H_



#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


struct Transform
{
    cv::Matx22d scale_rotation;
    cv::Matx21d translation;

    void Apply(std::vector<cv::Point2d> &x, bool need_translation = true) const;
};

template<typename T>
inline T Sqr(T a)
{
    return a * a;
}

// Find the transform from y to x
Transform Procrustes(const std::vector<cv::Point2d> &x,
    const std::vector<cv::Point2d> &y);

std::vector<cv::Point2d> ShapeAdjustment(const std::vector<cv::Point2d> &shape,
    const std::vector<cv::Point2d> &offset);

std::vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
    const std::vector<cv::Point2d> original_landmarks,
    cv::Rect new_face_rect);




struct Fern
{
    void ApplyMini(cv::Mat features, std::vector<double> &coeffs)const;

    void read(const cv::FileNode &fn);

    std::vector<double> thresholds;
    std::vector<std::pair<int, int>> features_index;
    std::vector<std::vector<std::pair<int, double>>> outputs_mini;
};

void read(const cv::FileNode& node, Fern& f, const Fern&);



class Regressor
{
public:
    std::vector<cv::Point2d> Apply(const Transform &t,
        cv::Mat image, const std::vector<cv::Point2d> &init_shape) const;

    void read(const cv::FileNode &fn);

private:

    std::vector<std::pair<int, cv::Point2d>> pixels_;
    std::vector<Fern> ferns_;
    cv::Mat base_;
};

void read(const cv::FileNode& node, Regressor& r, const Regressor&);





class FaceX
{
public:
    // Construct the object and load model from file.
    //
    // filename: The file name of the model file.
    //
    // Throw runtime_error if the model file cannot be opened.
    FaceX(const std::string &filename);

    // Do face alignment.
    //
    // image: The image which contains face. Must be 8 bits gray image.
    // face_rect: Where the face locates.
    //
    // Return the landmarks. The number and positions of landmarks depends on
    // the model.
    std::vector<cv::Point2d> Alignment(cv::Mat image, cv::Rect face_rect) const;

    // Do face alignment incrementally. Useful for videos.
    //
    // image: The image which contains face. Must be 8 bits gray image.
    // initial_landmarks: Initial guess of where each landmark is.
    //
    // Return the landmarks. The number and positions of landmarks depends on
    // the model.
    std::vector<cv::Point2d> Alignment(cv::Mat image,
        std::vector<cv::Point2d> initial_landmarks) const;

    // Return how many landmarks the model provides for a face.
    int landmarks_count() const
    {
        return mean_shape_.size();
    }

private:
    std::vector<cv::Point2d> mean_shape_;
    std::vector<std::vector<cv::Point2d>> test_init_shapes_;
    std::vector<Regressor> stage_regressors_;
};

#endif
