/* 
 * File:   rbmUtils.cpp
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
//#include <unistd.h>
#include "rbmUtils.h"

namespace artelab
{

    float average_mse(RBM* rbm, const cv::Mat& patterns)
    {
        float averageError = 0;

        for(int i=0; i < patterns.rows; i++)
        {
            cv::Mat pattern = patterns.row(i);
            cv::Mat reconstruction;
            rbm->reconstruct(pattern, reconstruction);

            cv::Mat square = pattern - reconstruction;
            square = square * square.t() / pattern.cols;

            averageError += square.at<float>(0,0) / patterns.rows;
        }
        return averageError;
    }

    void feature_patterns(RBM* rbm, cv::Mat patterns, cv::Mat& featurePatterns, const bool probabilities)
    {
        featurePatterns.create(0, rbm->num_hidden(), CV_32F);
        for(int r=0; r < patterns.rows; r++)
        {
            cv::Mat row = patterns.row(r);
            cv::Mat target;
            rbm->hidden_activations_for(row, target, probabilities);
            featurePatterns.push_back(target);
        }
    }


    cv::Mat histogram_of_matrix(const cv::Mat& mat, int bins, float min, float max)
    {
        float range[] = { min, max }; // upper bound is exclusive
        const float* hist_range[] = { range };
        bool uniform = true;
        bool accumulate = false;
        int channels[] = { 0 };

        cv::Mat hist;
        cv::calcHist(&mat, 1, channels, cv::Mat(), hist, 1, &bins, hist_range, uniform, accumulate);

        return hist.t();
    }

    cv::Mat histogram_yx(const cv::Mat& mat, int bins, float& min, float& max)
    {
        cv::Mat out(2, bins, CV_32F);
        if(min == 0 && max == 0)
        {
            double mi, ma;
            cv::minMaxLoc(mat, &mi, &ma);
            min = float(mi);
            max = float(ma) + 1e-5f; // upper bound is exclusive
        }

        float step = (max - min) / float(bins);
        step *= step > 0? 1 : -1;
        for(int i=0; i < bins; i++)
        {
            out.at<float>(1, i) = min + i * step;
        }

        cv::Mat hist = histogram_of_matrix(mat, bins, min, max);

        hist.copyTo(out.row(0));

        return out;
    }

    cv::Mat weight_distribution(RBM* rbm, int bins, float& min, float& max)
    {
        return histogram_yx(rbm->weights, bins, min, max);
    }


    cv::Mat updates_distribution(RBM* rbm, int bins, float& min, float& max)
    {
        return histogram_yx(rbm->weights_update(), bins, min, max);
    }

    //void save_histogram_image(cv::Mat hist, FileInfo file, std::string title)
    //{
    //    //CV_Assert(hist.rows == 2);
    //    //using std::endl;

    //    //cv::Mat x = hist.row(1);
    //    //cv::Mat y = hist.row(0);

    //    //char cwd[256];
    //    //getcwd(cwd, sizeof(cwd));

    //    //std::string  dir;
    //    //if(file.fullName() == "")
    //    //    dir = std::string(cwd);
    //    //else
    //    //    dir = file.getBaseDir();

    //    //std::ostringstream ss;
    //    //ss << "reset" << endl
    //    //   << "set term png truecolor" << endl
    //    //   << "set output \"" << file.getName() << "\"" << endl
    //    //   << "set title \"" << title << "\"" << endl
    //    //   << "set grid" << endl
    //    //   << "set autoscale" << endl
    //    //   << "set boxwidth 0.95 relative" << endl
    //    //   << "set style fill transparent solid 0.5" << endl
    //    //   << "plot \"-\" u 1:2 w boxes lc rgb\"red\" notitle" << endl;

    //    //bool directory_exists = chdir(dir.c_str()) == 0;
    //    //CV_Assert(directory_exists);

    //    //FILE* p = popen("gnuplot > /dev/null 2>&1", "w");
    //    //fputs(ss.str().c_str(), p);

    //    //for(int i=0; i < x.cols; i++)
    //    //{
    //    //    std::ostringstream s;
    //    //    s << x.at<float>(0, i) << " " << y.at<float>(0,i) << endl;
    //    //    fputs(s.str().c_str(), p);
    //    //}
    //    //fputs("e", p);
    //    //pclose(p);

    //    //chdir(cwd);
    //}


    //cv::Mat save_and_load_histogram_image(cv::Mat hist, FileInfo file, std::string title, bool remove_tmp_image)
    //{
    //    artelab::save_histogram_image(hist, file, title);
    //    cv::Mat image = cv::imread(file.fullName(), CV_LOAD_IMAGE_COLOR);
    //    //remove(file.fullName().c_str());
    //    return image;
    //}


    cv::Mat show_bases(RBM* rbm, cv::Size basedim, const bool rgb, cv::Size canvas)
    {
        cv::Mat res;
        cv::Mat weights = weights_image(rbm, basedim, rgb);
        cv::resize(weights, res, canvas, 0,0, cv::INTER_NEAREST);
        cv::imshow("Weights", res);
        return weights;
    }



    cv::Mat weightImageOf(RBM* rbm, const int hidden_unit_index, const cv::Size base_size, const bool rgb)
    {
        const int n_channels = rgb? 3 : 1;
        CV_Assert(hidden_unit_index >= 0);
        CV_Assert(base_size.height*base_size.width == rbm->num_visible() / n_channels);

        double min, max;
        cv::minMaxLoc(rbm->weights, &min, &max);

        cv::Mat base = rbm->weights.row(hidden_unit_index).clone().reshape(n_channels, base_size.height);
        cv::Mat weight_image(base.size(), rgb? CV_8UC3 : CV_8U);

        std::vector<cv::Mat> w_channels, out_channels;
        cv::split(base, w_channels);
        cv::split(weight_image, out_channels);


        for(size_t ch=0; ch < w_channels.size(); ch++)
        {
            cv::Mat base_ch = w_channels[ch];
            cv::Mat out_ch = out_channels[ch];
            for(int r=0; r < base_ch.rows; r++)
            {
                float* w_row = base_ch.ptr<float>(r);
                uchar* out_row = out_ch.ptr<uchar>(r);
                for(int c=0; c < base_ch.cols; c++)
                {
                    out_row[c] = cv::saturate_cast<uchar>((w_row[c] - float(min)) / (float(max) - float(min)) * 255);
                }
            }
        }

        cv::merge(out_channels, weight_image);
        return weight_image;
    }


    cv::Mat weights_image(RBM* rbm, const cv::Size weight_img_canvas, const bool rgb, const int images_column)
    {
        cv::Mat out_image;

        int num_col = images_column <= 0? int(sqrt((double)rbm->num_hidden())) : images_column;
        int num_row = rbm->num_hidden() / num_col;
        num_row = num_row*num_col < rbm->num_hidden()? num_row+1 : num_row;

        const int sep = 2;
        const int canvas_width = weight_img_canvas.width * num_col + sep * (num_col+1);
        const int canvas_height = weight_img_canvas.height * num_row + sep * (num_row+1);
        out_image = cv::Mat::zeros(canvas_height, canvas_width, rgb? CV_8UC3 : CV_8U);

        int current_row = 0, current_col = 0;
        for(int h=0; h < rbm->num_hidden(); h++)
        {
            // top-left offset coordinate in the canvas
            int y_offset = current_row * (weight_img_canvas.height + sep) + 2;
            int x_offset = current_col * (weight_img_canvas.width + sep) + 2;

            // get weightImageOf(h)
            cv::Mat w_image = weightImageOf(rbm, h, weight_img_canvas, rgb);

            // copy the image at the correct position
            cv::Mat sub = out_image.rowRange(y_offset, y_offset + weight_img_canvas.height).colRange(x_offset, x_offset + weight_img_canvas.width);
            w_image.copyTo(sub);

            // update indices
            if(++current_col == num_col)
            {
                current_col = 0;
                current_row++;
            }
        }

        return out_image;
    }

    void print_info(RBM* rbm, int epoch, cv::Mat& trainset, cv::Mat valset)
    {
    //        float mset = rbm->avgReconsturctionMse(trainset);
        float mset = average_mse(rbm, trainset);;
        float ft = rbm->avg_free_energy(trainset);

        if(valset.data)
        {
    //            float msev = rbm->avgReconsturctionMse(valset);
            float msev = average_mse(rbm, valset);
            float fv = rbm->avg_free_energy(valset);
            float diff = ft-fv;

            std::cout << epoch << " | " << mset << " | " << msev << " | " << ft << " | " 
                    << fv << " | " << (diff>0? diff : -diff) << std::endl;
        }
        else
        {
            std::cout << epoch << " | " << mset << " | N/A | " 
                 << ft << " | N/A | N/A" << std::endl;
        }
    }

    void train_and_monitor_learning(RBM* rbm)
    {
        cv::Mat train = rbm->trainset();
        cv::Mat val = rbm->valset();


        int epoch = 0;
        std::cout << "Epoch | Train mse | Val mse | Train F | Val F | diff " << std::endl;
        while(++epoch && rbm->step())
        {
            print_info(rbm, epoch, train, val);
        }
        print_info(rbm, epoch, train, val);
    }

}