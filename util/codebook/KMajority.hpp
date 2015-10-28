#include "opencv2/opencv.hpp" 
#include "opencv2/flann.hpp" 
#include <opencv2/flann/linear_index.h>
#include <opencv2/flann/random.h>
#include <opencv2/flann/dist.h>

typedef cvflann::Hamming<uchar> HammingDistance;
typedef cvflann::LinearIndex<HammingDistance> HammingIndex;


struct KMajority 
{
    /**
     * Initializes cluster centers choosing among the data points indicated by indices.
     */
    static cv::Mat initCentroids(const cv::Mat &trainData, int numClusters) 
    {
        // Initializing variables useful for obtaining indexes of random chosen center
        std::vector<int> centers_idx(numClusters);
        randu(centers_idx, Scalar(0), Scalar(numClusters));
        //randu(centers_idx, Scalar(0), Scalar(trainData.rows));
        std::sort(centers_idx.begin(), centers_idx.end());

        // Assign centers based on the chosen indexes
        cv::Mat centroids(centers_idx.size(), trainData.cols, trainData.type());
        for (int i = 0; i < numClusters; ++i) 
        {
            trainData.row(centers_idx[i]).copyTo(centroids(cv::Range(i, i + 1), cv::Range(0, trainData.cols)));
        }
        return centroids;
    }

    /**
     * Implements majority voting scheme for cluster centers computation
     * based on component wise majority of bits from data matrix
     * as proposed by Grana2013.
     */
    static void computeCentroids(const Mat &features, Mat &centroids,
        std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo) 
    {
        // Warning: using matrix of integers, there might be an overflow when summing too much descriptors
        cv::Mat bitwiseCount(centroids.rows, features.cols * 8, CV_32S);
        // Zeroing matrix of cumulative bits
        bitwiseCount = cv::Scalar::all(0);
        // Zeroing all cluster centers dimensions
        centroids = cv::Scalar::all(0);

        // Bitwise summing the data into each center
        for (int i=0; i<features.cols; ++i) 
        {
            cv::Mat b = bitwiseCount.row(belongsTo[i]);
            KMajority::cumBitSum(features.row(i), b);
        }

        // Bitwise majority voting
        for (int j=0; j<centroids.rows; j++) 
        {
            cv::Mat centroid = centroids.row(j);
            KMajority::majorityVoting(bitwiseCount.row(j), centroid, clusterCounts[j]);
        }
    }
    /**
     * Decomposes data into bits and accumulates them into cumResult.
     *
     * @param feature- Row vector containing the data to accumulate
     * @param accVector - Row oriented accumulator vector
     */
    static void cumBitSum(const cv::Mat &feature, cv::Mat &accVector) 
    {
        // cumResult and data must be row vectors
        CV_Assert(feature.rows == 1 && accVector.rows == 1);
        // cumResult and data must be same length
        CV_Assert(feature.cols * 8 == accVector.cols);

        uchar byte = 0;
        for (int l = 0; l < accVector.cols; l++) 
        {
            // bit: 7-(l%8) col: (int)l/8 descriptor: i
            // Load byte every 8 bits
            if ((l % 8) == 0) 
            {
                byte = *(feature.col((int) l / 8).data);
            }
            // Note: ignore maybe-uninitialized warning because loop starts with l=0 that means byte gets a value as soon as the loop start
            // bit at ith position is mod(bitleftshift(byte,i),2) where ith position is 7-mod(l,8) i.e 7, 6, 5, 4, 3, 2, 1, 0
            accVector.at<int>(0, l) += ((int) ((byte >> (7 - (l % 8))) % 2));
        }
    }
   
    static void majorityVoting(const cv::Mat &accVector, cv::Mat &result, int threshold)
    {
        // cumResult and data must be row vectors
        CV_Assert(result.rows == 1 && accVector.rows == 1);
        // cumResult and data must be same length
        CV_Assert(result.cols * 8 == accVector.cols);

        // In this point I already have stored in bitwiseCount the bitwise sum of all data assigned to jth cluster
        for (int l = 0; l < accVector.cols; ++l) 
        {
            // If the bitcount for jth cluster at dimension l is greater than half of the data assigned to it
            // then set lth centroid bit to 1 otherwise set it to 0 (break ties randomly)
            bool bit;
            // There is a tie if the number of data assigned to jth cluster is even
            // AND the number of bits set to 1 in lth dimension is the half of the data assigned to jth cluster
            if ((threshold % 2 == 1) && (2 * accVector.at<int>(0, l) == (int) threshold))
            {
                bit = (bool)(rand() % 2);
            } 
            else 
            {
                bit = 2 * accVector.at<int>(0, l) > (int) (threshold);
            }
            // Stores the majority voting result from the LSB to the MSB
            result.at<unsigned char>(0, (int) (accVector.cols - 1 - l) / 8) += (bit) << ((accVector.cols - 1 - l) % 8);
        }
    }
    
    /**
     * Assigns data to clusters by means of Hamming distance.
     *
     * @return true if convergence was achieved (cluster assignment didn't changed), false otherwise
     */
    static bool quantize(cv::Ptr<HammingIndex> index, const Mat &descriptors,
        std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo, int numClusters) 
    {
        bool converged = true;

        // Number of nearest neighbors
        int knn = 1;

        // The indices of the nearest neighbors found (numQueries X numNeighbors)
        cvflann::Matrix<int> indices(new int[1 * knn], 1, knn);

        // Distances to the nearest neighbors found (numQueries X numNeighbors)
        cvflann::Matrix<int> distances(new int[1 * knn], 1, knn);

        for (int i=0; i<descriptors.rows; ++i) 
        {
            std::fill(indices.data, indices.data + indices.rows * indices.cols, 0);
            std::fill(distances.data, distances.data + distances.rows * distances.cols, 0);

            cvflann::Matrix<uchar> descriptor(descriptors.row(i).data, 1, descriptors.cols);

            // Get new cluster it belongs to 
            index->knnSearch(descriptor, indices, distances, knn, cvflann::SearchParams());

            // Check if cluster assignment changed 
            // If it did then algorithm hasn't converged yet
            if (belongsTo[i] != indices[0][0]) {
                converged = false;
            }

            // Update cluster assignment and cluster counts 
            // Decrease cluster count in case it was assigned to some valid cluster before.
            // Recall that initially all transaction are assigned to kth cluster which
            // is not valid since valid clusters run from 0 to k-1 both inclusive.
            if (belongsTo[i] != numClusters) {
                --clusterCounts[belongsTo[i]];
            }
            belongsTo[i] = indices[0][0];
            ++clusterCounts[indices[0][0]];
            distanceTo[i] = distances[0][0];
        }

        delete[] indices.data;
        delete[] distances.data;

        return converged;
    }

    /**
     * Fills empty clusters using data assigned to the most populated ones.
     */
    static void handleEmptyClusters(std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo, int numClusters, int numDatapoints)
    {
        // If some cluster appeared to be empty then:
        // 1. Find the biggest cluster.
        // 2. Find farthest point in the biggest cluster
        // 3. Exclude the farthest point from the biggest cluster and form a new 1-point cluster.

        for (int k = 0; k < numClusters; ++k) {
            if (clusterCounts[k] != 0) {
                continue;
            }

            // 1. Find the biggest cluster
            int max_k = 0;
            for (int k1 = 1; k1 < numClusters; ++k1) {
                if (clusterCounts[max_k] < clusterCounts[k1])
                    max_k = k1;
            }

            // 2. Find farthest point in the biggest cluster
            int maxDist(-1);
            int idxFarthestPt = -1;
            for (int i = 0; i < numDatapoints; ++i) {
                if (belongsTo[i] == max_k) {
                    if (maxDist < distanceTo[i]) {
                        maxDist = distanceTo[i];
                        idxFarthestPt = i;
                    }
                }
            }

            // 3. Exclude the farthest point from the biggest cluster and form a new 1-point cluster
            --clusterCounts[max_k];
            ++clusterCounts[k];
            belongsTo[idxFarthestPt] = k;
        }
    }
};
