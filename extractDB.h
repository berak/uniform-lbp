#ifndef __extractDB_onboard__
#define __extractDB_onboard__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

int readtxt( const char * fname, std::vector<std::string> & names, std::vector<int> & labels, size_t maxim=400  );

int extractDB(const std::string &txtfile, std::vector<cv::Mat> & images, cv::Mat & labels, int preproc, int maxim=400, int fixed_size=90);

void setupPersons( const std::vector<int> & labels, std::vector<std::vector<int>> & persons );

#endif // __extractDB_onboard__

