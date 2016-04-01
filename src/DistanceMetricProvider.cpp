#include "DistanceMetricProvider.h"
#include <cmath>
#include <algorithm>

bool DistanceMetricProvider::EuclideanDistance(const std::vector<double> & a,
                                               const std::vector<double> & b,
                                               double *result) {
  size_t size = a.size();
  if (size != b.size())
    return false;
  double sum = 0.0;
  for (size_t i = 0; i < size; i++) {
    double diff = a[i] - b[i];
    sum += diff*diff;
  }
  *result = std::sqrt(sum);
  return true;
};

bool DistanceMetricProvider::EuclideanSquaredDistance(const std::vector<double> & a,
                                                      const std::vector<double> & b,
                                                      double *result) {
  size_t size = a.size();
  if (size != b.size())
    return false;
  double sum = 0.0;
  for (size_t i = 0; i < size; i++) {
    double diff = a[i] - b[i];
    sum += diff*diff;
  }
  *result = sum;
  return true;
};
bool DistanceMetricProvider::CityBlockDistance(const std::vector<double> & a,
                                               const std::vector<double> & b,
                                               double *result) {
  size_t size = a.size();
  if (size != b.size())
    return false;
  double sum = 0.0;
  for (size_t i = 0; i < size; i++) {
    double diff = a[i] - b[i];
    sum += std::abs(diff);
  }
  *result = sum;
  return true;
};
bool DistanceMetricProvider::ChebyshevDistance(const std::vector<double> & a,
                                               const std::vector<double> & b,
                                               double *result) {
  size_t size = a.size();
  if (size != b.size())
    return false;
  double cheb_dist = 0.0;
  for (size_t i = 0; i < size; i++) {
    double abs_diff = std::abs(a[i] - b[i]);
    if (abs_diff > cheb_dist)
      cheb_dist = abs_diff;
  }
  *result = cheb_dist;
  return true;
};
