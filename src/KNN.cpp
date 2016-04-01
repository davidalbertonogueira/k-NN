#include "KNN.h"
#include <algorithm>
#include <map>
#include <unordered_map>

template<class THash>
typename THash::iterator IncrementHashCount(const typename THash::key_type &key,
                                            THash *hash) {
  static_assert(std::is_integral<THash::mapped_type>::value,
                "Only value integral types are support");
  auto it = hash->find(key);
  if (it == hash->end())
    it = hash->emplace(key, 1).first;
  else
    ++(it->second);
  return it;
}

bool KNN::AddInstance(const Instance & instance) {
  instances.push_back(instance);
  return true;
}

bool KNN::Classify(uint32_t k,
                   bool(*distance_function)(const std::vector<double> &,
                                            const std::vector<double> &,
                                            double *),
                   const std::vector<double> & new_instance_features,
                   uint32_t* class_membership) {
  //Maps distance to current document from nearest neighbors to their classes
  std::multimap<double, uint32_t> closest_neighbors;

  //Select the k nearest neighbors
  for (const auto & instance : instances) {
    double dist;
    if (!distance_function(instance.features, new_instance_features, &dist))
      return false;

    if (closest_neighbors.size() < k) {
      closest_neighbors.insert({ dist, instance.class_membership });
    } else {
      auto & last_pair = closest_neighbors.rbegin();
      double worst_dist_from_nn = last_pair->first;
      if (dist < worst_dist_from_nn) {
        closest_neighbors.erase(std::next(last_pair).base());
        closest_neighbors.insert({ dist, instance.class_membership });
      }
    }
  }
  //Find what is the most common class of the k nearest neighbors
  // For each class, count number of occurrences
  // Select maximum value of occurrences, and corresponding class
  std::unordered_map<uint32_t, size_t> class_counts;
  for (const auto & element : closest_neighbors) {
    IncrementHashCount(element.second, &class_counts);
  }

  uint32_t selected_class = class_counts.begin()->first;
  size_t max_count = class_counts.begin()->second;
  for (const auto & element : class_counts) {
    if (element.second >max_count) {
      max_count = element.second;
      selected_class = element.first;
    }
  }
  *class_membership = selected_class;
  return true;
};
