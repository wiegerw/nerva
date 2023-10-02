// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/utilities/timer.h
/// \brief add your file description here.

#ifndef NERVA_UTILITIES_TIMER_H
#define NERVA_UTILITIES_TIMER_H

#include "nerva/utilities/print.h"
#include "fmt/format.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <utility>

namespace nerva::utilities {

/// \brief Timer class with a map interface. For each key a start and stop value is stored.
class map_timer
{
  using time = std::chrono::time_point<std::chrono::steady_clock>;

  protected:
    std::map<std::string, std::vector<std::pair<time, time>>> m_values;
    bool m_report_on_destruction;

  public:
    explicit map_timer(bool report_on_destruction = false)
     : m_report_on_destruction(report_on_destruction)
    {}

    ~map_timer()
    {
      if (m_report_on_destruction)
      {
        double total_time = 0;
        std::cout << "--- timing results ---" << std::endl;
        for (const auto& [key, value]: m_values)
        {
          auto seconds = total_seconds(key);
          total_time += seconds;
          std::cout << fmt::format("{:20} = {:.4f}", key, seconds) << std::endl;
        }
        std::cout << fmt::format("{:20} = {:.4f}", "TOTAL TIME", total_time) << std::endl;
      }
    }

    /// \brief Sets the start value for the given key to the current time
    void start(const std::string& key)
    {
      auto& values = m_values[key];
      auto& [t1, t2] = values.emplace_back();
      t1 = std::chrono::steady_clock::now();
    }

    /// \brief Sets the stop value for the given key to the current time
    /// \return The time difference in seconds
    double stop(const std::string& key)
    {
      auto t = std::chrono::steady_clock::now();
      auto& [t1, t2] = m_values[key].back();
      t2 = t;
      return std::chrono::duration<double>(t2 - t1).count();
    }

    /// \returns The time difference for the given key in milliseconds
    [[nodiscard]] long long milliseconds(const std::string& key) const
    {
      const auto& [t1, t2] = m_values.at(key).back();
      return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }

    /// \returns The time difference for the given key in seconds
    [[nodiscard]] double seconds(const std::string& key) const
    {
      const auto& [t1, t2] = m_values.at(key).back();
      return std::chrono::duration<double>(t2 - t1).count();
    }

    /// \returns The time difference for the given key in seconds
    [[nodiscard]] double total_seconds(const std::string& key) const
    {
      double result = 0;
      for (const auto& [t1, t2]: m_values.at(key))
      {
        result += std::chrono::duration<double>(t2 - t1).count();
      }
      return result;
    }

    /// \brief Returns the mapping with timer values
    [[nodiscard]] const std::vector<std::pair<time, time>>& values(const std::string& key) const
    {
      return m_values.at(key);
    }
};

} // namespace nerva::utilities

#endif // NERVA_UTILITIES_TIMER_H
