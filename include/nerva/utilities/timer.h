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
    std::map<std::string, std::pair<time, time>> m_values;

  public:
    /// \brief Sets the start value for the given key to the current time
    void start(const std::string& key)
    {
      auto& [t1, t2] = m_values[key];
      t1 = std::chrono::steady_clock::now();
    }

    /// \brief Sets the stop value for the given key to the current time
    void stop(const std::string& key)
    {
      auto t = std::chrono::steady_clock::now();
      auto& [t1, t2] = m_values[key];
      t2 = t;
    }

    /// \brief Sets the stop value for the given key, and returns the time passed since the start
    /// \returns The time difference for the given key in milliseconds
    [[nodiscard]] long long milliseconds(const std::string& key)
    {
      auto t = std::chrono::steady_clock::now();
      auto& [t1, t2] = m_values[key];
      t2 = t;
      return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }

    /// \brief Sets the stop value for the given key, and returns the time passed since the start
    /// \returns The time difference for the given key in seconds
    [[nodiscard]] double seconds(const std::string& key)
    {
      auto t = std::chrono::steady_clock::now();
      auto& [t1, t2] = m_values[key];
      t2 = t;
      return std::chrono::duration<double>(t2 - t1).count();
    }

    /// \brief Returns the mapping with timer values
    const std::map<std::string, std::pair<time, time>>& values() const
    {
      return m_values;
    }

    /// \brief Returns the mapping with timer values
    std::map<std::string, std::pair<time, time>>& values()
    {
      return m_values;
    }
};

} // namespace nerva::utilities

#endif // NERVA_UTILITIES_TIMER_H
