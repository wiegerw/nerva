// Author(s): Maurice Laveaux, Wieger Wesselink
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef NERVA_UTILITIES_STOPWATCH_H
#define NERVA_UTILITIES_STOPWATCH_H

#include <chrono>
#include <iostream>
#include <utility>

namespace nerva::utilities {

/// \brief Implements a simple stopwatch that starts on construction.
class stopwatch
{
  public:
    stopwatch()
    {
      reset();
    }

    /// \brief Reset the stopwatch to count from this moment onwards.
    void reset()
    {
      m_timestamp = std::chrono::steady_clock::now();
    }

    /// \returns The time in milliseconds since the last reset.
    [[nodiscard]] long long time() const
    {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - m_timestamp).count();
    }

    /// \returns The time in seconds since the last reset.
    [[nodiscard]] double seconds() const
    {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - m_timestamp).count();
    }

  private:
    std::chrono::time_point<std::chrono::steady_clock> m_timestamp;
};

} // namespace nerva::utilities

#endif // NERVA_UTILITIES_STOPWATCH_H
