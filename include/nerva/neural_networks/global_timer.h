// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/global_timer.h
/// \brief add your file description here.

#pragma once

#include "nerva/utilities/stopwatch.h"
#include "fmt/format.h"

namespace nerva {

enum class timer_status
{
  disabled,
  active,
  suspended
};

inline utilities::stopwatch global_timer;
inline std::size_t global_timer_index = 0;
inline timer_status global_timer_status = timer_status::disabled;

inline
void global_timer_enable()
{
  if (global_timer_status == timer_status::disabled)
  {
    global_timer_status = timer_status::active;
  }
}

inline
void global_timer_disable()
{
  global_timer_status = timer_status::disabled;
}

inline
void global_timer_suspend()
{
  if (global_timer_status != timer_status::disabled)
  {
    global_timer_status = timer_status::suspended;
  }
}

inline
void global_timer_resume()
{
  if (global_timer_status != timer_status::disabled)
  {
    global_timer_status = timer_status::active;
  }
}

inline
void global_timer_reset()
{
  if (global_timer_status != timer_status::disabled)
  {
    global_timer.reset();
  }
}

inline
void global_timer_display(const std::string& msg)
{
  if (global_timer_status == timer_status::active)
  {
    auto s = global_timer.seconds();
    std::cout << fmt::format("{}-{} {:.6f}s", msg, global_timer_index++, s) << std::endl;
  }
}

} // namespace nerva

