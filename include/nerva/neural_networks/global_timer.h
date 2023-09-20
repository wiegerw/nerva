// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/global_timer.h
/// \brief add your file description here.

#pragma once

#include "nerva/utilities/timer.h"
#include "fmt/format.h"

namespace nerva {

enum class timer_status
{
  disabled,
  active,
  suspended
};

inline utilities::map_timer global_timer;
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
void global_timer_start(const std::string& key)
{
  if (global_timer_status == timer_status::active)
  {
    global_timer.start(key);
  }
}

inline
void global_timer_stop(const std::string& key)
{
  if (global_timer_status == timer_status::active)
  {
    double s = global_timer.stop(key);
    auto index = global_timer.values(key).size();
    std::cout << fmt::format("{:>15}-{:<4} {:.6f}s", key, index, s) << std::endl;
  }
}

#ifdef NERVA_TIMING
#define GLOBAL_TIMER_START(name) global_timer_start(name);
#define GLOBAL_TIMER_STOP(name) global_timer_stop(name);
#else
#define GLOBAL_TIMER_START(name)
#define GLOBAL_TIMER_STOP(name)
#endif

} // namespace nerva

