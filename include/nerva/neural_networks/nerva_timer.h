// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/nerva_timer.h
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

inline utilities::map_timer nerva_timer;
inline timer_status nerva_timer_status = timer_status::disabled;

inline
void nerva_timer_enable()
{
  if (nerva_timer_status == timer_status::disabled)
  {
    nerva_timer_status = timer_status::active;
  }
}

inline
void nerva_timer_disable()
{
  nerva_timer_status = timer_status::disabled;
}

inline
void nerva_timer_suspend()
{
  if (nerva_timer_status != timer_status::disabled)
  {
    nerva_timer_status = timer_status::suspended;
  }
}

inline
void nerva_timer_resume()
{
  if (nerva_timer_status != timer_status::disabled)
  {
    nerva_timer_status = timer_status::active;
  }
}

inline
void nerva_timer_start(const std::string& key)
{
  if (nerva_timer_status == timer_status::active)
  {
    nerva_timer.start(key);
  }
}

inline
void nerva_timer_stop(const std::string& key)
{
  if (nerva_timer_status == timer_status::active)
  {
    double s = nerva_timer.stop(key);
    auto index = nerva_timer.values(key).size();
    std::cout << fmt::format("{:>15}-{:<4} {:.6f}s", key, index, s) << std::endl;
  }
}

#ifdef NERVA_TIMER
#define NERVA_TIMER_START(name) nerva_timer_start(name);
#define NERVA_TIMER_STOP(name) nerva_timer_stop(name);
#else
#define NERVA_TIMER_START(name)
#define NERVA_TIMER_STOP(name)
#endif

} // namespace nerva

