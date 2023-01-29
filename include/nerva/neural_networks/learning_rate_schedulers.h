// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/learning_rate_schedulers.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_LEARNING_RATE_SCHEDULERS_H
#define NERVA_NEURAL_NETWORKS_LEARNING_RATE_SCHEDULERS_H

#include <cmath>
#include <iostream>
#include <memory>
#include <regex>
#include <utility>
#include "nerva/neural_networks/eigen.h"
#include "nerva/utilities/string_utility.h"

namespace nerva {

struct learning_rate_scheduler
{
  // Returns the learning rate at iteration i
  virtual scalar operator()(unsigned int i) = 0;

  virtual ~learning_rate_scheduler() = default;
};

struct constant_scheduler: public learning_rate_scheduler
{
  scalar eta; // the learning rate

  explicit constant_scheduler(scalar eta_)
      : eta(eta_)
  {}

  scalar operator()(unsigned int i) override
  {
    return eta;
  }
};

struct time_based_scheduler: public learning_rate_scheduler
{
  scalar eta; // the current value of the learning rate
  scalar decay;

  explicit time_based_scheduler(scalar eta_, scalar decay_)
   : eta(eta_), decay(decay_)
  {}

  scalar operator()(unsigned int i) override
  {
    eta = eta / (1 + decay * scalar(i));
    return eta;
  }
};

struct step_based_scheduler: public learning_rate_scheduler
{
  scalar eta0; // the initial value of the learning rate
  scalar d;    // the change rate
  scalar r;    // the drop rate

  explicit step_based_scheduler(scalar eta, scalar drop_rate, scalar change_rate)
      : eta0(eta), d(drop_rate), r(change_rate)
  {}

  scalar operator()(unsigned int i) override
  {
    return eta0 * std::pow(d, std::floor((1.0 + i) / r));
  }
};

struct multi_step_lr_scheduler: public learning_rate_scheduler
{
  scalar eta0; // the initial value of the learning rate
  std::vector<unsigned int> milestones; // an increasing list of epoch indices
  scalar gamma; // the multiplicative factor of the decay

  explicit multi_step_lr_scheduler(scalar eta, std::vector<unsigned int> milestones_, scalar gamma_)
   : eta0(eta), milestones(std::move(milestones_)), gamma(gamma_)
  {}

  scalar operator()(unsigned int i) override
  {
    scalar eta = eta0;
    for (unsigned int milestone: milestones)
    {
      if (i >= milestone)
      {
        eta *= gamma;
      }
      else
      {
        break;
      }
    }
    return eta;
  }
};

struct exponential_scheduler: public learning_rate_scheduler
{
  scalar eta0; // the initial value of the learning rate
  scalar d;    // the change rate

  explicit exponential_scheduler(scalar eta, scalar change_rate)
      : eta0(eta), d(change_rate)
  {}

  scalar operator()(unsigned int i) override
  {
    return eta0 * std::exp(-d * scalar(i));
  }
};

inline
std::shared_ptr<learning_rate_scheduler> parse_constant_scheduler(const std::string& text)
{
  std::regex re{R"(constant\((.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("Error: could not parse learning scheduler '" + text + "'");
  }
  scalar value = std::stod(m[1]);
  return std::make_shared<constant_scheduler>(value);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_time_based_scheduler(const std::string& text)
{
  std::regex re{R"(time_based\((.*?),(.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("Error: could not parse learning scheduler '" + text + "'");
  }
  scalar eta = std::stod(m[1]);
  scalar decay = std::stod(m[2]);
  return std::make_shared<time_based_scheduler>(eta, decay);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_step_based_scheduler(const std::string& text)
{
  std::regex re{R"(step_based\((.*?),(.*?),(.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("Error: could not parse learning scheduler '" + text + "'");
  }
  scalar eta = std::stod(m[1]);
  scalar drop_rate = std::stod(m[2]);
  scalar change_rate = std::stod(m[3]);
  return std::make_shared<step_based_scheduler>(eta, drop_rate, change_rate);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_exponential_scheduler(const std::string& text)
{
  std::regex re{R"(exponential\((.*?),(.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("Error: could not parse learning scheduler '" + text + "'");
  }
  scalar eta = std::stod(m[1]);
  scalar change_rate = std::stod(m[2]);
  return std::make_shared<exponential_scheduler>(eta, change_rate);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_learning_rate_scheduler(const std::string& text)
{
  if (utilities::starts_with(text, "constant"))
  {
    return parse_constant_scheduler(text);
  }
  else if (utilities::starts_with(text, "time_based"))
  {
    return parse_time_based_scheduler(text);
  }
  else if (utilities::starts_with(text, "step_based"))
  {
    return parse_step_based_scheduler(text);
  }
  else if (utilities::starts_with(text, "exponential"))
  {
    return parse_exponential_scheduler(text);
  }
  throw std::runtime_error("Error: could not parse learning scheduler '" + text + "'");
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LEARNING_RATE_SCHEDULERS_H
