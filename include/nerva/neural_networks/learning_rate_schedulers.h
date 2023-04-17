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
#include "fmt/format.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/string_utility.h"

namespace nerva {

struct learning_rate_scheduler
{
  // Returns the learning rate at iteration i
  virtual scalar operator()(unsigned int i) = 0;

  [[nodiscard]] virtual std::string to_string() const = 0;

  virtual ~learning_rate_scheduler() = default;
};

struct constant_scheduler: public learning_rate_scheduler
{
  scalar lr; // the learning rate

  explicit constant_scheduler(scalar lr_)
      : lr(lr_)
  {}

  scalar operator()(unsigned int i) override
  {
    return lr;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("ConstantScheduler(lr={})", lr);
  }
};

struct time_based_scheduler: public learning_rate_scheduler
{
  scalar lr; // the current value of the learning rate
  scalar decay;

  explicit time_based_scheduler(scalar lr_, scalar decay_)
   : lr(lr_), decay(decay_)
  {}

  scalar operator()(unsigned int i) override
  {
    lr = lr / (1 + decay * scalar(i));
    return lr;
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("TimeBasedScheduler(lr={}, decay={})", lr, decay);
  }
};

struct step_based_scheduler: public learning_rate_scheduler
{
  scalar lr; // the initial value of the learning rate
  scalar drop_rate;
  scalar change_rate;

  explicit step_based_scheduler(scalar lr_, scalar drop_rate_, scalar change_rate_)
      : lr(lr_), drop_rate(drop_rate_), change_rate(change_rate_)
  {}

  scalar operator()(unsigned int i) override
  {
    return lr * std::pow(drop_rate, std::floor((1.0 + i) / change_rate));
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("StepBasedScheduler(lr={}, drop_rate={}, change_rate={})", lr, drop_rate, change_rate);
  }
};

struct multi_step_lr_scheduler: public learning_rate_scheduler
{
  scalar lr; // the initial value of the learning rate
  std::vector<unsigned int> milestones; // an increasing list of epoch indices
  scalar gamma; // the multiplicative factor of the decay

  explicit multi_step_lr_scheduler(scalar lr_, std::vector<unsigned int> milestones_, scalar gamma_)
   : lr(lr_), milestones(std::move(milestones_)), gamma(gamma_)
  {}

  scalar operator()(unsigned int i) override
  {
    scalar eta = lr;
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

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("MultiStepLRScheduler(lr={}, milestones={}, gamma={})", lr, print_list(milestones), gamma);
  }
};

struct exponential_scheduler: public learning_rate_scheduler
{
  scalar lr;           // the initial value of the learning rate
  scalar change_rate;

  exponential_scheduler(scalar lr_, scalar change_rate_)
      : lr(lr_), change_rate(change_rate_)
  {}

  scalar operator()(unsigned int i) override
  {
    return lr * std::exp(-change_rate * scalar(i));
  }

  [[nodiscard]] std::string to_string() const override
  {
    return fmt::format("ExponentialScheduler(lr={}, change_rate={})", lr, change_rate);
  }
};

inline
std::shared_ptr<learning_rate_scheduler> parse_constant_scheduler(const std::string& text)
{
  std::regex re{R"(Constant\((.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("could not parse learning scheduler '" + text + "'");
  }
  scalar value = parse_scalar(m[1]);
  return std::make_shared<constant_scheduler>(value);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_multistep_lr_scheduler(const std::string& text)
{
  std::regex re{R"(MultistepLR\((.*?);(.*?);(.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("could not parse learning scheduler '" + text + "'");
  }
  scalar lr = parse_scalar(m[1]);
  std::vector<unsigned int> milestones = parse_comma_separated_numbers<unsigned int>(m[2]);
  scalar gamma = parse_scalar(m[3]);
  return std::make_shared<multi_step_lr_scheduler>(lr, milestones, gamma);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_time_based_scheduler(const std::string& text)
{
  std::regex re{R"(TimeBased\((.*?),(.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("could not parse learning scheduler '" + text + "'");
  }
  scalar lr = parse_scalar(m[1]);
  scalar decay = parse_scalar(m[2]);
  return std::make_shared<time_based_scheduler>(lr, decay);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_step_based_scheduler(const std::string& text)
{
  std::regex re{R"(StepBased\((.*?),(.*?),(.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("could not parse learning scheduler '" + text + "'");
  }
  scalar lr = parse_scalar(m[1]);
  scalar drop_rate = parse_scalar(m[2]);
  scalar change_rate = parse_scalar(m[3]);
  return std::make_shared<step_based_scheduler>(lr, drop_rate, change_rate);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_exponential_scheduler(const std::string& text)
{
  std::regex re{R"(exponential\((.*?),(.*?)\))"};
  std::smatch m;
  bool result = std::regex_match(text, m, re);
  if (!result)
  {
    throw std::runtime_error("could not parse learning scheduler '" + text + "'");
  }
  scalar lr = parse_scalar(m[1]);
  scalar change_rate = parse_scalar(m[2]);
  return std::make_shared<exponential_scheduler>(lr, change_rate);
}

inline
std::shared_ptr<learning_rate_scheduler> parse_learning_rate_scheduler(const std::string& text)
{
  if (utilities::starts_with(text, "Constant"))
  {
    return parse_constant_scheduler(text);
  }
  else if (utilities::starts_with(text, "MultistepLR"))
  {
    return parse_multistep_lr_scheduler(text);
  }
  else if (utilities::starts_with(text, "TimeBased"))
  {
    return parse_time_based_scheduler(text);
  }
  else if (utilities::starts_with(text, "StepBased"))
  {
    return parse_step_based_scheduler(text);
  }
  else if (utilities::starts_with(text, "Exponential"))
  {
    return parse_exponential_scheduler(text);
  }
  throw std::runtime_error("could not parse learning scheduler '" + text + "'");
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_LEARNING_RATE_SCHEDULERS_H
