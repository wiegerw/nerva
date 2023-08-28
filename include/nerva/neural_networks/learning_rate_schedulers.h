// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/learning_rate_schedulers.h
/// \brief add your file description here.

#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <regex>
#include <utility>
#include "fmt/format.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/utilities/parse.h"
#include "nerva/utilities/parse_numbers.h"
#include "nerva/utilities/print.h"
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
std::shared_ptr<learning_rate_scheduler> parse_learning_rate_scheduler(const std::string& text)
{
  auto func = utilities::parse_function_call(text);

  if (func.name == "Constant")
  {
    auto lr = func.as_scalar("lr");
    return std::make_shared<constant_scheduler>(lr);
  }
  else if (func.name == "TimeBased")
  {
    auto lr = func.as_scalar("lr");
    auto decay = func.as_scalar("decay");
    return std::make_shared<time_based_scheduler>(lr, decay);
  }
  else if (func.name == "StepBased")
  {
    auto lr = func.as_scalar("lr");
    auto drop_rate = func.as_scalar("drop_rate");
    auto change_rate = func.as_scalar("change_rate");
    return std::make_shared<step_based_scheduler>(lr, drop_rate, change_rate);
  }
  else if (func.name == "MultistepLR")
  {
    auto lr = func.as_scalar("lr");
    auto milestones = parse_comma_separated_numbers<unsigned int>(func.as_string("milestones"));
    auto gamma = func.as_scalar("gamma");
    return std::make_shared<multi_step_lr_scheduler>(lr, milestones, gamma);
  }
  else if (func.name == "Exponential")
  {
    auto lr = func.as_scalar("lr");
    auto change_rate = func.as_scalar("change_rate");
    return std::make_shared<exponential_scheduler>(lr, change_rate);
  }
  throw std::runtime_error("could not parse learning scheduler '" + text + "'");
}

} // namespace nerva

