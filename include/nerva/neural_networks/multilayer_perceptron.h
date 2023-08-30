// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/multilayer_perceptron.h
/// \brief add your file description here.

#pragma once

#ifdef NERVA_COLWISE
#include"nerva/neural_networks/multilayer_perceptron_colwise.h"
#else
#include"nerva/neural_networks/multilayer_perceptron_rowwise.h"
#endif
