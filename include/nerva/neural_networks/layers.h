// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/layers.h
/// \brief add your file description here.

#pragma once

#ifdef NERVA_ROWWISE
#include"nerva/neural_networks/layers_rowwise.h"
#else
#include"nerva/neural_networks/layers_colwise.h"
#endif

