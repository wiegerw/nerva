// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/dropout_layers.h
/// \brief add your file description here.

#pragma once

#ifdef NERVA_COLWISE
#include"nerva/neural_networks/dropout_layers_colwise.h"
#else
#include"nerva/neural_networks/dropout_layers_rowwise.h"
#endif
