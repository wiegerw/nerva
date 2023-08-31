// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/parse_layer.h
/// \brief add your file description here.

#pragma once

// supported layers:
//
// Linear
// Sigmoid
// ReLU
// Softmax
// LogSoftmax
// HyperbolicTangent
// AllRelu(<alpha>)
// LeakyRelu(<alpha>)
// TReLU(<epsilon>)
// SReLU(<al>,<tl>,<ar>,<tr>)
//
// BatchNorm
// Dropout(<rate>)

#ifdef NERVA_COLWISE
#include"nerva/neural_networks/parse_layer_colwise.h"
#else
#include"nerva/neural_networks/parse_layer_rowwise.h"
#endif
