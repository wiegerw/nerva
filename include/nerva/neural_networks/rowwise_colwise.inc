// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/rowwise_colwise.h
/// \brief add your file description here.

#pragma once

#if defined NERVA_COLWISE
namespace nerva {
  namespace colwise {}   // to avoid compilation errors
  using namespace colwise;
#define NERVA_ORIENTATION colwise
}
#else
namespace nerva {
  namespace rowwise {}   // to avoid compilation errors
  using namespace rowwise;
#define NERVA_ORIENTATION rowwise
}
#endif