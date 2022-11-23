// Copyright: Wieger Wesselink 2021
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/utilities/print.h
/// \brief add your file description here.

#ifndef NERVA_UTILITIES_PRINT_H
#define NERVA_UTILITIES_PRINT_H

#include <sstream>
#include <string>

namespace nerva {

/// \brief Creates a string representation of a container.
/// \param v A container
template <typename Container>
void print_container(std::ostream& out, const Container& v, const std::string& begin_marker = "", const std::string& end_marker = "", const std::string& separator = " ")
{
  out << begin_marker;
  for (auto i = v.begin(); i != v.end(); ++i)
  {
    if (i != v.begin())
    {
      out << separator;
    }
    out << *i;
  }
  out << end_marker;
}

/// \brief Creates a string representation of a container.
/// \param v A container
template <typename Container>
std::string print_container(const Container& v, const std::string& begin_marker = "", const std::string& end_marker = "", const std::string& separator = " ")
{
  std::ostringstream out;
  out << begin_marker;
  for (auto i = v.begin(); i != v.end(); ++i)
  {
    if (i != v.begin())
    {
      out << separator;
    }
    out << *i;
  }
  out << end_marker;
  return out.str();
}

/// \brief Creates a string representation of a container.
/// \param v A container
template <typename Container>
std::string print_list(const Container& v, const std::string& separator = ", ")
{
  return print_container(v, "[", "]", separator);
}

/// \brief Creates a string representation of a container.
/// \param v A container
template <typename Container>
std::string print_set(const Container& v, const std::string& separator = ", ")
{
  return print_container(v, "{", "}", separator);
}

/// \brief Creates a string representation of a matrix.
/// \param v A matrix (e.g. vector<vector<double>>)
template <typename Matrix>
std::string print_matrix(const Matrix& m)
{
  std::string begin_marker = "[";
  std::string end_marker = "]";
  std::ostringstream out;
  out << begin_marker << "\n";
  for (auto i = m.begin(); i != m.end(); ++i)
  {
    if (i != m.begin())
    {
      out << "\n";
    }
    out << "  " << print_list(*i);
  }
  out << "\n" << end_marker;
  return out.str();
}

} // namespace nerva

#endif // NERVA_UTILITIES_PRINT_H
