// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/utilities/text_utility.h
/// \brief add your file description here.

#ifndef NERVA_UTILITIES_TEXT_UTILITY_H
#define NERVA_UTILITIES_TEXT_UTILITY_H

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

namespace nerva {

inline
std::string read_text(const std::string& filename)
{
  std::ifstream in(filename.c_str());
  if (!in)
  {
    throw std::runtime_error("Could not open input file: " + filename);
  }
  in.unsetf(std::ios::skipws); //  Turn of white space skipping on the stream

  std::string text;
  std::copy(std::istream_iterator<char>(in), std::istream_iterator<char>(), std::back_inserter(text));
  return text;
}

inline
std::string read_text_fast(const std::string& filename)
{
  std::ifstream in(filename, std::ios::binary);
  if (!in)
  {
    throw std::runtime_error("Could not read file " + filename);
  }
  in.seekg(0, std::ios::end);
  std::string text(in.tellg(), 0);
  in.seekg(0);
  in.read(text.data(), text.size());
  return text;
}

/// \brief Saves text to the file filename, or to stdout if filename equals "-".
inline
void write_text(const std::string& filename, const std::string& text)
{
  if (filename.empty())
  {
    std::cout << text;
  }
  else
  {
    std::ofstream to(filename);
    if (!to.good())
    {
      throw std::runtime_error("Could not write to filename " + filename);
    }
    to << text;
  }
}

/// \brief Replaces newline characters by spaces
inline
std::string remove_newlines(const std::string& text)
{
  const std::regex re(R"(\n)");
  std::smatch m;
  return std::regex_replace(text, re, " ");
}

// See https://en.cppreference.com/w/cpp/types/byte
// See https://stackoverflow.com/questions/68994331/why-is-there-no-overload-for-printing-stdbyte
inline
std::vector<std::byte> read_binary_file(const std::string& filename)
{
  std::ifstream from(filename, std::ios::binary | std::ios::ate);
  if (!from)
  {
    throw std::runtime_error("Could not open file " + filename);
  }

  std::streamsize size = from.tellg();
  from.seekg(0, std::ios::beg);

  std::vector<std::byte> buffer(size);
  auto data = reinterpret_cast<char*>(buffer.data());
  if (from.read(data, size))
  {
    return buffer;
  }

  throw std::runtime_error("Could not read file " + filename);
}

} // namespace nerva

#endif // NERVA_UTILITIES_TEXT_UTILITY_H
