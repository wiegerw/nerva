// Copyright: Wieger Wesselink 2021
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/utilities/command_line_tool.h
/// \brief add your file description here.

#ifndef NERVA_UTILITIES_COMMAND_LINE_TOOL_H
#define NERVA_UTILITIES_COMMAND_LINE_TOOL_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <lyra/lyra.hpp>
#include "nerva/utilities/logger.h"

namespace nerva {

class command_line_tool
{
  protected:
    bool m_show_help = false;
    bool m_verbose = false;
    bool m_debug = false;
    lyra::cli m_cli{};

    virtual void add_options(lyra::cli& cli)
    {}

    virtual bool run() = 0;

    virtual std::string description() const
    {
      return "";
    }

  public:
    int execute(int argc, const char** argv)
    {
      m_cli.add_argument(lyra::help(m_show_help).description(this->description()));
      m_cli.add_argument(lyra::opt(m_verbose)["--verbose"]["-v"]("Show verbose output."));
      m_cli.add_argument(lyra::opt(m_debug)["--debug"]["-d"]("Show debug output."));

      try
      {
        add_options(m_cli);
        auto parse_result = m_cli.parse({argc, argv});
        if (m_show_help)
        {
          std::cout << m_cli;
          return EXIT_SUCCESS;
        }
        if (!parse_result)
        {
          std::cerr << parse_result.message() << "\n";
        }
        if (m_debug)
        {
          log::nerva_logger::set_reporting_level(log::debug);
        }
        else if (m_verbose)
        {
          log::nerva_logger::set_reporting_level(log::verbose);
        }

        bool result = run();
        return result ? EXIT_SUCCESS : EXIT_FAILURE;
      }
      catch (const std::exception& e)
      {
        std::cout << "Error: " << e.what() << std::endl;
      }
      return EXIT_FAILURE;
    }
};

} // namespace nerva

#endif // NERVA_UTILITIES_COMMAND_LINE_TOOL_H
