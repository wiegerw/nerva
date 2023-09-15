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

#include "nerva/utilities/logger.h"
#include "nerva/utilities/string_utility.h"
#include <lyra/lyra.hpp>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>

namespace nerva {

class command_line_tool
{
  protected:
    bool m_show_help = false;
    bool m_verbose = false;
    bool m_debug = false;
    std::string m_command_line_call;
    lyra::cli m_cli{};

    virtual void add_options(lyra::cli& cli)
    {}

    virtual bool run() = 0;

    virtual std::string description() const
    {
      return "";
    }

    static std::string quote(const std::string& text)
    {
      std::regex unsafe_characters("[^\\w@%+=:,./-]");

      auto contains_special_characters = [](const std::string& str)
      {
        const std::string special_characters = "^@%+=:;,./-\"'";
        return std::any_of(special_characters.begin(), special_characters.end(), [&str](char c) { return str.find(c) != std::string::npos; });
      };

      if (contains_special_characters(text))
      {
        if (text.find('"') == std::string::npos)
        {
          return "\"" + text + "\"";
        }
        else if (text.find('\'') == std::string::npos)
        {
          return "'" + text + "'";
        }
        else
        {
          return "'" + std::regex_replace(text, unsafe_characters, "'\"'\"'") + "'";
        }
      }
      return text;
    }

    // Does an attempt to print the original command line call.
    // TODO: handle nested quotes
    std::string reconstruct_command_line_call(int argc, const char** argv)
    {
      std::ostringstream out;
      for (int i = 0; i < argc; i++)
      {
        std::vector<std::string> words = utilities::regex_split(argv[i], "=");
        if (words.size() == 1)
        {
          out << words[0];
        }
        else
        {
          out << words[0] << '=' << quote(words[1]);
        }
        if (i < argc - 1)
        {
          out << " ";
        }
      }
      return out.str();
    }

  public:
    int execute(int argc, const char** argv)
    {
      m_command_line_call = reconstruct_command_line_call(argc, argv);
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

    bool is_verbose() const
    {
      return m_verbose;
    }

    bool is_debug() const
    {
      return m_debug;
    }

    const std::string& command_line_call() const
    {
      return m_command_line_call;
    }

    virtual ~command_line_tool() = default;
};

} // namespace nerva

#endif // NERVA_UTILITIES_COMMAND_LINE_TOOL_H
