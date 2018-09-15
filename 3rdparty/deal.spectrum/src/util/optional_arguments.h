/*
 * <DEAL.SPECTRUM>/util/optional_arguments.h
 *
 *  Created on: Mar 02, 2018
 *      Author: muench
 */

#ifndef DEAL_SPECTRUM_OPTIONAL_ARGUMENTS
#define DEAL_SPECTRUM_OPTIONAL_ARGUMENTS

#include <map>
#include <string>

namespace dealspectrum
{
/**
 * Process optional command line arguments of the following format:
 *      -NAME VALUE
 * The value is put into a map with the name being the key. The argument count
 * is accordingly reduced and the pointer to the first element is accordingly
 * moved.
 *
 * @param argc  count of arguments
 * @param argv  value of arguments
 * @return map of optional arguments and their values
 */
std::map<std::string, std::string>
processOptionalArguments(int & argc, char **& argv)
{
  // create empty map
  std::map<std::string, std::string> map;

  // skip first argument (it is the program name)
  argv++;

  // loop over all command line arguments
  for(int i = 1; i < argc; i++)
  {
    if(argv[0][0] == '-' && argv[0][1] == '-')
    {
      // optional flag (--FLAG) found: put into map...
      std::string str1(argv[0] + 2);
      map[str1] = "true";
      // ... reduce the argument count and move pointer to the first argument
      argc -= 1;
      argv += 1;
    }
    else if(argv[0][0] == '-')
    {
      // optional argument (--LABEL VALUE) found: put into map...
      std::string str1(argv[0] + 1);
      std::string str2(argv[1] + 0);
      map[str1] = str2;
      // ... reduce the argument count and move pointer to the first argument
      argc -= 2;
      argv += 2;
    }
    else
    {
      break;
    }
  }

  // move pointer one back s.t. first command line argument has index 1
  argv--;

  // return map
  return map;
}

} // namespace dealspectrum

#endif
