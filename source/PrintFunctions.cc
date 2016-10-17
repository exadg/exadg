/*
 * PrintFunctions.cc
 *
 *  Created on: Aug 19, 2016
 *      Author: fehn
 */

#include <iomanip>

#include "../include/PrintFunctions.h"

void print_name(ConditionalOStream &pcout,
                const std::string  name)
{
  const unsigned int max_length_name = 45;

  pcout << "  " /* 2 */ << name  /* name.lenght*/ << ":" /* 1 */;
  const int remaining_spaces = max_length_name - 2 - 1 - name.length();

  for(int i=0; i<remaining_spaces; ++i)
    pcout << " ";
}

template<typename ParameterType>
void print_value(ConditionalOStream  &pcout,
                 const ParameterType value)
{
  pcout << value << std::endl;
}

// specialization of template function for parameters of type bool
template<>
void print_value(ConditionalOStream &pcout,
                 const bool         value)
{
  std::string value_string = "test";
  value_string = (value == true) ? "true" : "false";
  print_value(pcout,value_string);
}

// specialization of template function for parameters of type double
template<>
void print_value(ConditionalOStream &pcout,
                 const double       value)
{
  pcout << std::scientific << std::setprecision(4) << value << std::endl;
}

template<typename ParameterType>
void print_parameter(ConditionalOStream  &pcout,
                     const std::string   name,
                     const ParameterType value)
{
  print_name(pcout, name);
  print_value(pcout, value);
}


// explicit instantiation of template functions

template void print_parameter<std::string> (ConditionalOStream &pcout,
                                            const std::string  name,
                                            const std::string  value);

template void print_parameter<bool> (ConditionalOStream &pcout,
                                     const std::string  name,
                                     const bool         value);


template void print_parameter<int> (ConditionalOStream &pcout,
                                    const std::string  name,
                                    const int          value);

template void print_parameter<unsigned int> (ConditionalOStream &pcout,
                                             const std::string  name,
                                             const unsigned int value);


template void print_parameter<double> (ConditionalOStream &pcout,
                                       const std::string  name,
                                       const double       value);

//required on SuperMUC
template void print_parameter<unsigned long long> (ConditionalOStream &pcout,
                                    const std::string  name,
                                    const unsigned long long          value);

template void print_value<std::string> (ConditionalOStream &pcout,
                                        const std::string  value);


template void print_value<int> (ConditionalOStream &pcout,
                                const int          value);

template void print_value<unsigned int> (ConditionalOStream &pcout,
                                         const unsigned int value);

