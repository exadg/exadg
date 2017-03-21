/*
 * PrintFunctions.h
 *
 *  Created on: Aug 19, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_PRINTFUNCTIONS_H_
#define INCLUDE_FUNCTIONALITIES_PRINTFUNCTIONS_H_

#include <deal.II/base/conditional_ostream.h>

using namespace dealii;

template<typename ParameterType>
void print_parameter(ConditionalOStream  &pcout,
                     const std::string   name,
                     const ParameterType value);

void print_name(ConditionalOStream &pcout, 
                const std::string  name);
  
template<typename ParameterType>
void print_value(ConditionalOStream  &pcout, 
                 const ParameterType value);

template <>
void print_value(ConditionalOStream &pcout, 
                 const bool         value);

template <>
void print_value(ConditionalOStream &pcout,
                 const double       value);

#include <iomanip>

// print a parameter (which has a name and a value)
template<typename ParameterType>
void print_parameter(ConditionalOStream  &pcout,
                     const std::string   name,
                     const ParameterType value)
{
  print_name(pcout, name);
  print_value(pcout, value);
}

// print name and insert spaces so that the output is aligned
// needs inline because this function has no template
inline void print_name(ConditionalOStream &pcout,
                       const std::string  name)
{
  const unsigned int max_length_name = 45;

  pcout << "  " /* 2 */ << name  /* name.length*/ << ":" /* 1 */;
  const int remaining_spaces = max_length_name - 2 - 1 - name.length();

  for(int i=0; i<remaining_spaces; ++i)
    pcout << " " /* 1 */;
}

// print value for general parameter data type
template<typename ParameterType>
void print_value(ConditionalOStream  &pcout,
                 const ParameterType value)
{
  pcout << value << std::endl;
}

// specialization of template function for parameters of type bool
template<>
inline void print_value(ConditionalOStream &pcout,
                        const bool         value)
{
  std::string value_string = "default";
  value_string = (value == true) ? "true" : "false";
  print_value(pcout,value_string);
}

// specialization of template function for parameters of type double
template<>
inline void print_value(ConditionalOStream &pcout,
                        const double       value)
{
  pcout << std::scientific << std::setprecision(4) << value << std::endl;
}

#endif /* INCLUDE_FUNCTIONALITIES_PRINTFUNCTIONS_H_ */
