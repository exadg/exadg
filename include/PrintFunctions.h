/*
 * PrintFunctions.h
 *
 *  Created on: Aug 19, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRINTFUNCTIONS_H_
#define INCLUDE_PRINTFUNCTIONS_H_

#include <deal.II/base/conditional_ostream.h>

using namespace dealii;

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

template<typename ParameterType>
void print_parameter(ConditionalOStream  &pcout, 
                     const std::string   name, 
                     const ParameterType value);

#endif /* INCLUDE_PRINTFUNCTIONS_H_ */
