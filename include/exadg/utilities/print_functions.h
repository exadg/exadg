/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_FUNCTIONALITIES_PRINTFUNCTIONS_H_
#define INCLUDE_FUNCTIONALITIES_PRINTFUNCTIONS_H_

// C/C++
#include <iomanip>
#include <iostream>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

// ExaDG
#include <exadg/utilities/enum_utilities.h>

namespace ExaDG
{
template<typename ParameterType>
void
print_parameter(dealii::ConditionalOStream const & pcout,
                std::string const                  name,
                ParameterType const                value);

void
print_name(dealii::ConditionalOStream const & pcout, std::string const name);

template<typename ParameterType>
void
print_value(dealii::ConditionalOStream const & pcout, ParameterType const value);

template<>
void
print_value(dealii::ConditionalOStream const & pcout, bool const value);

template<>
void
print_value(dealii::ConditionalOStream const & pcout, double const value);

// print a parameter (which has a name and a value)
template<typename ParameterType>
void
print_parameter(dealii::ConditionalOStream const & pcout,
                std::string const                  name,
                ParameterType const                value)
{
  print_name(pcout, name);
  print_value(pcout, value);
}

// print name and insert spaces so that the output is aligned
// needs inline because this function has no template
inline void
print_name(dealii::ConditionalOStream const & pcout, std::string const name)
{
  unsigned int const max_length_name = 45;

  pcout << "  " /* 2 */ << name /* name.length*/ << ":" /* 1 */;
  int const remaining_spaces = max_length_name - 2 - 1 - name.length();

  for(int i = 0; i < remaining_spaces; ++i)
    pcout << " " /* 1 */;
}

// print value for general parameter data type
template<typename ParameterType>
void
print_value(dealii::ConditionalOStream const & pcout, ParameterType const value)
{
  if constexpr(Utilities::is_enum<ParameterType>())
    pcout << Utilities::enum_to_string(value) << std::endl;
  else
    pcout << value << std::endl;
}

// specialization of template function for parameters of type bool
template<>
inline void
print_value(dealii::ConditionalOStream const & pcout, bool const value)
{
  std::string value_string = "default";
  value_string             = (value == true) ? "true" : "false";
  print_value(pcout, value_string);
}

// specialization of template function for parameters of type double
template<>
inline void
print_value(dealii::ConditionalOStream const & pcout, double const value)
{
  pcout << std::scientific << std::setprecision(4) << value << std::endl;
}

inline std::string
print_horizontal_line()
{
  return "________________________________________________________________________________";
}

inline void
print_write_output_time(double const       time,
                        unsigned int const counter,
                        bool const         unsteady,
                        MPI_Comm const &   mpi_comm)
{
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);
  if(unsteady)
  {
    pcout << std::endl
          << "OUTPUT << Write data at time t = " << std::scientific << std::setprecision(4) << time
          << std::endl;
  }
  else
  {
    pcout << std::endl
          << "OUTPUT << Write " << (counter == 0 ? "initial" : "solution") << " data" << std::endl;
  }
}

} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_PRINTFUNCTIONS_H_ */
