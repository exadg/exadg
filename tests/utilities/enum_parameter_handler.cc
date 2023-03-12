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

// C++
#include <iostream>
#include <sstream>

#include <deal.II/base/parameter_handler.h>

#include <exadg/utilities/enum_patterns.h>

enum class EnumClass
{
  Undefined,
  AScoped,
  BScoped,
  CScoped
};

enum Enum
{
  Undefined,
  A,
  B,
  C
};

template<typename EnumType>
void
test_enum(EnumType enum_type)
{
  dealii::ParameterHandler prm;

  prm.add_parameter("EnumType", enum_type);

  prm.print_parameters(std::cout, dealii::ParameterHandler::OutputStyle::Description);
  std::cout << std::endl;

  prm.print_parameters(std::cout, dealii::ParameterHandler::OutputStyle::PRM);
  std::cout << std::endl;

  auto const enum_strings = magic_enum::enum_names<EnumType>();
  for(const auto e : enum_strings)
  {
    std::istringstream is("{\"EnumType\" : \"" + std::string(e) + "\"}");
    prm.parse_input_from_json(is);
    prm.print_parameters(std::cout, dealii::ParameterHandler::OutputStyle::PRM);
    std::cout << std::endl;
  }
}


int
main()
{
  test_enum(EnumClass::Undefined);
  test_enum(Enum::Undefined);

  return 0;
}
