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

#ifndef INCLUDE_EXADG_UTILITIES_ENUM_UTILITIES_H_
#define INCLUDE_EXADG_UTILITIES_ENUM_UTILITIES_H_

#include <boost/algorithm/string/join.hpp>
#include <magic_enum/magic_enum.hpp>

#include <deal.II/base/exceptions.h>

namespace ExaDG
{
/**
 * Collection of enums utilities.
 */
namespace Utilities
{
/// constexpr which checks if given type is an enum or enum class
template<typename Type>
constexpr bool
is_enum()
{
  return (std::is_enum_v<Type> || magic_enum::is_scoped_enum_v<Type>);
}

/// returns the names of the enums joined with |
template<typename EnumType>
std::string
serialized_string()
{
  auto const enum_strings = magic_enum::enum_names<EnumType>();

  std::vector<std::string> const enums_strings_vec(enum_strings.begin(), enum_strings.end());
  return boost::algorithm::join(enums_strings_vec, "|");
}

template<typename EnumType>
std::string
enum_to_string(EnumType const enum_type)
{
  return (std::string)magic_enum::enum_name(enum_type);
}

template<typename EnumType>
void
string_to_enum(EnumType & enum_type, std::string const & enum_name)
{
  auto casted_enum = magic_enum::enum_cast<EnumType>(enum_name);
  if(casted_enum.has_value())
    enum_type = casted_enum.value();
  else
    AssertThrow(false, dealii::ExcMessage("Could not convert string to enum."));
}

} // namespace Utilities
} // namespace ExaDG

#endif /* INCLUDE_EXADG_UTILITIES_ENUM_UTILITIES_H_ */
