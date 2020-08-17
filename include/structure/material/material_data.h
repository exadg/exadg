/*
 * material_data.h
 *
 *  Created on: 19.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_MATERIAL_MATERIAL_DATA_H_
#define INCLUDE_STRUCTURE_MATERIAL_MATERIAL_DATA_H_

#include "../user_interface/enum_types.h"

namespace ExaDG
{
namespace Structure
{
struct MaterialData
{
  MaterialData(MaterialType const & type) : type(type)
  {
  }

  MaterialType type;
};

} // namespace Structure
} // namespace ExaDG

#endif /* INCLUDE_STRUCTURE_MATERIAL_MATERIAL_DATA_H_ */
