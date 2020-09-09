/*
 * material_descriptor.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_MATERIAL_DESCRIPTOR_H_
#define INCLUDE_EXADG_STRUCTURE_USER_INTERFACE_MATERIAL_DESCRIPTOR_H_

// C/C++
#include <map>

// deal.II
#include <deal.II/base/types.h>

// ExaDG
#include <exadg/structure/material/material_data.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

using MaterialDescriptor = std::map<types::material_id, std::shared_ptr<MaterialData>>;
} // namespace Structure
} // namespace ExaDG

#endif
