/*
 * material_handler.h
 *
 *  Created on: 18.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_
#define INCLUDE_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_

// deal.II
#include <deal.II/matrix_free/matrix_free.h>

#include "../user_interface/material_descriptor.h"
#include "library/st_venant_kirchhoff.h"
#include "material.h"

using namespace dealii;

namespace Structure
{
template<int dim, typename Number>
class MaterialHandler
{
public:
  typedef std::pair<types::material_id, std::shared_ptr<Material<dim, Number>>> Pair;
  typedef std::map<types::material_id, std::shared_ptr<Material<dim, Number>>>  Materials;

  void
  initialize(std::shared_ptr<MaterialDescriptor> material_descriptor_in)
  {
    material_descriptor = material_descriptor_in;

    for(auto iter = material_descriptor->begin(); iter != material_descriptor->end(); ++iter)
    {
      types::material_id            id   = iter->first;
      std::shared_ptr<MaterialData> data = iter->second;
      MaterialType                  type = data->type;

      switch(type)
      {
        case MaterialType::Undefined:
        {
          AssertThrow(false, ExcMessage("Material type is undefined."));
          break;
        }
        case MaterialType::StVenantKirchhoff:
        {
          std::shared_ptr<StVenantKirchhoffData> data_svk =
            std::static_pointer_cast<StVenantKirchhoffData>(data);
          material_map.insert(Pair(id, new StVenantKirchhoff<dim, Number>(*data_svk)));
          break;
        }
        default:
        {
          AssertThrow(false, ExcMessage("Specified material type is not implemented."));
          break;
        }
      }
    }
  }

  void
  reinit(MatrixFree<dim, Number> const & matrix_free, unsigned int const cell)
  {
    auto mid = matrix_free.get_cell_iterator(cell, 0)->material_id();

#ifdef DEBUG
    for(unsigned int v = 1; v < matrix_free.n_active_entries_per_cell_batch(cell); v++)
      AssertThrow(mid == matrix_free.get_cell_iterator(cell, v)->material_id(),
                  ExcMessage("You have to categorize cells according to their materials!"));
#endif

    material = material_map[mid];
  }

  std::shared_ptr<Material<dim, Number>>
  get_material() const
  {
    return material;
  }

private:
  std::shared_ptr<MaterialDescriptor> material_descriptor;
  Materials                           material_map;

  // pointer to material of current cell
  std::shared_ptr<Material<dim, Number>> material;
};

} // namespace Structure


#endif /* INCLUDE_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_ */
