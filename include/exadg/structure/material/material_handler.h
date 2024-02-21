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

#ifndef INCLUDE_EXADG_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_
#define INCLUDE_EXADG_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_

// deal.II
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/structure/material/library/compressible_neo_hookean.h>
#include <exadg/structure/material/library/incompressible_fibrous_tissue.h>
#include <exadg/structure/material/library/incompressible_neo_hookean.h>
#include <exadg/structure/material/library/st_venant_kirchhoff.h>
#include <exadg/structure/material/material.h>
#include <exadg/structure/user_interface/material_descriptor.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class MaterialHandler
{
public:
  typedef std::pair<dealii::types::material_id, std::shared_ptr<Material<dim, Number>>> Pair;
  typedef std::map<dealii::types::material_id, std::shared_ptr<Material<dim, Number>>>  Materials;

  MaterialHandler() : dof_index(0)
  {
  }

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             unsigned int const                        dof_index,
             unsigned int const                        quad_index,
             std::shared_ptr<MaterialDescriptor const> material_descriptor,
             bool const                                large_deformation,
             bool const                                spatial_integration,
             bool const                                force_material_residual,
             unsigned int const                        check_type,
             unsigned int const                        cache_level)
  {
    this->dof_index           = dof_index;
    this->material_descriptor = material_descriptor;

    for(auto iter = material_descriptor->begin(); iter != material_descriptor->end(); ++iter)
    {
      dealii::types::material_id    id   = iter->first;
      std::shared_ptr<MaterialData> data = iter->second;
      MaterialType                  type = data->type;

      switch(type)
      {
        case MaterialType::Undefined:
        {
          AssertThrow(false, dealii::ExcMessage("Material type is undefined."));
          break;
        }
        case MaterialType::StVenantKirchhoff:
        {
          std::shared_ptr<StVenantKirchhoffData<dim>> data_StVenantKirchhoff =
            std::static_pointer_cast<StVenantKirchhoffData<dim>>(data);
          material_map.insert(Pair(id,
                                   new StVenantKirchhoff<dim, Number>(matrix_free,
                                                                      dof_index,
                                                                      quad_index,
                                                                      *data_StVenantKirchhoff,
                                                                      large_deformation,
                                                                      check_type)));
          break;
        }
        case MaterialType::IncompressibleNeoHookean:
        {
          AssertThrow(
            large_deformation == true,
            dealii::ExcMessage(
              "Incompressible Neo-Hookean material model defined for finite strain theory."));

          std::shared_ptr<IncompressibleNeoHookeanData<dim>> data_IncompressibleNeoHookean =
            std::static_pointer_cast<IncompressibleNeoHookeanData<dim>>(data);
          material_map.insert(
            Pair(id,
                 new IncompressibleNeoHookean<dim, Number>(matrix_free,
                                                           dof_index,
                                                           quad_index,
                                                           *data_IncompressibleNeoHookean,
                                                           spatial_integration,
                                                           force_material_residual,
                                                           check_type,
                                                           cache_level)));
          break;
        }
        case MaterialType::IncompressibleFibrousTissue:
        {
          AssertThrow(large_deformation == true,
                      dealii::ExcMessage(
                        "Incompressible fiber model defined for finite strain theory."));

          std::shared_ptr<IncompressibleFibrousTissueData<dim>> data_IncompressibleFibrousTissue =
            std::static_pointer_cast<IncompressibleFibrousTissueData<dim>>(data);
          material_map.insert(
            Pair(id,
                 new IncompressibleFibrousTissue<dim, Number>(matrix_free,
                                                              dof_index,
                                                              quad_index,
                                                              *data_IncompressibleFibrousTissue,
                                                              spatial_integration,
                                                              force_material_residual,
                                                              check_type,
                                                              cache_level)));
          break;
        }
        case MaterialType::CompressibleNeoHookean:
        {
          AssertThrow(
            large_deformation == true,
            dealii::ExcMessage(
              "Compressible Neo-Hookean material model defined for finite strain theory."));

          std::shared_ptr<CompressibleNeoHookeanData<dim>> data_CompressibleNeoHookean =
            std::static_pointer_cast<CompressibleNeoHookeanData<dim>>(data);
          material_map.insert(
            Pair(id,
                 new CompressibleNeoHookean<dim, Number>(matrix_free,
                                                         dof_index,
                                                         quad_index,
                                                         *data_CompressibleNeoHookean,
                                                         spatial_integration,
                                                         force_material_residual,
                                                         check_type,
                                                         cache_level)));
          break;
        }
        default:
        {
          AssertThrow(false, dealii::ExcMessage("Specified material type is not implemented."));
          break;
        }
      }
    }
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free, unsigned int const cell)
  {
    auto mid = matrix_free.get_cell_iterator(cell, 0, dof_index)->material_id();

#ifdef DEBUG
    for(unsigned int v = 1; v < matrix_free.n_active_entries_per_cell_batch(cell); v++)
      AssertThrow(mid == matrix_free.get_cell_iterator(cell, v)->material_id(),
                  dealii::ExcMessage("You have to categorize cells according to their materials!"));
#endif

    material = material_map[mid];
  }

  std::shared_ptr<Material<dim, Number>>
  get_material() const
  {
    return material;
  }

private:
  unsigned int dof_index;

  std::shared_ptr<MaterialDescriptor const> material_descriptor;
  Materials                                 material_map;

  // pointer to material of current cell
  std::shared_ptr<Material<dim, Number>> material;
};

} // namespace Structure
} // namespace ExaDG

#endif /* INCLUDE_EXADG_STRUCTURE_MATERIAL_MATERIAL_HANDLER_H_ */
