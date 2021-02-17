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

#ifndef INCLUDE_FUNCTIONALITIES_MATRIX_FREE_WRAPPER_H_
#define INCLUDE_FUNCTIONALITIES_MATRIX_FREE_WRAPPER_H_

// deal.II
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
struct MatrixFreeData
{
public:
  std::vector<DoFHandler<dim> const *> &
  get_dof_handler_vector()
  {
    return dof_handler_vec;
  }

  std::vector<AffineConstraints<Number> const *> &
  get_constraint_vector()
  {
    return constraint_vec;
  }

  std::vector<Quadrature<1>> &
  get_quadrature_vector()
  {
    return quadrature_vec;
  }

  DoFHandler<dim> const &
  get_dof_handler(std::string const & name)
  {
    return *dof_handler_vec.at(get_dof_index(name));
  }

  void
  append_mapping_flags(MappingFlags const & flags_other)
  {
    MappingFlags flags;

    flags.cells          = this->data.mapping_update_flags;
    flags.inner_faces    = this->data.mapping_update_flags_inner_faces;
    flags.boundary_faces = this->data.mapping_update_flags_boundary_faces;

    // append
    flags = flags || flags_other;

    this->data.mapping_update_flags                = flags.cells;
    this->data.mapping_update_flags_inner_faces    = flags.inner_faces;
    this->data.mapping_update_flags_boundary_faces = flags.boundary_faces;
  }

  void
  insert_dof_handler(DoFHandler<dim> const * dof_handler, std::string const & name)
  {
    insert_element(dof_handler_vec, dof_index_map, dof_handler, name);
  }

  void
  insert_constraint(AffineConstraints<Number> const * constraint, std::string const & name)
  {
    insert_element(constraint_vec, constraint_index_map, constraint, name);
  }

  void insert_quadrature(Quadrature<1> const & quadrature, std::string const & name)
  {
    insert_element(quadrature_vec, quad_index_map, quadrature, name);
  }

  unsigned int
  get_dof_index(std::string const & name)
  {
    return get_index(dof_index_map, name);
  }

  unsigned int
  get_constraint_index(std::string const & name)
  {
    return get_index(constraint_index_map, name);
  }

  unsigned int
  get_quad_index(std::string const & name)
  {
    return get_index(quad_index_map, name);
  }

  // additional data
  typename MatrixFree<dim, Number>::AdditionalData data;

private:
  template<typename T>
  void
  insert_element(std::vector<T> &                      vector,
                 std::map<std::string, unsigned int> & map,
                 T const &                             element,
                 std::string const &                   name)
  {
    unsigned int index = vector.size();

    auto it = map.find(name);

    // make sure that this element does not already exist
    if(it == map.end())
    {
      map.insert(std::pair<std::string, unsigned int>(name, index));
    }
    else
    {
      AssertThrow(it == map.end(), ExcMessage("Element already exists. Aborting."));
    }

    vector.resize(index + 1);
    vector.at(index) = element;
  }

  unsigned int
  get_index(std::map<std::string, unsigned int> const & map, std::string const & name)
  {
    auto it = map.find(name);

    unsigned int index = numbers::invalid_unsigned_int;

    if(it != map.end())
    {
      index = it->second;
    }
    else
    {
      AssertThrow(it != map.end(), ExcMessage("Could not find element. Aborting."));
    }

    return index;
  }

  // maps between names and indices for DoFHandler, Constraint, Quadrature
  std::map<std::string, unsigned int> dof_index_map;
  std::map<std::string, unsigned int> constraint_index_map;
  std::map<std::string, unsigned int> quad_index_map;

  // collection of data structures required for initialization and update of matrix_free
  std::vector<DoFHandler<dim> const *>           dof_handler_vec;
  std::vector<AffineConstraints<Number> const *> constraint_vec;
  std::vector<Quadrature<1>>                     quadrature_vec;
};
} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_MATRIX_FREE_WRAPPER_H_ */
