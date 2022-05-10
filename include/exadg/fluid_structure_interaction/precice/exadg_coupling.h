/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_EXADG_COUPLING_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_EXADG_COUPLING_H_

#include <exadg/fluid_structure_interaction/precice/coupling_base.h>
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>

namespace ExaDG
{
namespace preCICE
{
/**
 * Derived class of the CouplingBase: shallow wrapper around the preCICE API functions,
 * where the participant defines a vector of points and the interface class here handles
 * the data exchange with preCICE, i.e., defining a coupling mesh and passing data to preCICE.
 */
template<int dim, int data_dim, typename VectorizedArrayType>
class ExaDGCoupling : public CouplingBase<dim, data_dim, VectorizedArrayType>
{
public:
  static unsigned int const rank =
    (data_dim == 1) ? 0 : ((data_dim == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  ExaDGCoupling(
    std::shared_ptr<dealii::MatrixFree<dim, double, VectorizedArrayType> const> data,
#ifdef EXADG_WITH_PRECICE
    std::shared_ptr<precice::SolverInterface> precice,
#endif
    std::string const                                  mesh_name,
    std::shared_ptr<ContainerInterfaceData<rank, dim>> interface_data_,
    dealii::types::boundary_id const surface_id = dealii::numbers::invalid_unsigned_int);

  /**
   * @brief define_mesh_vertices Define a vertex coupling mesh for preCICE
   *        coupling the classical preCICE way
   */
  virtual void
  define_coupling_mesh() override;

  /**
   * @brief write_data
   *
   * @param[in] data_vector The data to be passed to preCICE (absolute
   *            displacement for FSI). Note that the data_vector needs to
   *            contain valid ghost values for parallel runs, i.e.
   *            update_ghost_values must be calles before
   */
  virtual void
  write_data(dealii::LinearAlgebra::distributed::Vector<double> const & data_vector,
             std::string const &                                        data_name) override;

  virtual void
  read_block_data(std::string const & data_name) const override;

private:
  /// Accessor for ExaDG data structures
  std::shared_ptr<ContainerInterfaceData<rank, dim>> interface_data;

  /// The preCICE IDs
  std::vector<int> coupling_nodes_ids;

  virtual std::string
  get_surface_type() const override;
};



template<int dim, int data_dim, typename VectorizedArrayType>
ExaDGCoupling<dim, data_dim, VectorizedArrayType>::ExaDGCoupling(
  std::shared_ptr<dealii::MatrixFree<dim, double, VectorizedArrayType> const> data,
#ifdef EXADG_WITH_PRECICE
  std::shared_ptr<precice::SolverInterface> precice,
#endif
  std::string const                                  mesh_name,
  std::shared_ptr<ContainerInterfaceData<rank, dim>> interface_data_,
  dealii::types::boundary_id const                   surface_id)
  : CouplingBase<dim, data_dim, VectorizedArrayType>(data,
#ifdef EXADG_WITH_PRECICE
                                                     precice,
#endif
                                                     mesh_name,
                                                     surface_id),
    interface_data(interface_data_)
{
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
ExaDGCoupling<dim, data_dim, VectorizedArrayType>::define_coupling_mesh()
{
  Assert(this->mesh_id != -1, dealii::ExcNotInitialized());
  Assert(interface_data.get() != nullptr, dealii::ExcNotInitialized());

  // In order to avoid that we define the surface multiple times when reader
  // and writer refer to the same object
  if(coupling_nodes_ids.size() > 0)
    return;

  for(auto quad_index : interface_data->get_quad_indices())
  {
    // returns std::vector<dealii::Point<dim>>
    auto const & points = interface_data->get_array_q_points(quad_index);

    // Get current size of our interface
    auto const start_index = coupling_nodes_ids.size();

    // Allocate memory for new interface nodes
    coupling_nodes_ids.resize(start_index + points.size());

    // Set the vertices
#ifdef EXADG_WITH_PRECICE
    this->precice->setMeshVertices(this->mesh_id,
                                   points.size(),
                                   &points[0][0],
                                   &coupling_nodes_ids[start_index]);
#endif
  }

#ifdef EXADG_WITH_PRECICE
  if(this->read_data_map.size() > 0)
    this->print_info(true, this->precice->getMeshVertexSize(this->mesh_id));
  if(this->write_data_map.size() > 0)
    this->print_info(false, this->precice->getMeshVertexSize(this->mesh_id));
#endif
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
ExaDGCoupling<dim, data_dim, VectorizedArrayType>::read_block_data(
  std::string const & data_name) const
{
  Assert(interface_data.get() != nullptr, dealii::ExcNotInitialized());

  int const read_data_id = this->read_data_map.at(data_name);

  // summarizing the IDs already read
  unsigned int start_index = 0;
  // extract values of each quadrature rule
  for(auto quad_index : interface_data->get_quad_indices())
  {
    if constexpr(data_dim > 1)
    {
      auto &     array_solution = interface_data->get_array_solution(quad_index);
      auto const array_size     = array_solution.size();

      AssertIndexRange(start_index, coupling_nodes_ids.size());
#ifdef EXADG_WITH_PRECICE
      this->precice->readBlockVectorData(read_data_id,
                                         array_size,
                                         &coupling_nodes_ids[start_index],
                                         &array_solution[0][0]);
#else
      (void)read_data_id;
#endif
      start_index += array_size;
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
ExaDGCoupling<dim, data_dim, VectorizedArrayType>::write_data(
  dealii::LinearAlgebra::distributed::Vector<double> const &,
  std::string const &)
{
  AssertThrow(false, dealii::ExcNotImplemented());
}


template<int dim, int data_dim, typename VectorizedArrayType>
std::string
ExaDGCoupling<dim, data_dim, VectorizedArrayType>::get_surface_type() const
{
  return "exadg shallow wrapper";
}

} // namespace preCICE
} // namespace ExaDG

#endif
