
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DOF_COUPLING_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DOF_COUPLING_H_

#include <deal.II/dofs/dof_tools.h>

#include <exadg/fluid_structure_interaction/precice/coupling_base.h>
#include <exadg/fluid_structure_interaction/precice/dof_tools_extension.h>

namespace ExaDG
{
namespace preCICE
{
/**
 * Derived class of the CouplingBase: the classical coupling approach,
 * where each participant defines an interface based on the locally owned
 * triangulation. Here, dof support points are used for reading and writing.
 * data_dim is equivalent to n_components, indicating the type of your data in
 * the preCICE sense (Vector vs Scalar)
 */
template<int dim, int data_dim, typename VectorizedArrayType>
class DoFCoupling : public CouplingBase<dim, data_dim, VectorizedArrayType>
{
public:
  DoFCoupling(std::shared_ptr<dealii::MatrixFree<dim, double, VectorizedArrayType> const> data,
#ifdef EXADG_WITH_PRECICE
              std::shared_ptr<precice::SolverInterface> precice,
#endif
              std::string const                mesh_name,
              dealii::types::boundary_id const surface_id,
              int const                        mf_dof_index);
  /**
   * @brief define_mesh_vertices Define a vertex coupling mesh for preCICE
   *        coupling the classical preCICE way
   */
  virtual void
  define_coupling_mesh() override;

  /**
   * @brief write_data Evaluates the given @param data at the
   *        quadrature_points of the defined mesh and passes
   *        them to preCICE
   *
   * @param[in] data_vector The data to be passed to preCICE (absolute
   *            displacement for FSI)
   */
  virtual void
  write_data(dealii::LinearAlgebra::distributed::Vector<double> const & data_vector,
             std::string const &                                        data_name) override;


private:
  /// The preCICE IDs
  std::vector<int> coupling_nodes_ids;
  /// The deal.II associated IDs
  std::vector<std::array<dealii::types::global_dof_index, data_dim>> global_indices;

  /// Indices related to the FEEvaluation (have a look at the initialization
  /// of the MatrixFree)
  int const mf_dof_index;

  virtual std::string
  get_surface_type() const override;
};



template<int dim, int data_dim, typename VectorizedArrayType>
DoFCoupling<dim, data_dim, VectorizedArrayType>::DoFCoupling(
  std::shared_ptr<dealii::MatrixFree<dim, double, VectorizedArrayType> const> data,
#ifdef EXADG_WITH_PRECICE
  std::shared_ptr<precice::SolverInterface> precice,
#endif
  std::string const                mesh_name,
  dealii::types::boundary_id const surface_id,
  int const                        mf_dof_index)
  : CouplingBase<dim, data_dim, VectorizedArrayType>(data,
#ifdef EXADG_WITH_PRECICE
                                                     precice,
#endif
                                                     mesh_name,
                                                     surface_id),
    mf_dof_index(mf_dof_index)
{
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
DoFCoupling<dim, data_dim, VectorizedArrayType>::define_coupling_mesh()
{
  Assert(this->mesh_id != -1, dealii::ExcNotInitialized());

  // In order to avoid that we define the surface multiple times when reader
  // and writer refer to the same object
  if(coupling_nodes_ids.size() > 0)
    return;

  // Get and sort the global dof indices
  auto const get_component_dofs = [&](int const component) {
    // Get a component mask of the vector component
    dealii::ComponentMask component_mask(data_dim, false);
    component_mask.set(component, true);

    // Get the global DoF indices of the component
    // Compute the intersection with locally owned dofs
    // TODO: This is super inefficient, have a look at the
    // dof_handler.n_boundary_dofs implementation for a proper version
    dealii::IndexSet const indices =
      (dealii::DoFTools::extract_boundary_dofs(this->matrix_free->get_dof_handler(mf_dof_index),
                                               component_mask,
                                               std::set<dealii::types::boundary_id>{
                                                 this->dealii_boundary_surface_id}) &
       this->matrix_free->get_dof_handler(mf_dof_index).locally_owned_dofs());

    Assert(indices.n_elements() * data_dim ==
             this->matrix_free->get_dof_handler(mf_dof_index)
               .n_boundary_dofs(
                 std::set<dealii::types::boundary_id>{this->dealii_boundary_surface_id}),
           dealii::ExcInternalError());
    // Resize the global dof index conatiner in case we call this lambda for
    // the first time
    if(component == 0)
      global_indices.resize(indices.n_elements());
    // fill the first array entry with the respective component
    dealii::types::global_dof_index iterator = 0;
    for(auto const dof : indices)
    {
      global_indices[iterator][component] = dof;
      ++iterator;
    }
  };

  // Fill the indices object this class works on
  for(int d = 0; d < data_dim; ++d)
    get_component_dofs(d);
  // Compute the coordinates of the indices (only one component required here)
  // We select the zeroth component
  std::map<dealii::types::global_dof_index, dealii::Point<dim>> support_points;
  dealii::ComponentMask                                         component_mask(data_dim, false);
  component_mask.set(0, true);

  dealii::DoFTools::map_boundary_dofs_to_support_points(
    *(this->matrix_free->get_mapping_info().mapping),
    this->matrix_free->get_dof_handler(mf_dof_index),
    support_points,
    component_mask,
    this->dealii_boundary_surface_id);


  // Set size of the preCICE ID vector
  coupling_nodes_ids.reserve(global_indices.size());
  std::array<double, dim> nodes_position;
  for(std::size_t i = 0; i < global_indices.size(); ++i)
  {
    // Get index of the zeroth component
    auto const element = global_indices[i][0];
    for(int d = 0; d < dim; ++d)
      nodes_position[d] = support_points[element][d];

      // pass node coordinates to precice
#ifdef EXADG_WITH_PRECICE
    int const precice_id = this->precice->setMeshVertex(this->mesh_id, nodes_position.data());
#else
    int const precice_id = 0;
#endif
    coupling_nodes_ids.emplace_back(precice_id);
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
DoFCoupling<dim, data_dim, VectorizedArrayType>::write_data(
  dealii::LinearAlgebra::distributed::Vector<double> const & data_vector,
  std::string const &                                        data_name)
{
  int const write_data_id = this->write_data_map.at(data_name);
  Assert(write_data_id != -1, dealii::ExcNotInitialized());
  Assert(coupling_nodes_ids.size() > 0, dealii::ExcNotInitialized());

  std::array<double, data_dim> write_data;
  for(std::size_t i = 0; i < global_indices.size(); ++i)
  {
    // Extract relevant elements from global vector
    for(unsigned int d = 0; d < data_dim; ++d)
    {
      auto const element = global_indices[i][d];
      write_data[d]      = data_vector[element];
    }

#ifdef EXADG_WITH_PRECICE
    // and pass them to preCICE
    if constexpr(data_dim > 1)
    {
      this->precice->writeVectorData(write_data_id, coupling_nodes_ids[i], write_data.data());
    }
    else
    {
      this->precice->writeScalarData(write_data_id, coupling_nodes_ids[i], write_data[0]);
    }
#else
    (void)write_data_id;
#endif
  }
}



template<int dim, int data_dim, typename VectorizedArrayType>
std::string
DoFCoupling<dim, data_dim, VectorizedArrayType>::get_surface_type() const
{
  return "DoF support points using matrix-free dof index " +
         dealii::Utilities::to_string(mf_dof_index);
}

} // namespace preCICE
} // namespace ExaDG

#endif
