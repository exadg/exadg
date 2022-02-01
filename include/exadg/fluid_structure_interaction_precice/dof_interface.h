#pragma once

#include <deal.II/dofs/dof_tools.h>

#include <exadg/fluid_structure_interaction_precice/coupling_interface.h>
#include <exadg/fluid_structure_interaction_precice/dof_tools_extension.h>

namespace Adapter
{
using namespace dealii;

/**
 * Derived class of the CouplingInterface: the classical coupling approach,
 * where each participant defines an interface based on the locally owned
 * triangulation. Here, dof support points are used for reading and writing.
 * data_dim is equivalent to n_components, indicating the type of your data in
 * the preCICE sense (Vector vs Scalar)
 */
template<int dim, int data_dim, typename VectorizedArrayType>
class DoFInterface : public CouplingInterface<dim, data_dim, VectorizedArrayType>
{
public:
  DoFInterface(std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
               std::shared_ptr<precice::SolverInterface>                           precice,
               std::string                                                         mesh_name,
               types::boundary_id                                                  interface_id,
               int                                                                 mf_dof_index)
    : CouplingInterface<dim, data_dim, VectorizedArrayType>(data, precice, mesh_name, interface_id),
      mf_dof_index(mf_dof_index)
  {
  }

  /**
   * @brief define_mesh_vertices Define a vertex coupling mesh for preCICE
   *        coupling the classical preCICE way
   */
  virtual void
  define_coupling_mesh(const std::vector<Point<dim>> & vec) override;

  /**
   * @brief write_data Evaluates the given @param data at the
   *        quadrature_points of the defined mesh and passes
   *        them to preCICE
   *
   * @param[in] data_vector The data to be passed to preCICE (absolute
   *            displacement for FSI)
   */
  virtual void
  write_data(const LinearAlgebra::distributed::Vector<double> & data_vector) override;


private:
  /// The preCICE IDs
  std::vector<int> interface_nodes_ids;
  /// The deal.II associated IDs
  std::vector<std::array<types::global_dof_index, data_dim>> global_indices;

  bool interface_is_defined = false;
  /// Indices related to the FEEvaluation (have a look at the initialization
  /// of the MatrixFree)
  const int mf_dof_index;

  virtual std::string
  get_interface_type() const override;
};



template<int dim, int data_dim, typename VectorizedArrayType>
void
DoFInterface<dim, data_dim, VectorizedArrayType>::define_coupling_mesh(
  const std::vector<Point<dim>> &)
{
  Assert(this->mesh_id != -1, ExcNotInitialized());

  // In order to avoid that we define the interface multiple times when reader
  // and writer refer to the same object
  if(interface_is_defined)
    return;

  // Get and sort the global dof indices
  auto get_component_dofs = [&](const int component) {
    // Get a component mask of the vector component
    ComponentMask component_mask(data_dim, false);
    component_mask.set(component, true);

    // Get the global DoF indices of the component
    // Compute the intersection with locally owned dofs
    // TODO: This is super inefficient, have a look at the
    // dof_handler.n_boundary_dofs implementation for a proper version
    const IndexSet indices =
      (DoFTools::extract_boundary_dofs(this->mf_data->get_dof_handler(mf_dof_index),
                                       component_mask,
                                       std::set<types::boundary_id>{
                                         this->dealii_boundary_interface_id}) &
       this->mf_data->get_dof_handler(mf_dof_index).locally_owned_dofs());

    Assert(indices.n_elements() * data_dim ==
             this->mf_data->get_dof_handler(mf_dof_index)
               .n_boundary_dofs(std::set<types::boundary_id>{this->dealii_boundary_interface_id}),
           ExcInternalError());
    // Resize the global dof index conatiner in case we call this lambda for
    // the first time
    if(component == 0)
      global_indices.resize(indices.n_elements());
    // fill the first array entry with the respective component
    types::global_dof_index iterator = 0;
    for(const auto dof : indices)
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
  std::map<types::global_dof_index, Point<dim>> support_points;
  ComponentMask                                 component_mask(data_dim, false);
  component_mask.set(0, true);

  DoFTools::map_boundary_dofs_to_support_points(*(this->mf_data->get_mapping_info().mapping),
                                                this->mf_data->get_dof_handler(mf_dof_index),
                                                support_points,
                                                component_mask,
                                                this->dealii_boundary_interface_id);


  // Set size of the preCICE ID vector
  interface_nodes_ids.reserve(global_indices.size());
  std::array<double, dim> nodes_position;
  for(std::size_t i = 0; i < global_indices.size(); ++i)
  {
    // Get index of the zeroth component
    const auto element = global_indices[i][0];
    for(int d = 0; d < dim; ++d)
      nodes_position[d] = support_points[element][d];

    // pass node coordinates to precice
    const int precice_id = this->precice->setMeshVertex(this->mesh_id, nodes_position.data());
    interface_nodes_ids.emplace_back(precice_id);
  }

  interface_is_defined = true;

  if(this->read_data_id != -1)
    this->print_info(true, this->precice->getMeshVertexSize(this->mesh_id));
  if(this->write_data_id != -1)
    this->print_info(false, this->precice->getMeshVertexSize(this->mesh_id));
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
DoFInterface<dim, data_dim, VectorizedArrayType>::write_data(
  const LinearAlgebra::distributed::Vector<double> & data_vector)
{
  Assert(this->write_data_id != -1, ExcNotInitialized());
  Assert(interface_is_defined, ExcNotInitialized());

  std::array<double, data_dim> write_data;
  for(std::size_t i = 0; i < global_indices.size(); ++i)
  {
    // Extract relevant elements from global vector
    for(int d = 0; d < data_dim; ++d)
    {
      const auto element = global_indices[i][d];
      write_data[d]      = data_vector[element];
    }

    // and pass them to preCICE
    if constexpr(data_dim > 1)
    {
      this->precice->writeVectorData(this->write_data_id,
                                     interface_nodes_ids[i],
                                     write_data.data());
    }
    else
    {
      this->precice->writeScalarData(this->write_data_id, interface_nodes_ids[i], write_data[0]);
    }
  }
}



template<int dim, int data_dim, typename VectorizedArrayType>
std::string
DoFInterface<dim, data_dim, VectorizedArrayType>::get_interface_type() const
{
  return "DoF support points using matrix-free dof index " + Utilities::to_string(mf_dof_index);
}

// TODO
//  get_mesh_stats()
} // namespace Adapter
