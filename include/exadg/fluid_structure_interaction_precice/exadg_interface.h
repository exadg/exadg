#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/fluid_structure_interaction_precice/coupling_interface.h>

namespace Adapter
{
using namespace dealii;

/**
 * Derived class of the CouplingInterface: shallow wrapper,
 * where the participant defines a vector of points and
 * the interface handles only the exchange with preCICE.
 */
template<int dim, int data_dim, typename VectorizedArrayType>
class ExaDGInterface : public CouplingInterface<dim, data_dim, VectorizedArrayType>
{
public:
  ExaDGInterface(std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
                 std::shared_ptr<precice::SolverInterface>                           precice,
                 std::string                                                         mesh_name,
                 types::boundary_id                                                  interface_id)
    : CouplingInterface<dim, data_dim, VectorizedArrayType>(data, precice, mesh_name, interface_id)
  {
  }

  /// Alias as defined in the base class
  using FEFaceIntegrator =
    typename CouplingInterface<dim, data_dim, VectorizedArrayType>::FEFaceIntegrator;
  using value_type = typename CouplingInterface<dim, data_dim, VectorizedArrayType>::value_type;
  /**
   * @brief define_mesh_vertices Define a vertex coupling mesh for preCICE
   *        coupling the classical preCICE way
   */
  virtual void
  define_coupling_mesh(const std::vector<Point<dim>> & vec) override;

  /**
   * @brief write_data
   *
   * @param[in] data_vector The data to be passed to preCICE (absolute
   *            displacement for FSI). Note that the data_vector needs to
   *            contain valid ghost values for parallel runs, i.e.
   *            update_ghost_values must be calles before
   */
  virtual void
  write_data(const LinearAlgebra::distributed::Vector<double> & data_vector) override;

  virtual std::vector<Tensor<1, dim>>
  read_block_data() const override;

private:
  /// The preCICE IDs
  std::vector<int> interface_nodes_ids;

  bool interface_is_defined = false;

  virtual std::string
  get_interface_type() const override;
};



template<int dim, int data_dim, typename VectorizedArrayType>
void
ExaDGInterface<dim, data_dim, VectorizedArrayType>::define_coupling_mesh(
  const std::vector<Point<dim>> & vec)
{
  Assert(this->mesh_id != -1, ExcNotInitialized());

  // In order to avoid that we define the interface multiple times when reader
  // and writer refer to the same object
  if(interface_is_defined)
    return;

  // Initial guess: half of the boundary is part of the coupling interface
  interface_nodes_ids.resize(vec.size());

  this->precice->setMeshVertices(this->mesh_id, vec.size(), &vec[0][0], interface_nodes_ids.data());

  interface_is_defined = true;

  if(this->read_data_id != -1)
    this->print_info(true, this->precice->getMeshVertexSize(this->mesh_id));
  if(this->write_data_id != -1)
    this->print_info(false, this->precice->getMeshVertexSize(this->mesh_id));
}


template<int dim, int data_dim, typename VectorizedArrayType>
std::vector<Tensor<1, dim>>
ExaDGInterface<dim, data_dim, VectorizedArrayType>::read_block_data() const
{
  std::vector<Tensor<1, dim>> values(interface_nodes_ids.size());

  if constexpr(data_dim > 1)
  {
    this->precice->readBlockVectorData(this->read_data_id,
                                       interface_nodes_ids.size(),
                                       interface_nodes_ids.data(),
                                       &values[0][0]);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  return values;
}

template<int dim, int data_dim, typename VectorizedArrayType>
void
ExaDGInterface<dim, data_dim, VectorizedArrayType>::write_data(
  const LinearAlgebra::distributed::Vector<double> &)
{
  AssertThrow(false, ExcNotImplemented());
}


template<int dim, int data_dim, typename VectorizedArrayType>
std::string
ExaDGInterface<dim, data_dim, VectorizedArrayType>::get_interface_type() const
{
  return "exadg shallow wrapper ";
}

// TODO
//  get_mesh_stats()
} // namespace Adapter
