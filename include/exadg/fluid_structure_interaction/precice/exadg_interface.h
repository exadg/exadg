#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/fluid_structure_interaction/precice/coupling_interface.h>
#include <exadg/fluid_structure_interaction/precice/interface_coupling.h>

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
                 const std::string                                                   mesh_name,
                 const types::boundary_id interface_id = numbers::invalid_unsigned_int)
    : CouplingInterface<dim, data_dim, VectorizedArrayType>(data, precice, mesh_name, interface_id)
  {
  }

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
  write_data(const LinearAlgebra::distributed::Vector<double> & data_vector,
             const std::string &                                data_name) override;

  virtual void
  read_block_data(const std::string & data_name) const override;

  void
  set_data_pointer(std::shared_ptr<ExaDG::InterfaceCoupling<dim, dim, double>> exadg_terminal_);

private:
  /// Accessor for ExaDG data structures
  std::shared_ptr<ExaDG::InterfaceCoupling<dim, dim, double>> exadg_terminal;
  /// The preCICE IDs
  std::vector<int> interface_nodes_ids;

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
  if(interface_nodes_ids.size() > 0)
    return;

  // Initial guess: half of the boundary is part of the coupling interface
  interface_nodes_ids.resize(vec.size());

  this->precice->setMeshVertices(this->mesh_id, vec.size(), &vec[0][0], interface_nodes_ids.data());

  if(this->read_data_map.size() > 0)
    this->print_info(true, this->precice->getMeshVertexSize(this->mesh_id));
  if(this->write_data_map.size() > 0)
    this->print_info(false, this->precice->getMeshVertexSize(this->mesh_id));
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
ExaDGInterface<dim, data_dim, VectorizedArrayType>::read_block_data(
  const std::string & data_name) const
{
  const int read_data_id = this->read_data_map.at(data_name);

  std::vector<Tensor<1, dim>> values(interface_nodes_ids.size());
  if constexpr(data_dim > 1)
  {
    this->precice->readBlockVectorData(read_data_id,
                                       interface_nodes_ids.size(),
                                       interface_nodes_ids.data(),
                                       &values[0][0]);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }
  Assert(exadg_terminal.get() != nullptr, ExcNotInitialized());
  exadg_terminal->update_data(values);
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
ExaDGInterface<dim, data_dim, VectorizedArrayType>::set_data_pointer(
  std::shared_ptr<ExaDG::InterfaceCoupling<dim, dim, double>> exadg_terminal_)
{
  exadg_terminal = exadg_terminal_;
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
ExaDGInterface<dim, data_dim, VectorizedArrayType>::write_data(
  const LinearAlgebra::distributed::Vector<double> &,
  const std::string &)
{
  AssertThrow(false, ExcNotImplemented());
}


template<int dim, int data_dim, typename VectorizedArrayType>
std::string
ExaDGInterface<dim, data_dim, VectorizedArrayType>::get_interface_type() const
{
  return "exadg shallow wrapper ";
}

} // namespace Adapter
