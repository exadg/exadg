#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/fluid_structure_interaction/precice/coupling_interface.h>

namespace Adapter
{
using namespace dealii;

/**
 * Derived class of the CouplingInterface: the classical coupling approach,
 * where each participant defines an interface based on the locally owned
 * triangulation. Here, quadrature points are used for reading and writing.
 * data_dim is equivalent to n_components, indicating the type of your data in
 * the preCICE sense (Vector vs Scalar)
 */
template<int dim, int data_dim, typename VectorizedArrayType>
class QuadInterface : public CouplingInterface<dim, data_dim, VectorizedArrayType>
{
public:
  QuadInterface(std::shared_ptr<const MatrixFree<dim, double, VectorizedArrayType>> data,
                std::shared_ptr<precice::SolverInterface>                           precice,
                const std::string                                                   mesh_name,
                const types::boundary_id                                            interface_id,
                const int                                                           mf_dof_index,
                const int                                                           mf_quad_index)
    : CouplingInterface<dim, data_dim, VectorizedArrayType>(data, precice, mesh_name, interface_id),
      mf_dof_index(mf_dof_index),
      mf_quad_index(mf_quad_index)
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
   * @brief write_data Evaluates the given @param data at the
   *        quadrature_points of the defined mesh and passes
   *        them to preCICE
   *
   * @param[in] data_vector The data to be passed to preCICE (absolute
   *            displacement for FSI). Note that the data_vector needs to
   *            contain valid ghost values for parallel runs, i.e.
   *            update_ghost_values must be calles before
   */
  virtual void
  write_data(const LinearAlgebra::distributed::Vector<double> & data_vector,
             const std::string &                                data_name) override;

private:
  /**
   * @brief write_data_factory Factory function in order to write different
   *        data (gradients, values..) to preCICE
   *
   * @param[in] data_vector The data to be passed to preCICE (absolute
   *            displacement for FSI)
   * @param[in] flags
   * @param[in] get_write_value
   */
  void
  write_data_factory(
    const LinearAlgebra::distributed::Vector<double> &                  data_vector,
    const int                                                           write_data_id,
    const EvaluationFlags::EvaluationFlags                              flags,
    const std::function<value_type(FEFaceIntegrator &, unsigned int)> & get_write_value);

  /// The preCICE IDs
  std::vector<std::array<int, VectorizedArrayType::size()>> interface_nodes_ids;

  /// Indices related to the FEEvaluation (have a look at the initialization
  /// of the MatrixFree)
  const int mf_dof_index;
  const int mf_quad_index;

  virtual std::string
  get_interface_type() const override;
};



template<int dim, int data_dim, typename VectorizedArrayType>
void
QuadInterface<dim, data_dim, VectorizedArrayType>::define_coupling_mesh(
  const std::vector<Point<dim>> &)
{
  Assert(this->mesh_id != -1, ExcNotInitialized());

  // In order to avoid that we define the interface multiple times when reader
  // and writer refer to the same object
  if(interface_nodes_ids.size() > 0)
    return;

  // Initial guess: half of the boundary is part of the coupling interface
  interface_nodes_ids.reserve(this->mf_data->n_boundary_face_batches() * 0.5);

  // Set up data structures
  FEFaceIntegrator phi(*this->mf_data, true, mf_dof_index, mf_quad_index);
  std::array<double, dim * VectorizedArrayType::size()> unrolled_vertices;
  std::array<int, VectorizedArrayType::size()>          node_ids;
  unsigned int                                          size = 0;
  // Loop over all boundary faces
  for(unsigned int face = this->mf_data->n_inner_face_batches();
      face < this->mf_data->n_boundary_face_batches() + this->mf_data->n_inner_face_batches();
      ++face)
  {
    const auto boundary_id = this->mf_data->get_boundary_id(face);

    // Only for interface nodes
    if(boundary_id != this->dealii_boundary_interface_id)
      continue;

    phi.reinit(face);
    const int active_faces = this->mf_data->n_active_entries_per_face_batch(face);

    // Loop over all quadrature points and pass the vertices to preCICE
    for(unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto local_vertex = phi.quadrature_point(q);

      // Transform Point<Vectorized> into preCICE conform format
      // We store here also the potential 'dummy'/empty lanes (not only
      // active_faces), but it allows us to use a static loop as well as a
      // static array for the indices
      for(int d = 0; d < dim; ++d)
        for(unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
          unrolled_vertices[d + dim * v] = local_vertex[d][v];

      this->precice->setMeshVertices(this->mesh_id,
                                     active_faces,
                                     unrolled_vertices.data(),
                                     node_ids.data());
      interface_nodes_ids.emplace_back(node_ids);
      ++size;
    }
  }
  // resize the IDs in case the initial guess was too large
  interface_nodes_ids.resize(size);
  // Consistency check: the number of IDs we stored is equal or greater than
  // the IDs preCICE knows
  Assert(size * VectorizedArrayType::size() >=
           static_cast<unsigned int>(this->precice->getMeshVertexSize(this->mesh_id)),
         ExcInternalError());

  if(this->read_data_map.size() > 0)
    this->print_info(true, this->precice->getMeshVertexSize(this->mesh_id));
  if(this->write_data_map.size() > 0)
    this->print_info(false, this->precice->getMeshVertexSize(this->mesh_id));
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
QuadInterface<dim, data_dim, VectorizedArrayType>::write_data(
  const LinearAlgebra::distributed::Vector<double> & data_vector,
  const std::string &                                data_name)
{
  const int write_data_id = this->write_data_map.at(data_name);

  switch(this->write_data_type)
  {
    case WriteDataType::values_on_q_points:
      write_data_factory(data_vector,
                         write_data_id,
                         EvaluationFlags::values,
                         [](auto & phi, auto q_point) { return phi.get_value(q_point); });
      break;
    case WriteDataType::normal_gradients_on_q_points:
      write_data_factory(data_vector,
                         write_data_id,
                         EvaluationFlags::gradients,
                         [](auto & phi, auto q_point) {
                           return phi.get_normal_derivative(q_point);
                         });
      break;
    default:
      AssertThrow(false, ExcNotImplemented());
  }
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
QuadInterface<dim, data_dim, VectorizedArrayType>::write_data_factory(
  const LinearAlgebra::distributed::Vector<double> &                  data_vector,
  const int                                                           write_data_id,
  const EvaluationFlags::EvaluationFlags                              flags,
  const std::function<value_type(FEFaceIntegrator &, unsigned int)> & get_write_value)
{
  Assert(write_data_id != -1, ExcNotInitialized());
  Assert(interface_nodes_ids.size() > 0, ExcNotInitialized());
  // Similar as in define_coupling_mesh
  FEFaceIntegrator phi(*this->mf_data, true, mf_dof_index, mf_quad_index);

  // In order to unroll the vectorization
  std::array<double, data_dim * VectorizedArrayType::size()> unrolled_local_data;
  (void)unrolled_local_data;

  auto index = interface_nodes_ids.begin();

  // Loop over all faces
  for(unsigned int face = this->mf_data->n_inner_face_batches();
      face < this->mf_data->n_boundary_face_batches() + this->mf_data->n_inner_face_batches();
      ++face)
  {
    const auto boundary_id = this->mf_data->get_boundary_id(face);

    // Only for interface nodes
    if(boundary_id != this->dealii_boundary_interface_id)
      continue;

    // Read and interpolate
    phi.reinit(face);
    phi.read_dof_values_plain(data_vector);
    phi.evaluate(flags);
    const int active_faces = this->mf_data->n_active_entries_per_face_batch(face);

    for(unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto local_data = get_write_value(phi, q);
      Assert(index != interface_nodes_ids.end(), ExcInternalError());

      // Constexpr evaluation required in order to comply with the
      // compiler here
      if constexpr(data_dim > 1)
      {
        // Transform Tensor<1,dim,VectorizedArrayType> into preCICE
        // conform format
        for(int d = 0; d < data_dim; ++d)
          for(unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
            unrolled_local_data[d + data_dim * v] = local_data[d][v];

        this->precice->writeBlockVectorData(write_data_id,
                                            active_faces,
                                            index->data(),
                                            unrolled_local_data.data());
      }
      else
      {
        this->precice->writeBlockScalarData(write_data_id,
                                            active_faces,
                                            index->data(),
                                            &local_data[0]);
      }
      ++index;
    }
  }
}



template<int dim, int data_dim, typename VectorizedArrayType>
std::string
QuadInterface<dim, data_dim, VectorizedArrayType>::get_interface_type() const
{
  return "quadrature points using matrix-free quad index " + Utilities::to_string(mf_quad_index);
}

// TODO
//  get_mesh_stats()
} // namespace Adapter
