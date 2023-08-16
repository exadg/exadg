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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_QUAD_COUPLING_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_QUAD_COUPLING_H_

#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/fluid_structure_interaction/precice/coupling_base.h>

namespace ExaDG
{
namespace preCICE
{
/**
 * Derived class of the CouplingBase: the classical coupling approach,
 * where each participant defines an surface based on the locally owned
 * triangulation. Here, quadrature points are used for reading and writing.
 * data_dim is equivalent to n_components, indicating the type of your data in
 * the preCICE sense (Vector vs Scalar)
 */
template<int dim, int data_dim, typename VectorizedArrayType>
class QuadCoupling : public CouplingBase<dim, data_dim, VectorizedArrayType>
{
public:
  QuadCoupling(dealii::MatrixFree<dim, double, VectorizedArrayType> const & data,
#ifdef EXADG_WITH_PRECICE
               std::shared_ptr<precice::SolverInterface> precice,
#endif
               std::string const                mesh_name,
               dealii::types::boundary_id const surface_id,
               int const                        mf_dof_index,
               int const                        mf_quad_index);

  /// Alias as defined in the base class
  using FEFaceIntegrator =
    typename CouplingBase<dim, data_dim, VectorizedArrayType>::FEFaceIntegrator;
  using value_type = typename CouplingBase<dim, data_dim, VectorizedArrayType>::value_type;

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
   *            displacement for FSI). Note that the data_vector needs to
   *            contain valid ghost values for parallel runs, i.e.
   *            update_ghost_values must be calles before. In addition,
   *            constraints need to be applied manually to the data before
   *            passing it into this function
   */
  virtual void
  write_data(dealii::LinearAlgebra::distributed::Vector<double> const & data_vector,
             std::string const &                                        data_name) override;

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
    dealii::LinearAlgebra::distributed::Vector<double> const &          data_vector,
    int const                                                           write_data_id,
    dealii::EvaluationFlags::EvaluationFlags const                      flags,
    std::function<value_type(FEFaceIntegrator &, unsigned int)> const & get_write_value);

  /// The preCICE IDs
  std::vector<std::array<int, VectorizedArrayType::size()>> coupling_nodes_ids;

  /// Indices related to the FEEvaluation (have a look at the initialization
  /// of the MatrixFree)
  int const mf_dof_index;
  int const mf_quad_index;

  virtual std::string
  get_surface_type() const override;
};



template<int dim, int data_dim, typename VectorizedArrayType>
QuadCoupling<dim, data_dim, VectorizedArrayType>::QuadCoupling(
  dealii::MatrixFree<dim, double, VectorizedArrayType> const & data,
#ifdef EXADG_WITH_PRECICE
  std::shared_ptr<precice::SolverInterface> precice,
#endif
  std::string const                mesh_name,
  dealii::types::boundary_id const surface_id,
  int const                        mf_dof_index_,
  int const                        mf_quad_index_)
  : CouplingBase<dim, data_dim, VectorizedArrayType>(data,
#ifdef EXADG_WITH_PRECICE
                                                     precice,
#endif
                                                     mesh_name,
                                                     surface_id),
    mf_dof_index(mf_dof_index_),
    mf_quad_index(mf_quad_index_)
{
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
QuadCoupling<dim, data_dim, VectorizedArrayType>::define_coupling_mesh()
{
  Assert(this->mesh_id != -1, dealii::ExcNotInitialized());

  // In order to avoid that we define the surface multiple times when reader
  // and writer refer to the same object
  if(coupling_nodes_ids.size() > 0)
    return;

  // Initial guess: half of the boundary is part of the coupling surface
  coupling_nodes_ids.reserve(this->matrix_free.n_boundary_face_batches() / 2);

  // Set up data structures
  FEFaceIntegrator phi(this->matrix_free, true, mf_dof_index, mf_quad_index);
  std::array<double, dim * VectorizedArrayType::size()> unrolled_vertices;
  std::array<int, VectorizedArrayType::size()>          node_ids;
  unsigned int                                          size = 0;
  // Loop over all boundary faces
  for(unsigned int face = this->matrix_free.n_inner_face_batches();
      face < this->matrix_free.n_boundary_face_batches() + this->matrix_free.n_inner_face_batches();
      ++face)
  {
    auto const boundary_id = this->matrix_free.get_boundary_id(face);

    // Only for interface nodes
    if(boundary_id != this->dealii_boundary_surface_id)
      continue;

    phi.reinit(face);
    int const active_faces = this->matrix_free.n_active_entries_per_face_batch(face);

    // Loop over all quadrature points and pass the vertices to preCICE
    for(unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      auto const local_vertex = phi.quadrature_point(q);

      // Transform dealii::Point<Vectorized> into preCICE conform format
      // We store here also the potential 'dummy'/empty lanes (not only
      // active_faces), but it allows us to use a static loop as well as a
      // static array for the indices
      for(int d = 0; d < dim; ++d)
        for(unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
          unrolled_vertices[d + dim * v] = local_vertex[d][v];

#ifdef EXADG_WITH_PRECICE
      this->precice->setMeshVertices(this->mesh_id,
                                     active_faces,
                                     unrolled_vertices.data(),
                                     node_ids.data());
#else
      (void)active_faces;
#endif
      coupling_nodes_ids.emplace_back(node_ids);
      ++size;
    }
  }

  // resize the IDs in case the initial guess was too large
  coupling_nodes_ids.resize(size);

#ifdef EXADG_WITH_PRECICE
  // Consistency check: the number of IDs we stored is equal or greater than
  // the IDs preCICE knows
  Assert(size * VectorizedArrayType::size() >=
           static_cast<unsigned int>(this->precice->getMeshVertexSize(this->mesh_id)),
         dealii::ExcInternalError());

  if(this->read_data_map.size() > 0)
    this->print_info(true, this->precice->getMeshVertexSize(this->mesh_id));
  if(this->write_data_map.size() > 0)
    this->print_info(false, this->precice->getMeshVertexSize(this->mesh_id));
#endif
}


template<int dim, int data_dim, typename VectorizedArrayType>
void
QuadCoupling<dim, data_dim, VectorizedArrayType>::write_data(
  dealii::LinearAlgebra::distributed::Vector<double> const & data_vector,
  std::string const &                                        data_name)
{
  int const write_data_id = this->write_data_map.at(data_name);

  switch(this->write_data_type)
  {
    case WriteDataType::values_on_q_points:
      write_data_factory(data_vector,
                         write_data_id,
                         dealii::EvaluationFlags::values,
                         [](auto & phi, auto q_point) { return phi.get_value(q_point); });
      break;
    case WriteDataType::normal_gradients_on_q_points:
      write_data_factory(data_vector,
                         write_data_id,
                         dealii::EvaluationFlags::gradients,
                         [](auto & phi, auto q_point) {
                           return phi.get_normal_derivative(q_point);
                         });
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }
}



template<int dim, int data_dim, typename VectorizedArrayType>
void
QuadCoupling<dim, data_dim, VectorizedArrayType>::write_data_factory(
  dealii::LinearAlgebra::distributed::Vector<double> const &          data_vector,
  int const                                                           write_data_id,
  dealii::EvaluationFlags::EvaluationFlags const                      flags,
  std::function<value_type(FEFaceIntegrator &, unsigned int)> const & get_write_value)
{
  Assert(write_data_id != -1, dealii::ExcNotInitialized());
  Assert(coupling_nodes_ids.size() > 0, dealii::ExcNotInitialized());
  // Similar as in define_coupling_mesh
  FEFaceIntegrator phi(this->matrix_free, true, mf_dof_index, mf_quad_index);

  // In order to unroll the vectorization
  std::array<double, data_dim * VectorizedArrayType::size()> unrolled_local_data;
  (void)unrolled_local_data;

  auto index = coupling_nodes_ids.begin();

  // Loop over all faces
  for(unsigned int face = this->matrix_free.n_inner_face_batches();
      face < this->matrix_free.n_boundary_face_batches() + this->matrix_free.n_inner_face_batches();
      ++face)
  {
    auto const boundary_id = this->matrix_free.get_boundary_id(face);

    // Only for interface nodes
    if(boundary_id != this->dealii_boundary_surface_id)
      continue;

    // Read and interpolate
    phi.reinit(face);
    phi.read_dof_values_plain(data_vector);
    phi.evaluate(flags);
    int const active_faces = this->matrix_free.n_active_entries_per_face_batch(face);

    for(unsigned int q = 0; q < phi.n_q_points; ++q, ++index)
    {
      Assert(index != coupling_nodes_ids.end(), dealii::ExcInternalError());
      auto const local_data = get_write_value(phi, q);

      // Constexpr evaluation required in order to comply with the
      // compiler here
      if constexpr(data_dim > 1)
      {
        // Transform Tensor<1,dim,VectorizedArrayType> into preCICE
        // conform format
        for(int d = 0; d < data_dim; ++d)
          for(unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
            unrolled_local_data[d + data_dim * v] = local_data[d][v];

#ifdef EXADG_WITH_PRECICE
        this->precice->writeBlockVectorData(write_data_id,
                                            active_faces,
                                            index->data(),
                                            unrolled_local_data.data());
#else
        (void)active_faces;
        (void)write_data_id;
#endif
      }
      else
      {
#ifdef EXADG_WITH_PRECICE
        this->precice->writeBlockScalarData(write_data_id,
                                            active_faces,
                                            index->data(),
                                            &local_data[0]);
#endif
      }
    }
  }
}



template<int dim, int data_dim, typename VectorizedArrayType>
std::string
QuadCoupling<dim, data_dim, VectorizedArrayType>::get_surface_type() const
{
  return "quadrature points using matrix-free quad index " +
         dealii::Utilities::to_string(mf_quad_index);
}

} // namespace preCICE
} // namespace ExaDG

#endif
