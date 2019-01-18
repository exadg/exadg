/*
 * compatible_laplace_operator.h
 *
 *  Created on: Jul 18, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_OPERATOR_H_

#include "../../functionalities/set_zero_mean_value.h"
#include "../../operators/operator_preconditionable_dummy.h"
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"

#include "../spatial_discretization/operators/divergence_operator.h"
#include "../spatial_discretization/operators/gradient_operator.h"

namespace IncNS
{
template<int dim>
struct CompatibleLaplaceOperatorData : public PreconditionableOperatorData<dim>
{
  CompatibleLaplaceOperatorData()
    : dof_index_velocity(0), dof_index_pressure(1), dof_handler_u(nullptr)
  {
  }

  unsigned int                dof_index_velocity;
  unsigned int                dof_index_pressure;
  const DoFHandler<dim> *     dof_handler_u;
  GradientOperatorData<dim>   gradient_operator_data;
  DivergenceOperatorData<dim> divergence_operator_data;
};

template<int dim, int degree_u, int degree_p, typename Number = double>
class CompatibleLaplaceOperator : public PreconditionableOperatorDummy<dim, Number>
{
public:
  static const int DIM = dim;
  typedef Number   value_type;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  CompatibleLaplaceOperator();

  void
  initialize(
    MatrixFree<dim, Number> const &                             mf_data_in,
    CompatibleLaplaceOperatorData<dim> const &                  compatible_laplace_operator_data_in,
    GradientOperator<dim, degree_u, degree_p, Number> const &   gradient_operator_in,
    DivergenceOperator<dim, degree_u, degree_p, Number> const & divergence_operator_in,
    InverseMassMatrixOperator<dim, degree_u, Number> const &    inv_mass_matrix_operator_in) const;



  void
  reinit_preconditionable_operator_data(
    MatrixFree<dim, Number> const &           matrix_free,
    AffineConstraints<double> const &         constraint_matrix,
    PreconditionableOperatorData<dim> const & operator_data_in) const
  {
    auto operator_data =
      *static_cast<CompatibleLaplaceOperatorData<dim> const *>(&operator_data_in);
    this->reinit(matrix_free, constraint_matrix, operator_data);
  }


  void
  reinit(MatrixFree<dim, Number> const &            data,
         AffineConstraints<double> const &          constraint_matrix,
         CompatibleLaplaceOperatorData<dim> const & operator_data) const;

  bool
  is_singular() const;

  void
  disable_mean_value_constraint() const;

  // apply matrix vector multiplication
  void
  vmult(VectorType & dst, VectorType const & src) const;

  void
  Tvmult(VectorType & dst, VectorType const & src) const;

  void
  Tvmult_add(VectorType & dst, VectorType const & src) const;

  void
  vmult_add(VectorType & dst, VectorType const & src) const;

  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const;

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const;

  types::global_dof_index
  m() const;

  types::global_dof_index
  n() const;

  Number
  el(const unsigned int, const unsigned int) const;

  MatrixFree<dim, Number> const &
  get_data() const;

  void
  calculate_diagonal(VectorType & diagonal) const;

  void
  calculate_inverse_diagonal(VectorType & diagonal) const;

  void
  initialize_dof_vector(VectorType & vector) const;

  void
  initialize_dof_vector_pressure(VectorType & vector) const;

  void
  initialize_dof_vector_velocity(VectorType & vector) const;

  /*
   *  Apply block Jacobi preconditioner
   */
  void
  apply_inverse_block_diagonal(VectorType & /*dst*/, VectorType const & /*src*/) const;

  /*
   *  Update block Jacobi preconditioner
   */
  void
  update_inverse_block_diagonal() const;

  PreconditionableOperator<dim, Number> *
  get_new(unsigned int deg) const;

private:
  mutable MatrixFree<dim, Number> const *                           data;
  mutable GradientOperator<dim, degree_u, degree_p, Number> const * gradient_operator;

  mutable DivergenceOperator<dim, degree_u, degree_p, Number> const * divergence_operator;

  mutable InverseMassMatrixOperator<dim, degree_u, Number> const * inv_mass_matrix_operator;

  mutable CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data;

  VectorType mutable tmp;

  /*
   * The following variables are necessary when applying the multigrid preconditioner to the
   * compatible Laplace operator In that case, the CompatibleLaplaceOperator has to be generated for
   * each level of the multigrid algorithm. Accordingly, in a first step one has to setup own
   * objects of MatrixFree, GradientOperator, DivergenceOperator, e.g.,
   * own_matrix_free_storage.reinit(...); and later initialize the CompatibleLaplaceOperator with
   * these ojects by setting the above pointers to the own_objects_storage, e.g., data =
   * &own_matrix_free_storage;
   */
  MatrixFree<dim, Number> own_matrix_free_storage;

  mutable GradientOperator<dim, degree_u, degree_p, Number> own_gradient_operator_storage;

  mutable DivergenceOperator<dim, degree_u, degree_p, Number> own_divergence_operator_storage;

  mutable InverseMassMatrixOperator<dim, degree_u, Number> own_inv_mass_matrix_operator_storage;

  mutable bool needs_mean_value_constraint;
  mutable bool apply_mean_value_constraint_in_matvec;

  mutable VectorType tmp_projection_vector;
};


} // namespace IncNS

#include "compatible_laplace_operator.cpp"

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_OPERATOR_H_ */
