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
struct CompatibleLaplaceOperatorData
{
  CompatibleLaplaceOperatorData()
    : dof_index_velocity(0),
      dof_index_pressure(1),
      dof_handler_u(nullptr),
      underlying_operator_dof_index_velocity(0)
  {
  }

  unsigned int                dof_index_velocity;
  unsigned int                dof_index_pressure;
  const DoFHandler<dim> *     dof_handler_u;
  GradientOperatorData<dim>   gradient_operator_data;
  DivergenceOperatorData<dim> divergence_operator_data;
  unsigned int                underlying_operator_dof_index_velocity;
};

template<int dim, int fe_degree, int fe_degree_p, typename Number = double>
class CompatibleLaplaceOperator : public PreconditionableOperatorDummy<dim, Number>
{
public:
  static const int DIM = dim;
  typedef Number   value_type;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  CompatibleLaplaceOperator();

  // clang-format off
  void
  initialize(
      MatrixFree<dim,Number> const                                                                    &mf_data_in,
      CompatibleLaplaceOperatorData<dim> const                                                        &compatible_laplace_operator_data_in,
      GradientOperator<dim, fe_degree, fe_degree_p, Number>  const  &gradient_operator_in,
      DivergenceOperator<dim, fe_degree, fe_degree_p, Number> const &divergence_operator_in,
      InverseMassMatrixOperator<dim,fe_degree, Number> const                                          &inv_mass_matrix_operator_in);
  // clang-format on


  /*
   * This function is called by the multigrid algorithm to initialize the matrices on all levels. To
   * construct the matrices, and object of type UnderlyingOperator is used that provides all the
   * information for the setup, i.e., the information that is needed to call the member function
   * initialize(...).
   */
  void
  reinit(const DoFHandler<dim> & dof_handler_p,
         const Mapping<dim> &    mapping,
         void *                  operator_data_in,
         const MGConstrainedDoFs & /*mg_constrained_dofs*/,
         const unsigned int level)
  {
    // get compatible Laplace operator data
    CompatibleLaplaceOperatorData<dim> comp_laplace_operator_data =
      *static_cast<CompatibleLaplaceOperatorData<dim> *>(operator_data_in);

    unsigned int dof_index_velocity = comp_laplace_operator_data.dof_index_velocity;
    unsigned int dof_index_pressure = comp_laplace_operator_data.dof_index_pressure;

    const DoFHandler<dim> & dof_handler_u = *comp_laplace_operator_data.dof_handler_u;

    AssertThrow(dof_index_velocity == 0,
                ExcMessage("Expected that dof_index_velocity is 0."
                           " Fix implementation of CompatibleLaplaceOperator!"));

    AssertThrow(dof_index_pressure == 1,
                ExcMessage("Expected that dof_index_pressure is 1."
                           " Fix implementation of CompatibleLaplaceOperator!"));

    // setup own matrix free object

    // dof_handler
    std::vector<const DoFHandler<dim> *> dof_handler_vec;
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    dof_handler_vec.resize(2);
    dof_handler_vec[dof_index_velocity] = &dof_handler_u;
    dof_handler_vec[dof_index_pressure] = &dof_handler_p;

    // constraint matrix
    std::vector<AffineConstraints<double> const *> constraint_matrix_vec;
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    constraint_matrix_vec.resize(2);
    AffineConstraints<double> constraint_u, constraint_p;
    constraint_u.close();
    constraint_p.close();
    constraint_matrix_vec[dof_index_velocity] = &constraint_u;
    constraint_matrix_vec[dof_index_pressure] = &constraint_p;

    // quadratures:
    // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature
    // formula that is used for the gradient operator and the divergence operator (and the inverse
    // velocity mass matrix operator)
    const QGauss<1> quad(dof_handler_u.get_fe().degree + 1);

    // additional data
    typename MatrixFree<dim, Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;

    // TODO
    // continuous or discontinuous elements: discontinuous == 0
    //    if(dof_handler_p.get_fe().dofs_per_vertex == 0)
    //      addit_data.build_face_info = true;

    addit_data.level_mg_handler = level;

    // reinit
    own_matrix_free_storage.reinit(
      mapping, dof_handler_vec, constraint_matrix_vec, quad, addit_data);

    // setup own gradient operator
    GradientOperatorData<dim> gradient_operator_data =
      comp_laplace_operator_data.gradient_operator_data;
    own_gradient_operator_storage.initialize(own_matrix_free_storage, gradient_operator_data);

    // setup own divergence operator
    DivergenceOperatorData<dim> divergence_operator_data =
      comp_laplace_operator_data.divergence_operator_data;
    own_divergence_operator_storage.initialize(own_matrix_free_storage, divergence_operator_data);

    // setup own inverse mass matrix operator
    // NOTE: use quad_index = 0 since own_matrix_free_storage contains only one quadrature formula
    // (i.e. on would use quad_index = 0 also if quad_index_velocity would be 1 !)
    unsigned int quad_index = 0;
    own_inv_mass_matrix_operator_storage.initialize(
      own_matrix_free_storage,
      comp_laplace_operator_data.underlying_operator_dof_index_velocity,
      quad_index);

    // setup compatible Laplace operator
    initialize(own_matrix_free_storage,
               comp_laplace_operator_data,
               own_gradient_operator_storage,
               own_divergence_operator_storage,
               own_inv_mass_matrix_operator_storage);

    // we do not need the mean value constraint for smoothers on the
    // multigrid levels, so we can disable it
    disable_mean_value_constraint();
  }

  bool
  is_singular() const;

  void
  disable_mean_value_constraint();

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
  get_new(unsigned int deg) const
  {
    AssertThrow(deg == fe_degree_p, ExcMessage("Not compatible for p-GMG!"));

    return new CompatibleLaplaceOperator<dim, fe_degree, fe_degree_p, Number>();
  }

private:
  MatrixFree<dim, Number> const *                               data;
  GradientOperator<dim, fe_degree, fe_degree_p, Number> const * gradient_operator;

  DivergenceOperator<dim, fe_degree, fe_degree_p, Number> const * divergence_operator;

  InverseMassMatrixOperator<dim, fe_degree, Number> const * inv_mass_matrix_operator;

  CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data;

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

  GradientOperator<dim, fe_degree, fe_degree_p, Number> own_gradient_operator_storage;

  DivergenceOperator<dim, fe_degree, fe_degree_p, Number> own_divergence_operator_storage;

  InverseMassMatrixOperator<dim, fe_degree, Number> own_inv_mass_matrix_operator_storage;

  bool needs_mean_value_constraint;
  bool apply_mean_value_constraint_in_matvec;

  mutable VectorType tmp_projection_vector;
};


} // namespace IncNS

#include "compatible_laplace_operator.cpp"

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_OPERATOR_H_ */
