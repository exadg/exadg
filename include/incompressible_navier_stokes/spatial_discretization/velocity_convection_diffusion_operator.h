/*
 * VelocityConvDiffOperator.h
 *
 *  Created on: Aug 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VELOCITY_CONVECTION_DIFFUSION_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VELOCITY_CONVECTION_DIFFUSION_OPERATOR_H_

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include "../../incompressible_navier_stokes/spatial_discretization/helmholtz_operator.h"
#include "../../incompressible_navier_stokes/spatial_discretization/navier_stokes_operators.h"
#include "operators/matrix_operator_base.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"
#include "../../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"


#include "../../operators/matrix_operator_base_new.h"

#include <navier_constants.h>

namespace IncNS
{

template<typename UnderlyingOperator, typename Number>
class VelocityConvectionDiffusionBlockJacobiOperator
{
public:
  VelocityConvectionDiffusionBlockJacobiOperator(UnderlyingOperator const &underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {}

  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    underlying_operator.vmult_block_jacobi(dst,src);
  }

private:
  UnderlyingOperator const &underlying_operator;
};


template<int dim>
struct VelocityConvDiffOperatorData
{
  VelocityConvDiffOperatorData ()
    :
    unsteady_problem(true),
    convective_problem(true),
    dof_index(0),
    bc(new BoundaryDescriptorP<dim>()),
    scaling_factor_time_derivative_term(-1.0)
  {}

  bool unsteady_problem;
  bool convective_problem;
  unsigned int dof_index;
  
  // TODO
  std::shared_ptr<BoundaryDescriptorP<dim>> bc;
  double scaling_factor_time_derivative_term;
  MassMatrixOperatorData mass_matrix_operator_data;
  ViscousOperatorData<dim> viscous_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;
};

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule,typename Number = double>
class VelocityConvDiffOperator : public MatrixOperatorBaseNew<dim, Number>
{
public:
  static const int DIM = dim;
  typedef Number value_type;

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,
                              dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  typedef VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> This;

  VelocityConvDiffOperator();

  void initialize(MatrixFree<dim,Number> const                                                        &mf_data_in,
                  VelocityConvDiffOperatorData<dim> const                                             &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const &mass_matrix_operator_in,
                  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const     &viscous_operator_in,
                  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const  &convective_operator_in);

  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. To construct the matrices, and object of
   *  type UnderlyingOperator is used that provides all the information for
   *  the setup, i.e., the information that is needed to call the
   *  member function initialize(...).
   */
  void reinit (const DoFHandler<dim>                            &dof_handler,
               const Mapping<dim>                                &mapping,
               void * operator_data_in,
               const MGConstrainedDoFs &/*mg_constrained_dofs*/,
               const unsigned int level);

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void set_scaling_factor_time_derivative_term(double const &factor);

  double get_scaling_factor_time_derivative_term() const;

  /*
   *  Linearized velocity field for convective operator
   */
  void set_solution_linearization(parallel::distributed::Vector<Number> const &solution_linearization);

  parallel::distributed::Vector<value_type> & get_solution_linearization() const;

  /*
   *  Evaluation time that is needed for evaluation of linearized convective operator.
   */
  void set_evaluation_time(double const &evaluation_time_in);

  double get_evaluation_time() const;

  /*
   *  Operator data
   */
  VelocityConvDiffOperatorData<dim> const & get_operator_data() const;

  /*
   *  This function is needed to initialize the multigrid matrices
   *  for the HelmholtzOperator using VelocityConvDiffOperator as
   *  underlying operator.
   */
  HelmholtzOperatorData<dim> const get_helmholtz_operator_data() const;

  /*
   *  Operator data of basic operators: mass matrix, convective operator, viscous operator
   */
  MassMatrixOperatorData const & get_mass_matrix_operator_data() const;

  ConvectiveOperatorData<dim> const & get_convective_operator_data() const;

  ViscousOperatorData<dim> const & get_viscous_operator_data() const;

  /*
   *  This function does nothing in case of the velocity conv diff operator.
   *  It is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const;

  /*
   *  Other function needed in order to apply geometric multigrid to this operator
   */
  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const;

  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              const parallel::distributed::Vector<Number> &src) const;

  types::global_dof_index m() const;

  Number el (const unsigned int,  const unsigned int) const;

  MatrixFree<dim,value_type> const & get_data() const;

  /*
   *  This function applies the matrix vector multiplication.
   */
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const;

  /*
   *  This function applies matrix vector product and adds the result
   *  to the dst-vector.
   */
  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const;


  /*
   *  This function applies the matrix-vector multiplication for the block Jacobi operation.
   */
  void vmult_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const;

  unsigned int get_dof_index() const;

  /*
   *  This function initializes a global dof-vector.
   */
  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const;

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const;

  /*
   *  Apply block Jacobi preconditioner.
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const;


  /*
   *  This function updates the block Jacobi preconditioner.
   *  Since this function also initializes the block Jacobi preconditioner,
   *  make sure that the block Jacobi matrices are allocated before calculating
   *  the matrices and the LU factorization.
   */
  void update_block_jacobi () const;

private:
  /*
   *  This function calculates the diagonal of the discrete operator representing the
   *  velocity convection-diffusion operator.
   */
  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const;

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void calculate_block_jacobi_matrices() const;

  /*
   *  This function loops over all cells and applies the inverse block Jacobi matrices elementwise.
   */
  void cell_loop_apply_inverse_block_jacobi_matrices (MatrixFree<dim,Number> const                &data,
                                                      parallel::distributed::Vector<Number>       &dst,
                                                      parallel::distributed::Vector<Number> const &src,
                                                      std::pair<unsigned int,unsigned int> const  &cell_range) const;

  /*
   * Verify computation of block Jacobi matrices.
   */
  void check_block_jacobi_matrices(parallel::distributed::Vector<Number> const &src) const;

  /*
   * Apply matrix-vector multiplication (matrix-based) for global block Jacobi system
   * by looping over all cells and applying the matrix-based matrix-vector product cellwise.
   * This function is only needed for testing.
   */
  void vmult_block_jacobi_test (parallel::distributed::Vector<Number>       &dst,
                                parallel::distributed::Vector<Number> const &src) const;

  /*
   *  This function is only needed for testing.
   */
  void cell_loop_apply_block_jacobi_matrices_test (MatrixFree<dim,Number> const                &data,
                                                   parallel::distributed::Vector<Number>       &dst,
                                                   parallel::distributed::Vector<Number> const &src,
                                                   std::pair<unsigned int,unsigned int> const  &cell_range) const;

  virtual MatrixOperatorBaseNew<dim, Number>* get_new(unsigned int deg) const;
  
  mutable std::vector<LAPACKFullMatrix<Number> > matrices;
  mutable bool block_jacobi_matrices_have_been_initialized;

  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const * mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const * viscous_operator;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const * convective_operator;
  VelocityConvDiffOperatorData<dim> operator_data;
  parallel::distributed::Vector<Number> mutable temp_vector;
  parallel::distributed::Vector<Number> mutable velocity_linearization;
  double evaluation_time;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the velocity convection-diffusion operator.
   * In that case, the VelocityConvDiffOperator has to be generated
   * for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MatrixFree, MassMatrixOperator, ViscousOperator,
   *   e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the VelocityConvDiffOperator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_mass_matrix_operator_storage;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_viscous_operator_storage;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_convective_operator_storage;
};


}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VELOCITY_CONVECTION_DIFFUSION_OPERATOR_H_ */
