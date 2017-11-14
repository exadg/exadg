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
#include "solvers_and_preconditioners/invert_diagonal.h"
#include "solvers_and_preconditioners/verify_calculation_of_diagonal.h"


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
    dof_index(0)
  {}

  bool unsteady_problem;
  bool convective_problem;
  unsigned int dof_index;
};

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule,typename Number = double>
class VelocityConvDiffOperator : public MatrixOperatorBase
{
public:
  typedef Number value_type;

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,
                              dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  typedef VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> This;

  VelocityConvDiffOperator()
    :
    block_jacobi_matrices_have_been_initialized(false),
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    convective_operator(nullptr),
    scaling_factor_time_derivative_term(-1.0),
    evaluation_time(0.0)
  {}

  void initialize(MatrixFree<dim,Number> const                                                        &mf_data_in,
                  VelocityConvDiffOperatorData<dim> const                                             &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const &mass_matrix_operator_in,
                  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const     &viscous_operator_in,
                  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const  &convective_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->operator_data = operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->viscous_operator = &viscous_operator_in;
    this->convective_operator = &convective_operator_in;

    if(operator_data.convective_problem == true)
    {
      this->initialize_dof_vector(velocity_linearization);
    }
  }

  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. To construct the matrices, and object of
   *  type UnderlyingOperator is used that provides all the information for
   *  the setup, i.e., the information that is needed to call the
   *  member function initialize(...).
   */
  template<typename UnderlyingOperator>
  void initialize_mg_matrix (unsigned int const                               level,
                             DoFHandler<dim> const                            &dof_handler,
                             Mapping<dim> const                               &mapping,
                             UnderlyingOperator const                         &underlying_operator,
                             std::vector<GridTools::PeriodicFacePair<typename
                               Triangulation<dim>::cell_iterator> > const     &/*periodic_face_pairs_level0*/)
  {
    // setup own matrix free object

    // dof_handler
    std::vector<const DoFHandler<dim> * >  dof_handler_vec;
    dof_handler_vec.resize(1);
    dof_handler_vec[0] = &dof_handler;

    // constraint matrix
    std::vector<const ConstraintMatrix *> constraint_matrix_vec;
    constraint_matrix_vec.resize(1);
    ConstraintMatrix constraints;
    constraints.close();
    constraint_matrix_vec[0] = &constraints;

    // quadratures
    std::vector<Quadrature<1> > quadrature_vec;
    quadrature_vec.resize(2);
    quadrature_vec[0] = QGauss<1>(dof_handler.get_fe().degree+1);
    quadrature_vec[1] = QGauss<1>(dof_handler.get_fe().degree+(dof_handler.get_fe().degree+2)/2);

    // additional data
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;

    addit_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                       update_quadrature_points | update_normal_vectors |
                                       update_values);

    addit_data.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values |
                                                   update_quadrature_points | update_normal_vectors |
                                                   update_values);

    addit_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values |
                                                      update_quadrature_points | update_normal_vectors |
                                                      update_values);


    addit_data.level_mg_handler = level;

    // reinit
    own_matrix_free_storage.reinit(mapping, dof_handler_vec, constraint_matrix_vec, quadrature_vec, addit_data);


    // setup own mass matrix operator
    MassMatrixOperatorData mass_matrix_operator_data = underlying_operator.get_mass_matrix_operator_data();
    // set dof index to zero since matrix free object only contains one dof-handler
    mass_matrix_operator_data.dof_index = 0;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage, mass_matrix_operator_data);


    // setup own viscous operator
    ViscousOperatorData<dim> viscous_operator_data = underlying_operator.get_viscous_operator_data();
    // set dof index to zero since matrix free object only contains one dof-handler
    viscous_operator_data.dof_index = 0;
    own_viscous_operator_storage.initialize(mapping,own_matrix_free_storage, viscous_operator_data);


    // setup own convective operator
    ConvectiveOperatorData<dim> convective_operator_data = underlying_operator.get_convective_operator_data();
    // set dof index to zero since matrix free object only contains one dof-handler
    convective_operator_data.dof_index = 0;
    own_convective_operator_storage.initialize(own_matrix_free_storage, convective_operator_data);

    // setup velocity convection diffusion operator
    VelocityConvDiffOperatorData<dim> operator_data = underlying_operator.get_velocity_conv_diff_operator_data();
    initialize(own_matrix_free_storage,
               operator_data,
               own_mass_matrix_operator_storage,
               own_viscous_operator_storage,
               own_convective_operator_storage);

    // Initialize other variables:

    // mass matrix term: set scaling factor time derivative term
    set_scaling_factor_time_derivative_term(underlying_operator.get_scaling_factor_time_derivative_term());

    // convective term: evaluation_time and velocity_linearization
    // Note that velocity_linearization has already
    // been initialized in function initialize().
    // These variables are not set here. If the convective term
    // is considered, these variables have to be updated anyway,
    // which is done somewhere else.

    // viscous term:

    // initialize temp_vector: this is done in this function because
    // temp_vector is only used in the function vmult_add(), i.e.,
    // when using the multigrid preconditioner
    initialize_dof_vector(temp_vector);
  }

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void set_scaling_factor_time_derivative_term(double const &factor)
  {
    scaling_factor_time_derivative_term = factor;
  }

  double get_scaling_factor_time_derivative_term() const
  {
    return scaling_factor_time_derivative_term;
  }

  /*
   *  Linearized velocity field for convective operator
   */
  void set_solution_linearization(parallel::distributed::Vector<Number> const &solution_linearization)
  {
    velocity_linearization = solution_linearization;
  }

  parallel::distributed::Vector<value_type> & get_solution_linearization() const
  {
    AssertThrow(operator_data.convective_problem == true,
      ExcMessage("Attempt to access velocity_linearization which has not been initialized."));

    return velocity_linearization;
  }

  /*
   *  Evaluation time that is needed for evaluation of linearized convective operator.
   */
  void set_evaluation_time(double const &evaluation_time_in)
  {
    evaluation_time = evaluation_time_in;
  }

  double get_evaluation_time() const
  {
    return evaluation_time;
  }

  /*
   *  Operator data
   */
  VelocityConvDiffOperatorData<dim> const & get_velocity_conv_diff_operator_data() const
  {
    return this->operator_data;
  }

  /*
   *  This function is needed to initialize the multigrid matrices
   *  for the HelmholtzOperator using VelocityConvDiffOperator as
   *  underlying operator.
   */
  HelmholtzOperatorData<dim> const get_helmholtz_operator_data() const
  {
    HelmholtzOperatorData<dim> helmholtz_operator_data;
    helmholtz_operator_data.unsteady_problem = this->operator_data.unsteady_problem;
    helmholtz_operator_data.dof_index = this->operator_data.dof_index;

    return helmholtz_operator_data;
  }

  /*
   *  Operator data of basic operators: mass matrix, convective operator, viscous operator
   */
  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator->get_operator_data();
  }

  ConvectiveOperatorData<dim> const & get_convective_operator_data() const
  {
    return convective_operator->get_operator_data();
  }

  ViscousOperatorData<dim> const & get_viscous_operator_data() const
  {
    return viscous_operator->get_operator_data();
  }

  /*
   *  This function does nothing in case of the velocity conv diff operator.
   *  It is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const {}

  /*
   *  Other function needed in order to apply geometric multigrid to this operator
   */
  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const
  {
    vmult(dst,src);
  }

  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              const parallel::distributed::Vector<Number> &src) const
  {
    vmult_add(dst,src);
  }

  types::global_dof_index m() const
  {
    return data->get_vector_partitioner(get_dof_index())->size();
  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  MatrixFree<dim,value_type> const & get_data() const
  {
    return *data;
  }

  /*
   *  This function applies the matrix vector multiplication.
   */
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    viscous_operator->apply_add(dst,src);

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_linearized_add(dst,src,&velocity_linearization,evaluation_time);
    }
  }

  /*
   *  This function applies matrix vector product and adds the result
   *  to the dst-vector.
   */
  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

      mass_matrix_operator->apply(temp_vector,src);
      temp_vector *= scaling_factor_time_derivative_term;
      dst += temp_vector;
    }

    viscous_operator->apply_add(dst,src);

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_linearized_add(dst,src,&velocity_linearization,evaluation_time);
    }
  }


  /*
   *  This function applies the matrix-vector multiplication for the block Jacobi operation.
   */
  void vmult_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    viscous_operator->apply_block_jacobi_add(dst,src);

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_linearized_block_jacobi_add(dst,src,&velocity_linearization,evaluation_time);
    }
  }

  unsigned int get_dof_index() const
  {
    return operator_data.dof_index;
  }

  /*
   *  This function initializes a global dof-vector.
   */
  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,get_dof_index());
  }

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Apply block Jacobi preconditioner.
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const
  {
    // VARIANT 1: solve global block Jacobi problem iteratively
    /*
    IterationNumberControl control (30,1.e-20,1.e-6);
    typename SolverGMRES<parallel::distributed::Vector<Number> >::AdditionalData additional_data;
    additional_data.right_preconditioning = true;
    SolverGMRES<parallel::distributed::Vector<Number> > solver (control,additional_data);

    typedef VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> MY_TYPE;
    VelocityConvectionDiffusionBlockJacobiOperator<MY_TYPE, Number> block_jacobi_operator(*this);

    dst = 0.0;
    solver.solve(block_jacobi_operator,dst,src,PreconditionIdentity());
    // std::cout<<"Number of iterations block Jacobi solve = "<<control.last_step()<<std::endl;
    */

    // VARIANT 2: calculate block jacobi matrices and solve block Jacobi problem
    // elementwise using a direct solver

//    check_block_jacobi_matrices(src);

    data->cell_loop(&This::cell_loop_apply_inverse_block_jacobi_matrices, this, dst, src);
  }


  /*
   *  This function updates the block Jacobi preconditioner.
   *  Since this function also initializes the block Jacobi preconditioner,
   *  make sure that the block Jacobi matrices are allocated before calculating
   *  the matrices and the LU factorization.
   */
  void update_block_jacobi () const
  {
    if(block_jacobi_matrices_have_been_initialized == false)
    {
      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = data->get_shape_info().dofs_per_component_on_cell*dim;

      matrices.resize(data->n_macro_cells()*VectorizedArray<Number>::n_array_elements,
        LAPACKFullMatrix<Number>(dofs_per_cell, dofs_per_cell));

      block_jacobi_matrices_have_been_initialized = true;
    }

    calculate_block_jacobi_matrices();
    calculate_lu_factorization_block_jacobi(matrices);
  }

private:
  /*
   *  This function calculates the diagonal of the discrete operator representing the
   *  velocity convection-diffusion operator.
   */
  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {

    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

      mass_matrix_operator->calculate_diagonal(diagonal);
      diagonal *= scaling_factor_time_derivative_term;
    }
    else
    {
      diagonal = 0.0;
    }

    viscous_operator->add_diagonal(diagonal);

    if(operator_data.convective_problem == true)
    {
      convective_operator->add_diagonal(diagonal,&velocity_linearization,evaluation_time);
    }
  }

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void calculate_block_jacobi_matrices() const
  {
    // initialize block Jacobi matrices with zeros
    initialize_block_jacobi_matrices_with_zero(matrices);

    // calculate block Jacobi matrices
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->add_block_jacobi_matrices(matrices);

      for(typename std::vector<LAPACKFullMatrix<Number> >::iterator
          it = matrices.begin(); it != matrices.end(); ++it)
      {
        (*it) *= scaling_factor_time_derivative_term;
      }
    }

    viscous_operator->add_block_jacobi_matrices(matrices);

    if(operator_data.convective_problem == true)
    {
      convective_operator->add_block_jacobi_matrices(matrices,&velocity_linearization,evaluation_time);
    }
  }

  /*
   *  This function loops over all cells and applies the inverse block Jacobi matrices elementwise.
   */
  void cell_loop_apply_inverse_block_jacobi_matrices (MatrixFree<dim,Number> const                &data,
                                                      parallel::distributed::Vector<Number>       &dst,
                                                      parallel::distributed::Vector<Number> const &src,
                                                      std::pair<unsigned int,unsigned int> const  &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,viscous_operator->get_fe_param(),operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<Number> src_vector(dofs_per_cell);
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply inverse matrix
        matrices[cell*VectorizedArray<Number>::n_array_elements+v].apply_lu_factorization(src_vector,false);

        // write solution to dst-vector
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = src_vector(j);
      }

      fe_eval.set_dof_values (dst);
    }
  }

  /*
   * Verify computation of block Jacobi matrices.
   */
  void check_block_jacobi_matrices(parallel::distributed::Vector<Number> const &src) const
  {
    calculate_block_jacobi_matrices();

    // test matrix-vector product for block Jacobi problem by comparing
    // matrix-free matrix-vector product and matrix-based matrix-vector product
    // (where the matrices are generated using the matrix-free implementation)
    parallel::distributed::Vector<Number> tmp1(src), tmp2(src), diff(src);
    tmp1 = 0.0;
    tmp2 = 0.0;

    // variant 1 (matrix-free)
    vmult_block_jacobi(tmp1,src);

    // variant 2 (matrix-based)
    vmult_block_jacobi_test(tmp2,src);

    diff = tmp2;
    diff.add(-1.0,tmp1);

    std::cout << "L2 norm variant 1 = " << tmp1.l2_norm() << std::endl
              << "L2 norm variant 2 = " << tmp2.l2_norm() << std::endl
              << "L2 norm v2 - v1 = " << diff.l2_norm() << std::endl << std::endl;
  }

  /*
   * Apply matrix-vector multiplication (matrix-based) for global block Jacobi system
   * by looping over all cells and applying the matrix-based matrix-vector product cellwise.
   * This function is only needed for testing.
   */
  void vmult_block_jacobi_test (parallel::distributed::Vector<Number>       &dst,
                                parallel::distributed::Vector<Number> const &src) const
  {
    data->cell_loop(&This::cell_loop_apply_block_jacobi_matrices_test, this, dst, src);
  }

  /*
   *  This function is only needed for testing.
   */
  void cell_loop_apply_block_jacobi_matrices_test (MatrixFree<dim,Number> const                &data,
                                                   parallel::distributed::Vector<Number>       &dst,
                                                   parallel::distributed::Vector<Number> const &src,
                                                   std::pair<unsigned int,unsigned int> const  &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,viscous_operator->get_fe_param(),operator_data.dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell*dim;

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<Number> src_vector(dofs_per_cell);
        Vector<Number> dst_vector(dofs_per_cell);
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply matrix-vector product
        matrices[cell*VectorizedArray<Number>::n_array_elements+v].vmult(dst_vector,src_vector,false);

        // write solution to dst-vector
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = dst_vector(j);
      }

      fe_eval.set_dof_values (dst);
    }
  }

  mutable std::vector<LAPACKFullMatrix<Number> > matrices;
  mutable bool block_jacobi_matrices_have_been_initialized;

  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const * mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const * viscous_operator;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const * convective_operator;
  VelocityConvDiffOperatorData<dim> operator_data;
  parallel::distributed::Vector<Number> mutable temp_vector;
  double scaling_factor_time_derivative_term;
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

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VELOCITY_CONVECTION_DIFFUSION_OPERATOR_H_ */
