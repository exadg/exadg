#include "velocity_convection_diffusion_operator.h"

#include <navierstokes/config.h>

#include "../infrastructure/fe_evaluation_wrapper.h"
#include "../infrastructure/fe_parameters.h"

#include "../../solvers_and_preconditioners/util/block_jacobi_matrices.h"

namespace IncNS
{
template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::VelocityConvDiffOperator()
  : block_jacobi_matrices_have_been_initialized(false),
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    convective_operator(nullptr),
    evaluation_time(0.0)
{
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::initialize(
  MatrixFree<dim, Number> const &           mf_data_in,
  VelocityConvDiffOperatorData<dim> const & operator_data_in,
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const &
                                                                                       mass_matrix_operator_in,
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const &    viscous_operator_in,
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const & convective_operator_in)
{
  // copy parameters into element variables
  this->data                 = &mf_data_in;
  this->operator_data        = operator_data_in;
  this->mass_matrix_operator = &mass_matrix_operator_in;
  this->viscous_operator     = &viscous_operator_in;
  this->convective_operator  = &convective_operator_in;

  if(operator_data.convective_problem == true)
  {
    this->initialize_dof_vector(velocity_linearization);
  }
  
  // mass matrix term: set scaling factor time derivative term
  set_scaling_factor_time_derivative_term(this->operator_data.scaling_factor_time_derivative_term);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::reinit(
  const DoFHandler<dim> & dof_handler,
  const Mapping<dim> &    mapping,
  void *                  operator_data_in,
  const MGConstrainedDoFs & /*mg_constrained_dofs*/,
  const unsigned int level)
{
  // create copy of data and ...
  auto operator_data = *static_cast<VelocityConvDiffOperatorData<dim> *>(operator_data_in);

  // setup own matrix free object

  // dof_handler
  std::vector<const DoFHandler<dim> *> dof_handler_vec;
  dof_handler_vec.resize(1);
  dof_handler_vec[0] = &dof_handler;

  // constraint matrix
  std::vector<const ConstraintMatrix *> constraint_matrix_vec;
  constraint_matrix_vec.resize(1);
  ConstraintMatrix constraints;
  constraints.close();
  constraint_matrix_vec[0] = &constraints;

  // quadratures
  std::vector<Quadrature<1>> quadrature_vec;
  quadrature_vec.resize(2);
  quadrature_vec[0] = QGauss<1>(dof_handler.get_fe().degree + 1);
  quadrature_vec[1] = QGauss<1>(dof_handler.get_fe().degree + (dof_handler.get_fe().degree + 2) / 2);

  // additional data
  typename MatrixFree<dim, Number>::AdditionalData addit_data;
  addit_data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
  if(dof_handler.get_fe().dofs_per_vertex == 0)
    addit_data.build_face_info = true;

  addit_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);

  addit_data.mapping_update_flags_inner_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);

  addit_data.mapping_update_flags_boundary_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);


  addit_data.level_mg_handler = level;

  // reinit
  own_matrix_free_storage.reinit(mapping, dof_handler_vec, constraint_matrix_vec, quadrature_vec, addit_data);


  // setup own mass matrix operator
  MassMatrixOperatorData & mass_matrix_operator_data = operator_data.mass_matrix_operator_data;
  // set dof index to zero since matrix free object only contains one dof-handler
  mass_matrix_operator_data.dof_index = 0;
  own_mass_matrix_operator_storage.initialize(own_matrix_free_storage, mass_matrix_operator_data);


  // setup own viscous operator
  ViscousOperatorData<dim> & viscous_operator_data = operator_data.viscous_operator_data;
  // set dof index to zero since matrix free object only contains one dof-handler
  viscous_operator_data.dof_index = 0;
  own_viscous_operator_storage.initialize(mapping, own_matrix_free_storage, viscous_operator_data);


  // setup own convective operator
  ConvectiveOperatorData<dim> & convective_operator_data = operator_data.convective_operator_data;
  // set dof index to zero since matrix free object only contains one dof-handler
  convective_operator_data.dof_index = 0;
  own_convective_operator_storage.initialize(own_matrix_free_storage, convective_operator_data);

  // setup velocity convection diffusion operator
  initialize(own_matrix_free_storage,
             operator_data,
             own_mass_matrix_operator_storage,
             own_viscous_operator_storage,
             own_convective_operator_storage);

  // Initialize other variables:

  // mass matrix term: set scaling factor time derivative term
  set_scaling_factor_time_derivative_term(this->operator_data.scaling_factor_time_derivative_term);

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

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  set_scaling_factor_time_derivative_term(double const & factor)
{
  this->scaling_factor_time_derivative_term = factor;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
double
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  get_scaling_factor_time_derivative_term() const
{
  return this->scaling_factor_time_derivative_term;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  set_solution_linearization(parallel::distributed::Vector<Number> const & solution_linearization)
{
  velocity_linearization = solution_linearization;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
parallel::distributed::Vector<Number> &
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  get_solution_linearization() const
{
  AssertThrow(operator_data.convective_problem == true,
              ExcMessage("Attempt to access velocity_linearization which has not been initialized."));

  return velocity_linearization;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::set_evaluation_time(
  double const & evaluation_time_in)
{
  evaluation_time = evaluation_time_in;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
double
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_evaluation_time()
  const
{
  return evaluation_time;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
VelocityConvDiffOperatorData<dim> const &
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_operator_data() const
{
  return this->operator_data;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
HelmholtzOperatorData<dim> const
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  get_helmholtz_operator_data() const
{
  HelmholtzOperatorData<dim> helmholtz_operator_data;
  helmholtz_operator_data.unsteady_problem = this->operator_data.unsteady_problem;
  helmholtz_operator_data.dof_index        = this->operator_data.dof_index;

  return helmholtz_operator_data;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
MassMatrixOperatorData const &
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  get_mass_matrix_operator_data() const
{
  return mass_matrix_operator->get_operator_data();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
ConvectiveOperatorData<dim> const &
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  get_convective_operator_data() const
{
  return convective_operator->get_operator_data();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
ViscousOperatorData<dim> const &
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  get_viscous_operator_data() const
{
  return viscous_operator->get_operator_data();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  set_zero_mean_value(parallel::distributed::Vector<Number> & /*vec*/) const
{
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_interface_down(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  vmult(dst, src);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_add_interface_up(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  vmult_add(dst, src);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
types::global_dof_index
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::m() const
{
  return data->get_vector_partitioner(get_dof_index())->size();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
Number
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::el(
  const unsigned int,
  const unsigned int) const
{
  AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
MatrixFree<dim, Number> const &
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_data() const
{
  return *data;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(
      this->get_scaling_factor_time_derivative_term() > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

    mass_matrix_operator->apply(dst, src);
    dst *= this->get_scaling_factor_time_derivative_term();
  }
  else
  {
    dst = 0.0;
  }

  viscous_operator->apply_add(dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_linearized_add(dst, src, &velocity_linearization, evaluation_time);
  }
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_add(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(
      this->get_scaling_factor_time_derivative_term() > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

    mass_matrix_operator->apply(temp_vector, src);
    temp_vector *= this->get_scaling_factor_time_derivative_term();
    dst += temp_vector;
  }

  viscous_operator->apply_add(dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_linearized_add(dst, src, &velocity_linearization, evaluation_time);
  }
}


template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_block_jacobi(
  parallel::distributed::Vector<Number> &       dst,
  parallel::distributed::Vector<Number> const & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(
      this->get_scaling_factor_time_derivative_term() > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

    mass_matrix_operator->apply(dst, src);
    dst *= this->get_scaling_factor_time_derivative_term();
  }
  else
  {
    dst = 0.0;
  }

  viscous_operator->apply_block_jacobi_add(dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_linearized_block_jacobi_add(dst,
                                                           src,
                                                           &velocity_linearization,
                                                           evaluation_time);
  }
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
unsigned int
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_dof_index() const
{
  return operator_data.dof_index;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::initialize_dof_vector(
  parallel::distributed::Vector<Number> & vector) const
{
  data->initialize_dof_vector(vector, get_dof_index());
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  calculate_inverse_diagonal(parallel::distributed::Vector<Number> & diagonal) const
{
  calculate_diagonal(diagonal);

  // verify_calculation_of_diagonal(*this,diagonal);

  invert_diagonal(diagonal);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::apply_block_jacobi(
  parallel::distributed::Vector<Number> &       dst,
  parallel::distributed::Vector<Number> const & src) const
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

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::update_block_jacobi()
  const
{
  if(block_jacobi_matrices_have_been_initialized == false)
  {
    // Note that the velocity has dim components.
    unsigned int dofs_per_cell = data->get_shape_info().dofs_per_component_on_cell * dim;

    matrices.resize(data->n_macro_cells() * VectorizedArray<Number>::n_array_elements,
                    LAPACKFullMatrix<Number>(dofs_per_cell, dofs_per_cell));

    block_jacobi_matrices_have_been_initialized = true;
  }

  calculate_block_jacobi_matrices();
  calculate_lu_factorization_block_jacobi(matrices);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::calculate_diagonal(
  parallel::distributed::Vector<Number> & diagonal) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(
      this->get_scaling_factor_time_derivative_term() > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

    mass_matrix_operator->calculate_diagonal(diagonal);
    diagonal *= this->get_scaling_factor_time_derivative_term();
  }
  else
  {
    diagonal = 0.0;
  }

  viscous_operator->add_diagonal(diagonal);

  if(operator_data.convective_problem == true)
  {
    convective_operator->add_diagonal(diagonal, &velocity_linearization, evaluation_time);
  }
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  calculate_block_jacobi_matrices() const
{
  // initialize block Jacobi matrices with zeros
  initialize_block_jacobi_matrices_with_zero(matrices);

  // calculate block Jacobi matrices
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage(
                  "Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

    mass_matrix_operator->add_block_jacobi_matrices(matrices);

    for(typename std::vector<LAPACKFullMatrix<Number>>::iterator it = matrices.begin(); it != matrices.end();
        ++it)
    {
      (*it) *= this->get_scaling_factor_time_derivative_term();
    }
  }

  viscous_operator->add_block_jacobi_matrices(matrices);

  if(operator_data.convective_problem == true)
  {
    convective_operator->add_block_jacobi_matrices(matrices, &velocity_linearization, evaluation_time);
  }
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  cell_loop_apply_inverse_block_jacobi_matrices(
    MatrixFree<dim, Number> const &               data,
    parallel::distributed::Vector<Number> &       dst,
    parallel::distributed::Vector<Number> const & src,
    std::pair<unsigned int, unsigned int> const & cell_range) const
{
  FEEval_Velocity_Velocity_linear fe_eval(data, viscous_operator->get_fe_param(), operator_data.dof_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);

    unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      // fill source vector
      Vector<Number> src_vector(dofs_per_cell);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        src_vector(j) = fe_eval.begin_dof_values()[j][v];

      // apply inverse matrix
      matrices[cell * VectorizedArray<Number>::n_array_elements + v].solve(src_vector, false);

      // write solution to dst-vector
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = src_vector(j);
    }

    fe_eval.set_dof_values(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  check_block_jacobi_matrices(parallel::distributed::Vector<Number> const & src) const
{
  calculate_block_jacobi_matrices();

  // test matrix-vector product for block Jacobi problem by comparing
  // matrix-free matrix-vector product and matrix-based matrix-vector product
  // (where the matrices are generated using the matrix-free implementation)
  parallel::distributed::Vector<Number> tmp1(src), tmp2(src), diff(src);
  tmp1 = 0.0;
  tmp2 = 0.0;

  // variant 1 (matrix-free)
  vmult_block_jacobi(tmp1, src);

  // variant 2 (matrix-based)
  vmult_block_jacobi_test(tmp2, src);

  diff = tmp2;
  diff.add(-1.0, tmp1);

  std::cout << "L2 norm variant 1 = " << tmp1.l2_norm() << std::endl
            << "L2 norm variant 2 = " << tmp2.l2_norm() << std::endl
            << "L2 norm v2 - v1 = " << diff.l2_norm() << std::endl
            << std::endl;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_block_jacobi_test(
  parallel::distributed::Vector<Number> &       dst,
  parallel::distributed::Vector<Number> const & src) const
{
  data->cell_loop(&This::cell_loop_apply_block_jacobi_matrices_test, this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  cell_loop_apply_block_jacobi_matrices_test(MatrixFree<dim, Number> const &               data,
                                             parallel::distributed::Vector<Number> &       dst,
                                             parallel::distributed::Vector<Number> const & src,
                                             std::pair<unsigned int, unsigned int> const & cell_range) const
{
  FEEval_Velocity_Velocity_linear fe_eval(data, viscous_operator->get_fe_param(), operator_data.dof_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);

    unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      // fill source vector
      Vector<Number> src_vector(dofs_per_cell);
      Vector<Number> dst_vector(dofs_per_cell);
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        src_vector(j) = fe_eval.begin_dof_values()[j][v];

      // apply matrix-vector product
      matrices[cell * VectorizedArray<Number>::n_array_elements + v].vmult(dst_vector, src_vector, false);

      // write solution to dst-vector
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = dst_vector(j);
    }

    fe_eval.set_dof_values(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
MultigridOperatorBase<dim, Number> *
VelocityConvDiffOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_new(
  unsigned int deg) const
{
  switch(deg)
  {
// clang-format off      
#if DEGREE_1
    case 1: return new VelocityConvDiffOperator<dim, 1, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_2
    case 2: return new VelocityConvDiffOperator<dim, 2, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_3
    case 3: return new VelocityConvDiffOperator<dim, 3, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_4
    case 4: return new VelocityConvDiffOperator<dim, 4, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_5
    case 5: return new VelocityConvDiffOperator<dim, 5, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_6
    case 6: return new VelocityConvDiffOperator<dim, 6, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_7
    case 7: return new VelocityConvDiffOperator<dim, 7, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_8
    case 8: return new VelocityConvDiffOperator<dim, 8, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_9
    case 9: return new VelocityConvDiffOperator<dim, 9, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_10
    case 10: return new VelocityConvDiffOperator<dim, 10, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_11
    case 11: return new VelocityConvDiffOperator<dim, 11, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_12
    case 12: return new VelocityConvDiffOperator<dim, 12, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_13
    case 13: return new VelocityConvDiffOperator<dim, 13, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_14
    case 14: return new VelocityConvDiffOperator<dim, 14, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
#if DEGREE_15
    case 15: return new VelocityConvDiffOperator<dim, 15, fe_degree_xwall, xwall_quad_rule, Number>();
#endif
// clang-format on
    default:
      AssertThrow(false, ExcMessage("ConvectionDiffusionOperator not implemented for this degree!"));
      return nullptr;
      // dummy return (statement not reached)
  }
}

} // namespace IncNS

#include "velocity_convection_diffusion_operator.hpp"
