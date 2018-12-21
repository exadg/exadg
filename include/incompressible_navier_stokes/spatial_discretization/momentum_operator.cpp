#include "momentum_operator.h"

#include <deal.II/distributed/tria.h>

#include "functionalities/categorization.h"

#include "../../solvers_and_preconditioners/util/block_jacobi_matrices.h"

namespace IncNS
{
template<int dim, int degree, typename Number>
MomentumOperator<dim, degree, Number>::MomentumOperator()
  : data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    convective_operator(nullptr),
    evaluation_time(0.0),
    scaling_factor_time_derivative_term(-1.0),
    block_diagonal_preconditioner_is_initialized(false)
{
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::reinit(
  MatrixFree<dim, Number> const &                 data,
  MomentumOperatorData<dim> const &               operator_data,
  MassMatrixOperator<dim, degree, Number> const & mass_matrix_operator,
  ViscousOperator<dim, degree, Number> const &    viscous_operator,
  ConvectiveOperator<dim, degree, Number> const & convective_operator)
{
  // copy parameters into element variables
  this->data                 = &data;
  this->operator_data        = operator_data;
  this->mass_matrix_operator = &mass_matrix_operator;
  this->viscous_operator     = &viscous_operator;
  this->convective_operator  = &convective_operator;

  // mass matrix term: set scaling factor time derivative term (initialization only, has to be
  // updated in case of time-dependent coefficients)
  set_scaling_factor_time_derivative_term(this->operator_data.scaling_factor_time_derivative_term);
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::reinit_multigrid(
  DoFHandler<dim> const & dof_handler,
  Mapping<dim> const &    mapping,
  void *                  operator_data_in,
  MGConstrainedDoFs const & /*mg_constrained_dofs*/,
  std::vector<GridTools::PeriodicFacePair<
    typename Triangulation<dim>::cell_iterator>> & /*periodic_face_pairs*/,
  unsigned int const level)
{
  auto operator_data = *static_cast<MomentumOperatorData<dim> *>(operator_data_in);

  // setup own matrix free object

  // dof_handler
  std::vector<const DoFHandler<dim> *> dof_handler_vec;
  dof_handler_vec.resize(1);
  dof_handler_vec[0] = &dof_handler;

  // constraint matrix
  std::vector<const AffineConstraints<double> *> constraint_matrix_vec;
  constraint_matrix_vec.resize(1);
  AffineConstraints<double> constraints;
  constraints.close();
  constraint_matrix_vec[0] = &constraints;

  // quadratures
  std::vector<Quadrature<1>> quadrature_vec;
  quadrature_vec.resize(2);
  quadrature_vec[0] = QGauss<1>(dof_handler.get_fe().degree + 1);
  quadrature_vec[1] =
    QGauss<1>(dof_handler.get_fe().degree + (dof_handler.get_fe().degree + 2) / 2);

  // additional data
  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;

  additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.mapping_update_flags_inner_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.mapping_update_flags_boundary_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.level_mg_handler = level;

  if(operator_data.use_cell_based_loops)
  {
    auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
      &dof_handler.get_triangulation());
    Categorization::do_cell_based_loops(*tria, additional_data);
  }

  // reinit
  own_matrix_free_storage.reinit(
    mapping, dof_handler_vec, constraint_matrix_vec, quadrature_vec, additional_data);


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
  // set quad index to 1 since matrix free object only contains two quadrature formulas
  convective_operator_data.quad_index = 1;
  own_convective_operator_storage.initialize(own_matrix_free_storage, convective_operator_data);

  // When solving the reaction-convection-diffusion problem, it might be possible
  // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
  // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
  // reaction-convection-diffusion operator. Accordingly, we have to reset which
  // operators should be "active" for the multigrid preconditioner, independently of
  // the actual equation type that is solved.
  AssertThrow(operator_data.mg_operator_type != MultigridOperatorType::Undefined,
              ExcMessage("Invalid parameter mg_operator_type."));

  if(operator_data.mg_operator_type == MultigridOperatorType::ReactionDiffusion)
  {
    // deactivate convective term for multigrid preconditioner
    operator_data.convective_problem = false;
  }
  else if(operator_data.mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
  {
    AssertThrow(operator_data.convective_problem == true, ExcMessage("Invalid parameter."));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  reinit(own_matrix_free_storage,
         operator_data,
         own_mass_matrix_operator_storage,
         own_viscous_operator_storage,
         own_convective_operator_storage);

  // initialize temp_vector: this is done in this function because temp_vector is only used in the
  // function vmult_add(), i.e., when using the multigrid preconditioner
  this->initialize_dof_vector(temp_vector);
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::set_scaling_factor_time_derivative_term(
  double const & factor)
{
  this->scaling_factor_time_derivative_term = factor;
}

template<int dim, int degree, typename Number>
double
MomentumOperator<dim, degree, Number>::get_scaling_factor_time_derivative_term() const
{
  return this->scaling_factor_time_derivative_term;
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::set_solution_linearization(
  VectorType const & solution_linearization) const
{
  if(operator_data.convective_problem == true)
  {
    convective_operator->set_solution_linearization(solution_linearization);
  }
}

template<int dim, int degree, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
MomentumOperator<dim, degree, Number>::get_solution_linearization() const
{
  AssertThrow(operator_data.convective_problem == true,
              ExcMessage(
                "Attempt to access velocity_linearization which has not been initialized."));

  return convective_operator->get_solution_linearization();
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::set_evaluation_time(double const & evaluation_time_in)
{
  evaluation_time = evaluation_time_in;
}

template<int dim, int degree, typename Number>
double
MomentumOperator<dim, degree, Number>::get_evaluation_time() const
{
  return evaluation_time;
}

template<int dim, int degree, typename Number>
MomentumOperatorData<dim> const &
MomentumOperator<dim, degree, Number>::get_operator_data() const
{
  return this->operator_data;
}

template<int dim, int degree, typename Number>
MassMatrixOperatorData const &
MomentumOperator<dim, degree, Number>::get_mass_matrix_operator_data() const
{
  return mass_matrix_operator->get_operator_data();
}

template<int dim, int degree, typename Number>
ConvectiveOperatorData<dim> const &
MomentumOperator<dim, degree, Number>::get_convective_operator_data() const
{
  return convective_operator->get_operator_data();
}

template<int dim, int degree, typename Number>
ViscousOperatorData<dim> const &
MomentumOperator<dim, degree, Number>::get_viscous_operator_data() const
{
  return viscous_operator->get_operator_data();
}

template<int dim, int degree, typename Number>
MatrixFree<dim, Number> const &
MomentumOperator<dim, degree, Number>::get_data() const
{
  return *data;
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::vmult(VectorType & dst, VectorType const & src) const
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
    convective_operator->apply_add(dst, src, evaluation_time);
  }
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::vmult_add(VectorType & dst, VectorType const & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->apply(temp_vector, src);
    temp_vector *= this->get_scaling_factor_time_derivative_term();
    dst += temp_vector;
  }

  viscous_operator->apply_add(dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_add(dst, src, evaluation_time);
  }
}


template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::vmult_block_jacobi(VectorType &       dst,
                                                          VectorType const & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->apply(dst, src);
    dst *= this->get_scaling_factor_time_derivative_term();
  }
  else
  {
    dst = 0.0;
  }

  viscous_operator->apply_block_diagonal_add(dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_block_diagonal_add(dst, src, evaluation_time);
  }
}

template<int dim, int degree, typename Number>
unsigned int
MomentumOperator<dim, degree, Number>::get_dof_index() const
{
  return operator_data.dof_index;
}

template<int dim, int degree, typename Number>
unsigned int
MomentumOperator<dim, degree, Number>::get_quad_index() const
{
  return operator_data.quad_index_std;
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::calculate_inverse_diagonal(VectorType & diagonal) const
{
  calculate_diagonal(diagonal);

  // verify_calculation_of_diagonal(*this,diagonal);

  invert_diagonal(diagonal);
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::calculate_diagonal(VectorType & diagonal) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

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
    convective_operator->add_diagonal(diagonal, evaluation_time);
  }
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::initialize_block_diagonal_preconditioner_matrix_free() const
{
  elementwise_operator.reset(new ELEMENTWISE_OPERATOR(*this));

  if(this->operator_data.preconditioner_block_jacobi == PreconditionerBlockDiagonal::None)
  {
    typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;
    elementwise_preconditioner.reset(new IDENTITY(elementwise_operator->get_problem_size()));
  }
  else if(this->operator_data.preconditioner_block_jacobi ==
          PreconditionerBlockDiagonal::InverseMassMatrix)
  {
    typedef Elementwise::InverseMassMatrixPreconditioner<dim, dim, degree, Number> INVERSE_MASS;

    elementwise_preconditioner.reset(
      new INVERSE_MASS(this->get_data(), this->get_dof_index(), this->get_quad_index()));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  Elementwise::IterativeSolverData iterative_solver_data;
  if(operator_data.convective_problem)
    iterative_solver_data.solver_type = Elementwise::SolverType::GMRES;
  else
    iterative_solver_data.solver_type = Elementwise::SolverType::CG;

  iterative_solver_data.solver_data = this->operator_data.block_jacobi_solver_data;

  elementwise_solver.reset(new ELEMENTWISE_SOLVER(
    *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
    *std::dynamic_pointer_cast<PRECONDITIONER_BASE>(elementwise_preconditioner),
    iterative_solver_data));
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::update_block_diagonal_preconditioner() const
{
  // initialization

  if(!block_diagonal_preconditioner_is_initialized)
  {
    if(operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      initialize_block_diagonal_preconditioner_matrix_free();
    }
    else // matrix-based variant
    {
      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = data->get_shape_info().dofs_per_component_on_cell * dim;

      matrices.resize(data->n_macro_cells() * VectorizedArray<value_type>::n_array_elements,
                      LAPACKFullMatrix<value_type>(dofs_per_cell, dofs_per_cell));
    }

    block_diagonal_preconditioner_is_initialized = true;
  }

  // update

  // For the matrix-free variant there is nothing to do.
  // For the matrix-based variant we have to recompute the block matrices.
  if(!operator_data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // clear matrices
    initialize_block_jacobi_matrices_with_zero(matrices);

    // compute block matrices and add
    this->add_block_diagonal_matrices(matrices);

    // check_block_jacobi_matrices();

    calculate_lu_factorization_block_jacobi(matrices);
  }
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::apply_inverse_block_diagonal(VectorType &       dst,
                                                                    VectorType const & src) const
{
  // matrix-free
  if(this->operator_data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // Solve block Jacobi problems iteratively using an elementwise solver vectorized
    // over several elements.
    elementwise_solver->solve(dst, src);
  }
  else // matrix based
  {
    // Simply apply inverse of block matrices (using the LU factorization that has been computed
    // before).
    data->cell_loop(&This::cell_loop_apply_inverse_block_diagonal, this, dst, src);
  }
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::add_block_diagonal_matrices(
  std::vector<LAPACKFullMatrix<value_type>> & matrices) const
{
  // calculate block Jacobi matrices
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->add_block_diagonal_matrices(matrices);

    for(typename std::vector<LAPACKFullMatrix<Number>>::iterator it = matrices.begin();
        it != matrices.end();
        ++it)
    {
      (*it) *= this->get_scaling_factor_time_derivative_term();
    }
  }

  viscous_operator->add_block_diagonal_matrices(matrices);

  if(operator_data.convective_problem == true)
  {
    convective_operator->add_block_diagonal_matrices(matrices, evaluation_time);
  }
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::apply_add_block_diagonal_elementwise(
  unsigned int const                    cell,
  VectorizedArray<Number> * const       dst,
  VectorizedArray<Number> const * const src,
  unsigned int const                    problem_size) const
{
  // calculate block Jacobi matrices
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been set!"));

    mass_matrix_operator->apply_add_block_diagonal_elementwise(cell, dst, src);

    Elementwise::scale(dst, this->get_scaling_factor_time_derivative_term(), problem_size);
  }

  viscous_operator->apply_add_block_diagonal_elementwise(cell, dst, src);

  if(operator_data.convective_problem == true)
  {
    convective_operator->apply_add_block_diagonal_elementwise(cell, dst, src);
  }
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::cell_loop_apply_inverse_block_diagonal(
  MatrixFree<dim, Number> const &               data,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index_std);

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

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::check_block_jacobi_matrices() const
{
  VectorType src;
  this->initialize_dof_vector(src);

  // fill vector with values unequal zero
  for(unsigned int i = 0; i < src.local_size(); ++i)
    src.local_element(i) = i % 11;

  // test matrix-vector product for block Jacobi problem by comparing
  // matrix-free matrix-vector product and matrix-based matrix-vector product
  // (where the matrices are generated using the matrix-free implementation)
  VectorType tmp1(src), tmp2(src), diff(src);
  tmp1 = 0.0;
  tmp2 = 0.0;

  // variant 1 (matrix-free)
  vmult_block_jacobi(tmp1, src);

  // variant 2 (matrix-based)
  data->cell_loop(&This::cell_loop_apply_block_diagonal, this, tmp2, src);

  diff = tmp2;
  diff.add(-1.0, tmp1);

  std::cout << "L2 norm variant 1 = " << tmp1.l2_norm() << std::endl
            << "L2 norm variant 2 = " << tmp2.l2_norm() << std::endl
            << "L2 norm v2 - v1 = " << diff.l2_norm() << std::endl
            << std::endl;
}

template<int dim, int degree, typename Number>
void
MomentumOperator<dim, degree, Number>::cell_loop_apply_block_diagonal(
  MatrixFree<dim, Number> const &               data,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index_std);

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
      matrices[cell * VectorizedArray<Number>::n_array_elements + v].vmult(dst_vector,
                                                                           src_vector,
                                                                           false);

      // write solution to dst-vector
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j][v] = dst_vector(j);
    }

    fe_eval.set_dof_values(dst);
  }
}

template<int dim, int degree, typename Number>
MultigridOperatorBase<dim, Number> *
MomentumOperator<dim, degree, Number>::get_new(unsigned int deg) const
{
  AssertThrow(deg == degree, ExcMessage("Not compatible with p-MG!"));
  return new MomentumOperator<dim, degree, Number>();
}

} // namespace IncNS

#include "momentum_operator.hpp"
