#include "helmholtz_operator.h"

#include <navierstokes/config.h>

#include "../infrastructure/fe_evaluation_wrapper.h"
#include "../infrastructure/fe_parameters.h"

#include "../../solvers_and_preconditioners/util/block_jacobi_matrices.h"

namespace IncNS
{
template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::HelmholtzOperator()
  : block_jacobi_matrices_have_been_initialized(false),
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    wall_time(0.0),
    use_optimized_implementation(
      false) // TODO: use optimized implementation for performance measurements only
{
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
double
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_wall_time() const
{
  return wall_time;
}


template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::initialize(
  MatrixFree<dim, Number> const &    mf_data_in,
  HelmholtzOperatorData<dim> const & operator_data_in,
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const &
                                                                                    mass_matrix_operator_in,
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const & viscous_operator_in)
{
  // copy parameters into element variables
  this->data                 = &mf_data_in;
  this->operator_data        = operator_data_in;
  this->mass_matrix_operator = &mass_matrix_operator_in;
  this->viscous_operator     = &viscous_operator_in;
  
  // mass matrix term: set scaling factor time derivative term
  set_scaling_factor_time_derivative_term(this->operator_data.scaling_factor_time_derivative_term);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::reinit(
  const DoFHandler<dim> & dof_handler,
  const Mapping<dim> &    mapping,
  void *                  operator_data_in,
  const MGConstrainedDoFs & /*mg_constrained_dofs*/,
  const unsigned int level)
{
  // create copy of data and ...
  auto operator_data = *static_cast<HelmholtzOperatorData<dim> *>(operator_data_in);

  // setup own matrix free object
  const QGauss<1>                                  quad(dof_handler.get_fe().degree + 1);
  typename MatrixFree<dim, Number>::AdditionalData addit_data;
  addit_data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
  if(dof_handler.get_fe().dofs_per_vertex == 0)
    addit_data.build_face_info = true;
  addit_data.level_mg_handler = level;

  ConstraintMatrix constraints;

  // reinit
  own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

  // setup own mass matrix operator
  MassMatrixOperatorData & mass_matrix_operator_data = operator_data.mass_matrix_operator_data;
  mass_matrix_operator_data.dof_index                = 0;
  own_mass_matrix_operator_storage.initialize(own_matrix_free_storage, mass_matrix_operator_data);

  // setup own viscous operator
  ViscousOperatorData<dim> & viscous_operator_data = operator_data.viscous_operator_data;
  // set dof index to zero since matrix free object only contains one dof-handler
  viscous_operator_data.dof_index = 0;
  own_viscous_operator_storage.initialize(mapping, own_matrix_free_storage, viscous_operator_data);

  // setup Helmholtz operator
  initialize(own_matrix_free_storage,
             operator_data,
             own_mass_matrix_operator_storage,
             own_viscous_operator_storage);

  // Initialize other variables:

  // mass matrix term: set scaling factor time derivative term
  set_scaling_factor_time_derivative_term(this->operator_data.scaling_factor_time_derivative_term);

  // viscous term:
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  set_scaling_factor_time_derivative_term(double const & factor)
{
  this->scaling_factor_time_derivative_term = factor;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
double
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
  get_scaling_factor_time_derivative_term() const
{
  return this->scaling_factor_time_derivative_term;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
HelmholtzOperatorData<dim> const &
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_operator_data() const
{
  return this->operator_data;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
MassMatrixOperatorData const &
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_mass_matrix_operator_data()
  const
{
  return mass_matrix_operator->get_operator_data();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
ViscousOperatorData<dim> const &
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_viscous_operator_data() const
{
  return viscous_operator->get_operator_data();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_interface_down(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  vmult(dst, src);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_add_interface_up(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  vmult_add(dst, src);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
types::global_dof_index
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::m() const
{
  return data->get_vector_partitioner(get_dof_index())->size();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
Number
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::el(const unsigned int,
                                                                                const unsigned int) const
{
  AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
  return Number();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
MatrixFree<dim, Number> const &
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_data() const
{
  return *data;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  // TODO
  Timer timer;
  timer.restart();

  if(use_optimized_implementation == true) // optimized version (use only for performance measurements)
  {
    viscous_operator->apply_helmholtz_operator(dst, this->get_scaling_factor_time_derivative_term(), src);
  }
  else // standard implementation with modular implementation (operator by operator)
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(
        this->get_scaling_factor_time_derivative_term() > 0.0,
        ExcMessage(
          "Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->apply_scale(dst, this->get_scaling_factor_time_derivative_term(), src);
    }
    else
    {
      dst = 0.0;
    }

    viscous_operator->apply_add(dst, src);
  }

  // TODO
  wall_time += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_add(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  // TODO
  Timer timer;
  timer.restart();

  // helmholtz operator = mass_matrix_operator + viscous_operator
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage(
                  "Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

    mass_matrix_operator->apply_scale_add(dst, this->get_scaling_factor_time_derivative_term(), src);
  }

  viscous_operator->apply_add(dst, src);

  // TODO
  wall_time += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
unsigned int
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_dof_index() const
{
  return operator_data.dof_index;
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::initialize_dof_vector(
  parallel::distributed::Vector<Number> & vector) const
{
  data->initialize_dof_vector(vector, get_dof_index());
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::calculate_inverse_diagonal(
  parallel::distributed::Vector<Number> & diagonal) const
{
  calculate_diagonal(diagonal);

  // verify_calculation_of_diagonal(*this,diagonal);

  invert_diagonal(diagonal);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::apply_block_jacobi(
  parallel::distributed::Vector<Number> &       dst,
  parallel::distributed::Vector<Number> const & src) const
{
  // check_block_jacobi_matrices(src);

  data->cell_loop(&This::cell_loop_apply_inverse_block_jacobi_matrices, this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::update_block_jacobi() const
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
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::calculate_diagonal(
  parallel::distributed::Vector<Number> & diagonal) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage(
                  "Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

    mass_matrix_operator->calculate_diagonal(diagonal);
    diagonal *= this->get_scaling_factor_time_derivative_term();
  }
  else
  {
    diagonal = 0.0;
  }

  viscous_operator->add_diagonal(diagonal);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::calculate_block_jacobi_matrices()
  const
{
  // initialize block Jacobi matrices with zeros
  initialize_block_jacobi_matrices_with_zero(matrices);

  // calculate block Jacobi matrices
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage(
                  "Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

    mass_matrix_operator->add_block_diagonal_matrices(matrices);

    for(typename std::vector<LAPACKFullMatrix<Number>>::iterator it = matrices.begin(); it != matrices.end();
        ++it)
    {
      (*it) *= this->get_scaling_factor_time_derivative_term();
    }
  }

  viscous_operator->add_block_diagonal_matrices(matrices);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
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
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::check_block_jacobi_matrices(
  parallel::distributed::Vector<Number> const & src) const
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
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_block_jacobi(
  parallel::distributed::Vector<Number> &       dst,
  const parallel::distributed::Vector<Number> & src) const
{
  if(operator_data.unsteady_problem == true)
  {
    AssertThrow(this->get_scaling_factor_time_derivative_term() > 0.0,
                ExcMessage(
                  "Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

    // mass matrix operator has already "block Jacobi form" in DG
    mass_matrix_operator->apply_scale(dst, this->get_scaling_factor_time_derivative_term(), src);
  }
  else
  {
    dst = 0.0;
  }

  viscous_operator->apply_block_jacobi_add(dst, src);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::vmult_block_jacobi_test(
  parallel::distributed::Vector<Number> &       dst,
  parallel::distributed::Vector<Number> const & src) const
{
  data->cell_loop(&This::cell_loop_apply_block_jacobi_matrices_test, this, dst, src);
}

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::
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
HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>::get_new(unsigned int deg) const
{
  AssertThrow(deg==fe_degree, ExcMessage("Not compatible for p-GMG!"));
  return new HelmholtzOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>();
}

} // namespace IncNS
