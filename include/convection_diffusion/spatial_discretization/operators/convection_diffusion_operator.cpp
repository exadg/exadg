#include "convection_diffusion_operator.h"

#include <navierstokes/config.h>

namespace ConvDiff
{
template<int dim, int fe_degree, typename Number>
ConvectionDiffusionOperator<dim, fe_degree, Number>::ConvectionDiffusionOperator()
{
}

template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::initialize(
  MatrixFree<dim, Number> const &              mf_data_in,
  ConvectionDiffusionOperatorData<dim> const & operator_data_in,
  MassMatrixOp const &                         mass_matrix_operator_in,
  ConvectiveOp const &                         convective_operator_in,
  DiffusiveOp const &                          diffusive_operator_in)
{
  ConstraintMatrix constraint_matrix;
  Parent::reinit(mf_data_in, constraint_matrix, operator_data_in);
  this->mass_matrix_operator.reinit(mass_matrix_operator_in);
  this->convective_operator.reinit(convective_operator_in);
  this->diffusive_operator.reinit(diffusive_operator_in);
  
  // mass matrix term: set scaling factor time derivative term
  this->scaling_factor_time_derivative_term = this->operator_settings.scaling_factor_time_derivative_term;
}

template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::reinit(const DoFHandler<dim> &   dof_handler,
                                                            const Mapping<dim> &      mapping,
                                                            void *                    operator_data_in,
                                                            const MGConstrainedDoFs & mg_constrained_dofs,
                                                            const unsigned int        level)
{
  Parent::reinit(dof_handler, mapping, operator_data_in, mg_constrained_dofs, level);

  // use own operators
  mass_matrix_operator.reset();
  convective_operator.reset();
  diffusive_operator.reset();

  // setup own mass matrix operator
  {
    auto & op_data     = this->operator_settings.mass_matrix_operator_data;
    op_data.dof_index  = 0;
    op_data.quad_index = 0;
    mass_matrix_operator.own().initialize(this->get_data(), op_data);
  }

  // setup own convective operator
  {
    auto & op_data     = this->operator_settings.convective_operator_data;
    op_data.dof_index  = 0;
    op_data.quad_index = 0;
    convective_operator.own().initialize(this->get_data(), op_data);
  }

  // setup own viscous operator
  {
    auto & op_data     = this->operator_settings.diffusive_operator_data;
    op_data.dof_index  = 0;
    op_data.quad_index = 0;
    diffusive_operator.own().initialize(mapping, this->get_data(), op_data);
  }

  // When solving the reaction-convection-diffusion equations, it might be possible
  // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
  // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
  // reaction-convection-diffusion operator. Accordingly, we have to reset which
  // operators should be "active" for the multigrid preconditioner, independently of
  // the actual equation type that is solved.
  AssertThrow(this->operator_settings.mg_operator_type != MultigridOperatorType::Undefined,
              ExcMessage("Invalid parameter mg_operator_type."));

  if(this->operator_settings.mg_operator_type == MultigridOperatorType::ReactionDiffusion)
  {
    this->operator_settings.convective_problem =
      false; // deactivate convective term for multigrid preconditioner
    this->operator_settings.diffusive_problem = true;
  }
  else if(this->operator_settings.mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
  {
    this->operator_settings.convective_problem = true;
    this->operator_settings.diffusive_problem  = true;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // Initialize other variables:

  // mass matrix term: set scaling factor time derivative term
  this->scaling_factor_time_derivative_term = this->operator_settings.scaling_factor_time_derivative_term;

  // convective term: evaluation_time
  // This variables is not set here. If the convective term
  // is considered, this variables has to be updated anyway,
  // which is done somewhere else.

  // viscous term: nothing to do



  // initialize temp vector: this is done in this function because
  // the vector temp is only used in the function vmult_add(), i.e.,
  // when using the multigrid preconditioner
  this->initialize_dof_vector(temp);
}

template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::set_scaling_factor_time_derivative_term(
  double const & factor)
{
  this->scaling_factor_time_derivative_term = factor;
}

template<int dim, int fe_degree, typename Number>
double
ConvectionDiffusionOperator<dim, fe_degree, Number>::get_scaling_factor_time_derivative_term() const
{
  return this->scaling_factor_time_derivative_term;
}

template<int dim, int fe_degree, typename Number>
MassMatrixOperatorData<dim> const &
ConvectionDiffusionOperator<dim, fe_degree, Number>::get_mass_matrix_operator_data() const
{
  return mass_matrix_operator->get_operator_data(); // TODO: get it from data
}

template<int dim, int fe_degree, typename Number>
ConvectiveOperatorData<dim> const &
ConvectionDiffusionOperator<dim, fe_degree, Number>::get_convective_operator_data() const
{
  return convective_operator->get_operator_data(); // TODO: get it from data
}

template<int dim, int fe_degree, typename Number>
DiffusiveOperatorData<dim> const &
ConvectionDiffusionOperator<dim, fe_degree, Number>::get_diffusive_operator_data() const
{
  return diffusive_operator->get_operator_data(); // TODO: get it from data
}

template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::vmult(
  parallel::distributed::Vector<Number> &       dst,
  parallel::distributed::Vector<Number> const & src) const
{
  if(this->operator_settings.unsteady_problem == true)
  {
    AssertThrow(this->operator_settings.scaling_factor_time_derivative_term > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been initialized!"));

    mass_matrix_operator->apply(dst, src);
    dst *= this->operator_settings.scaling_factor_time_derivative_term;
  }
  else
  {
    dst = 0.0;
  }

  if(this->operator_settings.diffusive_problem == true)
  {
    diffusive_operator->apply_add(dst, src);
  }

  if(this->operator_settings.convective_problem == true)
  {
    convective_operator->apply_add(dst, src, this->eval_time);
  }
}

template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::vmult_add(
  parallel::distributed::Vector<Number> &       dst,
  parallel::distributed::Vector<Number> const & src) const
{
  if(this->operator_settings.unsteady_problem == true)
  {
    AssertThrow(
      this->operator_settings.scaling_factor_time_derivative_term > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

    mass_matrix_operator->apply(temp, src);
    temp *= this->operator_settings.scaling_factor_time_derivative_term;
    dst += temp;
  }

  if(this->operator_settings.diffusive_problem == true)
  {
    diffusive_operator->apply_add(dst, src);
  }

  if(this->operator_settings.convective_problem == true)
  {
    convective_operator->apply_add(dst, src, this->eval_time);
  }
}

template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::calculate_system_matrix(
  SparseMatrix & system_matrix) const
{
  
  // clear content of matrix since the next calculate_system_matrix-commands add their result
  system_matrix*=0.0;
    
  if(this->operator_settings.unsteady_problem == true)
  {
    AssertThrow(
      this->operator_settings.scaling_factor_time_derivative_term > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

    mass_matrix_operator->calculate_system_matrix(system_matrix);
    system_matrix *= this->operator_settings.scaling_factor_time_derivative_term;
  }
  else
  {
    // nothing to do since matrix is already explicitly set to zero
  }

  if(this->operator_settings.diffusive_problem == true)
  {
    diffusive_operator->calculate_system_matrix(system_matrix);
  }

  if(this->operator_settings.convective_problem == true)
  {
    convective_operator->calculate_system_matrix(system_matrix, this->eval_time);
  }
}


template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::calculate_diagonal(
  parallel::distributed::Vector<Number> & diagonal) const
{
  if(this->operator_settings.unsteady_problem == true)
  {
    AssertThrow(
      this->operator_settings.scaling_factor_time_derivative_term > 0.0,
      ExcMessage(
        "Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

    mass_matrix_operator->calculate_diagonal(diagonal);
    diagonal *= this->operator_settings.scaling_factor_time_derivative_term;
  }
  else
  {
    diagonal = 0.0;
  }

  if(this->operator_settings.diffusive_problem == true)
  {
    diffusive_operator->add_diagonal(diagonal);
  }

  if(this->operator_settings.convective_problem == true)
  {
    convective_operator->add_diagonal(diagonal, this->eval_time);
  }
}

template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::add_block_jacobi_matrices(BlockMatrix & matrices,
                                                                               Number const time) const
{
  Parent::add_block_jacobi_matrices(matrices, time);
}

template<int dim, int fe_degree, typename Number>
void
ConvectionDiffusionOperator<dim, fe_degree, Number>::add_block_jacobi_matrices(BlockMatrix & matrices) const
{
   
  Number const time = this->get_evaluation_time();
    
  // calculate block Jacobi matrices
  if(this->operator_settings.unsteady_problem == true)
  {
    AssertThrow(this->operator_settings.scaling_factor_time_derivative_term > 0.0,
                ExcMessage("Scaling factor of time derivative term has not been initialized!"));

    mass_matrix_operator->add_block_jacobi_matrices(matrices);

    for(typename std::vector<LAPACKFullMatrix<Number>>::iterator it = matrices.begin(); it != matrices.end();
        ++it)
    {
      (*it) *= this->operator_settings.scaling_factor_time_derivative_term;
    }
  }

  if(this->operator_settings.diffusive_problem == true)
  {
    diffusive_operator->add_block_jacobi_matrices(matrices);
  }

  if(this->operator_settings.convective_problem == true)
  {
    convective_operator->add_block_jacobi_matrices(matrices, time);
  }
}

template<int dim, int fe_degree, typename Number>
MultigridOperatorBase<dim, Number> *
ConvectionDiffusionOperator<dim, fe_degree, Number>::get_new(unsigned int deg) const
{
  switch(deg)
  {
#if DEGREE_1
    case 1:
      return new ConvectionDiffusionOperator<dim, 1, Number>();
#endif
#if DEGREE_2
    case 2:
      return new ConvectionDiffusionOperator<dim, 2, Number>();
#endif
#if DEGREE_3
    case 3:
      return new ConvectionDiffusionOperator<dim, 3, Number>();
#endif
#if DEGREE_4
    case 4:
      return new ConvectionDiffusionOperator<dim, 4, Number>();
#endif
#if DEGREE_5
    case 5:
      return new ConvectionDiffusionOperator<dim, 5, Number>();
#endif
#if DEGREE_6
    case 6:
      return new ConvectionDiffusionOperator<dim, 6, Number>();
#endif
#if DEGREE_7
    case 7:
      return new ConvectionDiffusionOperator<dim, 7, Number>();
#endif
#if DEGREE_8
    case 8:
      return new ConvectionDiffusionOperator<dim, 8, Number>();
#endif
#if DEGREE_9
    case 9:
      return new ConvectionDiffusionOperator<dim, 9, Number>();
#endif
#if DEGREE_10
    case 10:
      return new ConvectionDiffusionOperator<dim, 10, Number>();
#endif
#if DEGREE_11
    case 11:
      return new ConvectionDiffusionOperator<dim, 11, Number>();
#endif
#if DEGREE_12
    case 12:
      return new ConvectionDiffusionOperator<dim, 12, Number>();
#endif
#if DEGREE_13
    case 13:
      return new ConvectionDiffusionOperator<dim, 13, Number>();
#endif
#if DEGREE_14
    case 14:
      return new ConvectionDiffusionOperator<dim, 14, Number>();
#endif
#if DEGREE_15
    case 15:
      return new ConvectionDiffusionOperator<dim, 15, Number>();
#endif
    default:
      AssertThrow(false, ExcMessage("ConvectionDiffusionOperator not implemented for this degree!"));
      return nullptr;
  }
}

} // namespace ConvDiff

#include "convection_diffusion_operator.hpp"
