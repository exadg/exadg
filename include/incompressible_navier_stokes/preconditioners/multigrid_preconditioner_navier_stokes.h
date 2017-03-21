/*
 * MultigridPreconditionerNavierStokes.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_NAVIER_STOKES_H_


#include "solvers_and_preconditioners/multigrid_preconditioner_dg.h"


/*
 *  Multigrid preconditioner for velocity (reaction-)diffusion operator
 *  (Helmholtz operator) of the incompressible Navier-Stokes equations.
 */
template<int dim, typename value_type, typename Operator, typename UnderlyingOperator>
class MyMultigridPreconditionerVelocityDiffusion : public MyMultigridPreconditionerDG<dim,value_type,Operator,UnderlyingOperator>
{
public:
  MyMultigridPreconditionerVelocityDiffusion(){}

  virtual ~MyMultigridPreconditionerVelocityDiffusion(){};

  /*
   *  This function updates the multigrid preconditioner.
   */
  virtual void update(MatrixOperatorBase const * matrix_operator)
  {
    UnderlyingOperator const *underlying_operator =
        dynamic_cast<UnderlyingOperator const *>(matrix_operator);

    AssertThrow(underlying_operator != nullptr,
        ExcMessage("Multigrid preconditioner: UnderlyingOperator and MatrixOperator are not compatible!"));

    update_mg_matrices(underlying_operator->get_scaling_factor_time_derivative_term());
    update_smoothers();
    this->update_coarse_solver();
  }

private:
  /*
   *  This function updates mg_matrices by updating
   *  scaling_factor_time_derivative_term.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
   */
  void update_mg_matrices(double const &scaling_factor_time_derivative_term)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      this->mg_matrices[level].set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
    }
  }

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that mg_matrices[level] have
   *  been updated.
   */
  void update_smoothers()
  {
    // Start with level = 1!
    for (unsigned int level = 1; level<this->n_global_levels; ++level)
    {
      this->update_smoother(level);
    }
  }
};

/*
 *  Multigrid preconditioner for velocity (reaction-)convection-diffusion
 *  operator of the incompressible Navier-Stokes equations.
 */
template<int dim, typename value_type, typename Operator, typename UnderlyingOperator>
class MyMultigridPreconditionerVelocityConvectionDiffusion : public MyMultigridPreconditionerDG<dim,value_type,Operator,UnderlyingOperator>
{
public:
  MyMultigridPreconditionerVelocityConvectionDiffusion(){}

  virtual ~MyMultigridPreconditionerVelocityConvectionDiffusion(){};

  /*
   *  This function updates the multigrid preconditioner.
   */
  virtual void update(MatrixOperatorBase const * matrix_operator)
  {
    UnderlyingOperator const *underlying_operator =
        dynamic_cast<UnderlyingOperator const *>(matrix_operator);

    AssertThrow(underlying_operator != nullptr,
        ExcMessage("Multigrid preconditioner: UnderlyingOperator and MatrixOperator are not compatible!"));

    parallel::distributed::Vector<value_type> const & vector_linearization = underlying_operator->get_solution_linearization();

    // convert value_type --> Operator::value_type, e.g., double --> float, but only if necessary
    parallel::distributed::Vector<typename Operator::value_type> vector_multigrid_type_copy;
    parallel::distributed::Vector<typename Operator::value_type> const *vector_multigrid_type_ptr;
    if (types_are_equal<typename Operator::value_type, value_type>::value)
    {
      vector_multigrid_type_ptr = reinterpret_cast<parallel::distributed::Vector<typename Operator::value_type> const*>(&vector_linearization);
    }
    else
    {
      vector_multigrid_type_copy = vector_linearization;
      vector_multigrid_type_ptr = &vector_multigrid_type_copy;
    }

    update_mg_matrices(*vector_multigrid_type_ptr,
                       underlying_operator->get_evaluation_time(),
                       underlying_operator->get_scaling_factor_time_derivative_term());
    update_smoothers();
    this->update_coarse_solver();
  }

private:
  /*
   *  This function updates mg_matrices
   *  To do this, three functions are called:
   *   - set_vector_linearization
   *   - set_evaluation_time
   *   - set_scaling_factor_time_derivative_term
   */
  void update_mg_matrices(parallel::distributed::Vector<typename Operator::value_type> const &vector_linearization,
                          double const                                                       &evaluation_time,
                          double const                                                       &scaling_factor_time_derivative_term)
  {
    set_vector_linearization(vector_linearization);
    set_evaluation_time(evaluation_time);
    set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
  }

  /*
   *  This function updates vector_linearization.
   *  In order to update mg_matrices[level] this function has to be called.
   */
  void set_vector_linearization(parallel::distributed::Vector<typename Operator::value_type> const &vector_linearization)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      if(level == (int)this->n_global_levels-1) // finest level
      {
        this->mg_matrices[level].set_solution_linearization(vector_linearization);
      }
      else // all coarser levels
      {
        // restrict vector_linearization from fine to coarse level
        parallel::distributed::Vector<typename Operator::value_type> & vector_fine_level = this->mg_matrices[level+1].get_solution_linearization();
        parallel::distributed::Vector<typename Operator::value_type> & vector_coarse_level = this->mg_matrices[level].get_solution_linearization();
        // set vector_coarse_level to zero since ..._add is called
        vector_coarse_level = 0.0;
        this->mg_transfer.restrict_and_add(level+1,vector_coarse_level,vector_fine_level);
      }
    }
  }

  /*
   *  This function updates the evaluation time.
   *  In order to update mg_matrices[level] this function has to be called.
   *  (This is due to the fact that the linearized convective term does not
   *  only depend on the linearized velocity field but also on Dirichlet boundary
   *  data which itself depends on the current time.)
   */
  void set_evaluation_time(double const &evaluation_time)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      this->mg_matrices[level].set_evaluation_time(evaluation_time);
    }
  }

  /*
   *  This function updates scaling_factor_time_derivative_term.
   *  In order to update mg_matrices[level] this function has to be called.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
   */
  void set_scaling_factor_time_derivative_term(double const &scaling_factor_time_derivative_term)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      this->mg_matrices[level].set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
    }
  }

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that mg_matrices[level] have
   *  been updated.
   */
  void update_smoothers()
  {
    // Start with level = 1!
    for (unsigned int level = 1; level<this->n_global_levels; ++level)
    {
      this->update_smoother(level);
    }
  }
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_NAVIER_STOKES_H_ */
