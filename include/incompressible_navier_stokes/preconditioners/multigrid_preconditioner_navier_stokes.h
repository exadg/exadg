/*
 * MultigridPreconditionerNavierStokes.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_NAVIER_STOKES_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_NAVIER_STOKES_H_


#include "solvers_and_preconditioners/multigrid_preconditioner_dg.h"

namespace IncNS
{

/*
 *  Multigrid preconditioner for velocity (reaction-)diffusion operator
 *  (Helmholtz operator) of the incompressible Navier-Stokes equations.
 */
template<int dim, typename value_type, typename Operator, typename UnderlyingOperator>
class MyMultigridPreconditionerVelocityDiffusion : public MyMultigridPreconditionerDG<dim,value_type,Operator,UnderlyingOperator>
{
public:
  MyMultigridPreconditionerVelocityDiffusion() : 
    MyMultigridPreconditionerDG<dim,value_type,Operator,UnderlyingOperator>(){}

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
      this->mg_matrices[level]->set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
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
  MyMultigridPreconditionerVelocityConvectionDiffusion() : 
    MyMultigridPreconditionerDG<dim,value_type,Operator,UnderlyingOperator>(){}

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
    if (std::is_same<typename Operator::value_type, value_type>::value)
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
        this->mg_matrices[level]->set_solution_linearization(vector_linearization);
      }
      else // all coarser levels
      {
        // restrict vector_linearization from fine to coarse level
        parallel::distributed::Vector<typename Operator::value_type> & vector_fine_level = this->mg_matrices[level+1]->get_solution_linearization();
        parallel::distributed::Vector<typename Operator::value_type> & vector_coarse_level = this->mg_matrices[level]->get_solution_linearization();

        unsigned int dof_index_velocity = this->mg_matrices[level]->get_velocity_conv_diff_operator_data().dof_index;
        DoFHandler<dim> const & dof_handler_velocity = this->mg_matrices[level]->get_data().get_dof_handler(dof_index_velocity);
        unsigned int dofs_per_cell = dof_handler_velocity.get_fe().dofs_per_cell;

        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler_velocity, level+1, relevant_dofs);
        LinearAlgebra::distributed::Vector<typename Operator::value_type> ghosted_vector(dof_handler_velocity.locally_owned_mg_dofs(level+1),
                                                                 relevant_dofs, MPI_COMM_WORLD);
        ghosted_vector = vector_fine_level;
        ghosted_vector.update_ghost_values();

        Vector<typename Operator::value_type> dof_values_fine(dofs_per_cell);
        Vector<typename Operator::value_type> tmp(dofs_per_cell);
        std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
        std::vector<typename Operator::value_type> dof_values_coarse(dofs_per_cell);

        typename DoFHandler<dim>::cell_iterator cell=dof_handler_velocity.begin(level);
        typename DoFHandler<dim>::cell_iterator endc=dof_handler_velocity.end(level);
        for ( ; cell != endc; ++cell)
        {
          if (cell->is_locally_owned_on_level())
          {
            Assert(cell->has_children(), ExcNotImplemented());
            std::fill(dof_values_coarse.begin(), dof_values_coarse.end(), 0.);

            for (unsigned int child=0; child<cell->n_children(); ++child)
            {
              cell->child(child)->get_mg_dof_indices(dof_indices);
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                dof_values_fine(i) = ghosted_vector(dof_indices[i]);

              dof_handler_velocity.get_fe().get_restriction_matrix(child, cell->refinement_case()).vmult (tmp, dof_values_fine);

              for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                if (dof_handler_velocity.get_fe().restriction_is_additive(i)) // discontinuous case
                  dof_values_coarse[i] += tmp[i];
                else if (tmp[i] != 0.) // continuous case
                  dof_values_coarse[i] = tmp[i];
              }
            }
            cell->get_mg_dof_indices(dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              vector_coarse_level(dof_indices[i]) = dof_values_coarse[i];
          }
        }
        vector_coarse_level.compress(VectorOperation::insert); //continuous case
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
      this->mg_matrices[level]->set_evaluation_time(evaluation_time);
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
      this->mg_matrices[level]->set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
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


}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_NAVIER_STOKES_H_ */
