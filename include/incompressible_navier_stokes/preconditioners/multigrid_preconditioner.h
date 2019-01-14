/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_


#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/momentum_operator.h"

#include "../../solvers_and_preconditioners/util/restrict_vector_to_coarser_level.h"

namespace IncNS
{
/*
 * Multigrid preconditioner for velocity (reaction-)convection-diffusion operator of the
 * incompressible Navier-Stokes equations.
 */
template<int dim, int degree, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
private:
  typedef MomentumOperator<dim, degree, Number>          PDEOperator;
  typedef MomentumOperator<dim, degree, MultigridNumber> MultigridOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;

  typedef typename BASE::VectorType   VectorType;
  typedef typename BASE::VectorTypeMG VectorTypeMG;

public:
  MultigridPreconditioner()
    : MultigridPreconditionerBase<dim, Number, MultigridNumber>(
        std::shared_ptr<PreconditionableOperator<dim, MultigridNumber>>(new MultigridOperator()))
  {
  }

  virtual ~MultigridPreconditioner(){};


  void
  initialize_matrixfree(std::vector<MGLevelIdentifier> &          global_levels,
                        Mapping<dim> const &                      mapping,
                        PreconditionableOperatorData<dim> const & operator_data_in)
  {
    (void)global_levels;
    (void)mapping;
    (void)operator_data_in;

    AssertThrow(false, ExcMessage("Not implemented yet!"));
  }

  /*
   * This function updates the multigrid preconditioner.
   */
  virtual void
  update(LinearOperatorBase const * update_operator)
  {
    PDEOperator const * pde_operator = dynamic_cast<PDEOperator const *>(update_operator);

    AssertThrow(
      pde_operator != nullptr,
      ExcMessage(
        "Operator used to update multigrid preconditioner does not match actual PDE operator!"));

    MultigridOperatorType mg_operator_type = pde_operator->get_operator_data().mg_operator_type;

    if(mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      VectorType const & vector_linearization = pde_operator->get_solution_linearization();

      // convert Number --> Operator::value_type, e.g., double --> float, but only if necessary
      VectorTypeMG         vector_multigrid_type_copy;
      VectorTypeMG const * vector_multigrid_type_ptr;
      if(std::is_same<MultigridNumber, Number>::value)
      {
        vector_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&vector_linearization);
      }
      else
      {
        vector_multigrid_type_copy = vector_linearization;
        vector_multigrid_type_ptr  = &vector_multigrid_type_copy;
      }

      update_mg_matrices(pde_operator->get_evaluation_time(),
                         pde_operator->get_scaling_factor_time_derivative_term(),
                         vector_multigrid_type_ptr);
    }
    else if(mg_operator_type == MultigridOperatorType::ReactionDiffusion)
    {
      update_mg_matrices(pde_operator->get_evaluation_time(),
                         pde_operator->get_scaling_factor_time_derivative_term());
    }
    else
    {
      AssertThrow(false, ExcNotImplemented());
    }

    update_smoothers();
    this->update_coarse_solver();
  }

private:
  /*
   * This function updates mg_matrices
   * To do this, three functions are called:
   *  - set_vector_linearization
   *  - set_evaluation_time
   *  - set_scaling_factor_time_derivative_term
   */
  void
  update_mg_matrices(double const &       evaluation_time,
                     double const &       scaling_factor_time_derivative_term,
                     VectorTypeMG const * vector_linearization = nullptr)
  {
    set_evaluation_time(evaluation_time);
    set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);

    if(vector_linearization != nullptr)
      set_vector_linearization(*vector_linearization);
  }

  /*
   * This function updates vector_linearization.
   * In order to update mg_matrices[level] this function has to be called.
   */
  void
  set_vector_linearization(VectorTypeMG const & vector_linearization)
  {
    for(int level = this->n_global_levels - 1; level >= 0; --level)
    {
      if(level == (int)this->n_global_levels - 1) // finest level
      {
        // this->mg_matrices[level] is a std::shared_ptr<PreconditionableOperator>:
        // so we have to dereference the shared_ptr, get the reference to it and
        // finally we can cast it to pointer of type Operator
        dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
          ->set_solution_linearization(vector_linearization);
      }
      else // all coarser levels
      {
        // restrict vector_linearization from fine to coarse level
        VectorTypeMG const & vector_fine_level =
          dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level + 1])
            ->get_solution_linearization();

        VectorTypeMG vector_coarse_level =
          dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
            ->get_solution_linearization();

        unsigned int dof_index_velocity =
          dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
            ->get_operator_data()
            .dof_index;

        DoFHandler<dim> const & dof_handler_velocity =
          dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
            ->get_data()
            .get_dof_handler(dof_index_velocity);

        restrict_to_coarser_level<dim, MultigridNumber, VectorTypeMG>(vector_coarse_level,
                                                                      vector_fine_level,
                                                                      dof_handler_velocity,
                                                                      level);

        // this->mg_matrices[level] is a std::shared_ptr<PreconditionableOperator>:
        // so we have to dereference the shared_ptr, get the reference to it and
        // finally we can cast it to pointer of type Operator
        dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
          ->set_solution_linearization(vector_coarse_level);
      }
    }
  }

  /*
   * This function updates the evaluation time. In order to update mg_matrices[level] this function
   * has to be called. (This is due to the fact that the linearized convective term does not only
   * depend on the linearized velocity field but also on Dirichlet boundary data which itself
   * depends on the current time.)
   */
  void
  set_evaluation_time(double const & evaluation_time)
  {
    for(int level = this->n_global_levels - 1; level >= 0; --level)
    {
      // this->mg_matrices[level] is a std::shared_ptr<PreconditionableOperator>:
      // so we have to dereference the shared_ptr, get the reference to it and
      // finally we can cast it to pointer of type Operator
      dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
        ->set_evaluation_time(evaluation_time);
    }
  }

  /*
   * This function updates scaling_factor_time_derivative_term. In order to update
   * mg_matrices[level] this function has to be called. This is necessary if adaptive time stepping
   * is used where the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_time_derivative_term(double const & scaling_factor_time_derivative_term)
  {
    for(int level = this->n_global_levels - 1; level >= 0; --level)
    {
      // this->mg_matrices[level] is a std::shared_ptr<PreconditionableOperator>:
      // so we have to dereference the shared_ptr, get the reference to it and
      // finally we can cast it to pointer of type Operator
      dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
        ->set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
    }
  }

  /*
   * This function updates the smoother for all levels of the multigrid algorithm.
   * The prerequisite to call this function is that mg_matrices[level] have been updated.
   */
  void
  update_smoothers()
  {
    // Start with level = 1!
    for(unsigned int level = 1; level < this->n_global_levels; ++level)
    {
      this->update_smoother(level);
    }
  }
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ \
        */
