/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/operators/convection_diffusion_operator.h"

#include "../../solvers_and_preconditioners/util/restrict_vector_to_coarser_level.h"

namespace ConvDiff
{
/*
 *  Multigrid preconditioner for scalar (reaction-)convection-diffusion operator.
 */
template<int dim, int degree, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef PreconditionableOperator<dim, MultigridNumber> MG_OPERATOR_BASE;

  typedef ConvectionDiffusionOperator<dim, degree, Number>          PDEOperator;
  typedef ConvectionDiffusionOperator<dim, degree, MultigridNumber> MultigridOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;

  typedef typename BASE::VectorType   VectorType;
  typedef typename BASE::VectorTypeMG VectorTypeMG;

  MultigridPreconditioner()
    : MultigridPreconditionerBase<dim, Number, MultigridNumber>(
        std::shared_ptr<MG_OPERATOR_BASE>(new MultigridOperator()))
  {
  }

  virtual ~MultigridPreconditioner(){};


  void
  initialize_matrixfree(std::vector<MGLevelIdentifier> & global_levels,
                        Mapping<dim> const &             mapping,
                        void *                           operator_data_in)
  {
    auto operator_data = static_cast<ConvectionDiffusionOperatorData<dim> *>(operator_data_in);

    this->mg_matrixfree.resize(0, this->n_global_levels - 1);

    for(unsigned int i = 0; i < this->n_global_levels; i++)
    {
      auto data = new MatrixFree<dim, MultigridNumber>;

      typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;

      additional_data.level_mg_handler = global_levels[i].level;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, MultigridNumber>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);

      if(global_levels[i].is_dg)
      {
        additional_data.mapping_update_flags_inner_faces =
          (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
           update_values);

        additional_data.mapping_update_flags_boundary_faces =
          (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
           update_values);
      }

      if(operator_data->use_cell_based_loops && global_levels[i].is_dg)
      {
        auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
          &this->mg_dofhandler[i]->get_triangulation());
        Categorization::do_cell_based_loops(*tria, additional_data, global_levels[i].level);
      }

      if(operator_data->type_velocity_field == TypeVelocityField::Analytical)
      {
        QGauss<1> quadrature(global_levels[i].degree + 1);
        data->reinit(
          mapping, *this->mg_dofhandler[i], *this->mg_constrains[i], quadrature, additional_data);
      }
      // we need two dof-handlers in case the velocity field comes from the fluid solver.
      else if(operator_data->type_velocity_field == TypeVelocityField::Numerical)
      {
        AssertThrow(false, ExcMessage("No memory allocated yet."));
        // collect dof-handlers
        std::vector<const DoFHandler<dim> *> dof_handler_vec;
        dof_handler_vec.resize(2);
        dof_handler_vec[0] = &*this->mg_dofhandler[i];
        dof_handler_vec[1] = &*this->mg_dofhandler_vel[i];

        // collect affine matrices
        std::vector<const AffineConstraints<double> *> constraint_vec;
        constraint_vec.resize(2);
        constraint_vec[0] = &*this->mg_constrains[i];
        constraint_vec[1] = &*this->mg_constrains_vel[i];


        std::vector<Quadrature<1>> quadrature_vec;
        quadrature_vec.resize(1);
        quadrature_vec[0] = QGauss<1>(global_levels[i].degree + 1);

        data->reinit(mapping, dof_handler_vec, constraint_vec, quadrature_vec, additional_data);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }

      this->mg_matrixfree[i].reset(data);
    }
  }

  /*
   *  This function updates the multigrid preconditioner.
   */
  virtual void
  update(LinearOperatorBase const * matrix_operator)
  {
    PDEOperator const * pde_operator = dynamic_cast<PDEOperator const *>(matrix_operator);

    AssertThrow(
      pde_operator != nullptr,
      ExcMessage(
        "Operator used to update multigrid preconditioner does not match actual PDE operator!"));

    MultigridOperatorType mg_operator_type = pde_operator->get_operator_data().mg_operator_type;
    TypeVelocityField type_velocity_field  = pde_operator->get_operator_data().type_velocity_field;

    if(type_velocity_field == TypeVelocityField::Numerical &&
       (mg_operator_type == MultigridOperatorType::ReactionConvection ||
        mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion))
    {
      VectorType const & velocity = pde_operator->get_velocity();

      // convert Number --> Operator::value_type, e.g., double --> float, but only if necessary
      VectorTypeMG         vector_multigrid_type_copy;
      VectorTypeMG const * vector_multigrid_type_ptr;
      if(std::is_same<MultigridNumber, Number>::value)
      {
        vector_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&velocity);
      }
      else
      {
        vector_multigrid_type_copy = velocity;
        vector_multigrid_type_ptr  = &vector_multigrid_type_copy;
      }

      update_mg_matrices(pde_operator->get_evaluation_time(),
                         pde_operator->get_scaling_factor_time_derivative_term(),
                         vector_multigrid_type_ptr);
    }
    else
    {
      update_mg_matrices(pde_operator->get_evaluation_time(),
                         pde_operator->get_scaling_factor_time_derivative_term());
    }

    update_smoothers();
    this->update_coarse_solver();
  }

private:
  /*
   *  This function updates mg_matrices
   *  To do this, three functions are called:
   *   - set_evaluation_time
   *   - set_scaling_factor_time_derivative_term
   */
  void
  update_mg_matrices(double const &       evaluation_time,
                     double const &       scaling_factor_time_derivative_term,
                     VectorTypeMG const * velocity = nullptr)
  {
    set_evaluation_time(evaluation_time);
    set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);

    if(velocity != nullptr)
      set_velocity(*velocity);
  }

  /*
   * This function updates velocity.
   * In order to update mg_matrices[level] this function has to be called.
   */
  void
  set_velocity(VectorTypeMG const & velocity)
  {
    for(int level = this->n_global_levels - 1; level >= 0; --level)
    {
      if(level == (int)this->n_global_levels - 1) // finest level
      {
        // this->mg_matrices[level] is a std::shared_ptr<PreconditionableOperator>:
        // so we have to dereference the shared_ptr, get the reference to it and
        // finally we can cast it to pointer of type Operator
        dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])->set_velocity(velocity);
      }
      else // all coarser levels
      {
        // restrict velocity from fine to coarse level
        VectorTypeMG const & vector_fine_level =
          dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level + 1])->get_velocity();
        VectorTypeMG vector_coarse_level =
          dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])->get_velocity();

        unsigned int dof_index_velocity =
          dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
            ->get_operator_data()
            .dof_index_velocity;

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
          ->set_velocity(vector_coarse_level);
      }
    }
  }

  /*
   *  This function updates the evaluation time.
   *  In order to update mg_matrices[level] this function has to be called.
   *  (This is due to the fact that the velocity field of the convective term
   *  is a function of the time.)
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
   *  This function updates scaling_factor_time_derivative_term.
   *  In order to update mg_matrices[level] this function has to be called.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
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
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that mg_matrices[level] have
   *  been updated.
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

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>     mg_dofhandler_vel;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>         mg_constrained_dofs_vel;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>> mg_constrains_vel;
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */
