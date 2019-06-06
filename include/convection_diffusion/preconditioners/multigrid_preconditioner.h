/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/operators/convection_diffusion_operator.h"

namespace ConvDiff
{
/*
 *  Multigrid preconditioner for scalar (reaction-)convection-diffusion operator.
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef ConvectionDiffusionOperator<dim, MultigridNumber> PDEOperator;

  typedef MultigridOperatorBase<dim, MultigridNumber>          MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperator> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;

  typedef typename BASE::Map               Map;
  typedef typename BASE::PeriodicFacePairs PeriodicFacePairs;
  typedef typename BASE::VectorType        VectorType;
  typedef typename BASE::VectorTypeMG      VectorTypeMG;

  virtual ~MultigridPreconditioner(){};

  void
  initialize(MultigridData const &                        mg_data,
             const parallel::Triangulation<dim> *         tria,
             const FiniteElement<dim> &                   fe,
             Mapping<dim> const &                         mapping,
             ConvectionDiffusionOperatorData<dim> const & operator_data_in,
             Map const *                                  dirichlet_bc        = nullptr,
             PeriodicFacePairs *                          periodic_face_pairs = nullptr)
  {
    operator_data            = operator_data_in;
    operator_data.dof_index  = 0;
    operator_data.quad_index = 0;

    // When solving the reaction-convection-diffusion equations, it might be possible
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
      operator_data.diffusive_problem  = true;
    }
    else if(operator_data.mg_operator_type == MultigridOperatorType::ReactionConvection)
    {
      operator_data.convective_problem = true;
      // deactivate viscous term for multigrid preconditioner
      operator_data.diffusive_problem = false;
    }
    else if(operator_data.mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      operator_data.convective_problem = true;
      operator_data.diffusive_problem  = true;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    BASE::initialize(mg_data,
                     tria,
                     fe,
                     mapping,
                     operator_data.operator_is_singular,
                     dirichlet_bc,
                     periodic_face_pairs);
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<PDEOperator> pde_operator(new PDEOperator());
    pde_operator->reinit(*this->matrix_free_objects[level],
                         *this->constraints[level],
                         operator_data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

    return mg_operator;
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping)
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;

    additional_data.level_mg_handler      = this->level_info[level].level;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, MultigridNumber>::AdditionalData::none;

    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    if(this->level_info[level].is_dg)
    {
      additional_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);

      additional_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    }

    if(operator_data.use_cell_based_loops && this->level_info[level].is_dg)
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &this->dof_handlers[level]->get_triangulation());
      Categorization::do_cell_based_loops(*tria, additional_data, this->level_info[level].level);
    }

    if(operator_data.type_velocity_field == TypeVelocityField::Analytical)
    {
      QGauss<1> quadrature(this->level_info[level].degree + 1);
      matrix_free->reinit(mapping,
                          *this->dof_handlers[level],
                          *this->constraints[level],
                          quadrature,
                          additional_data);
    }
    // we need two dof-handlers in case the velocity field comes from the fluid solver.
    else if(operator_data.type_velocity_field == TypeVelocityField::Numerical)
    {
      // collect dof-handlers
      std::vector<const DoFHandler<dim> *> dof_handler_vec;
      dof_handler_vec.resize(2);
      dof_handler_vec[0] = &*this->dof_handlers[level];
      dof_handler_vec[1] = &*this->dof_handlers_velocity[level];

      // collect affine matrices
      std::vector<const AffineConstraints<double> *> constraint_vec;
      constraint_vec.resize(2);
      constraint_vec[0] = &*this->constraints[level];
      constraint_vec[1] = &*this->constraints_velocity[level];


      std::vector<Quadrature<1>> quadrature_vec;
      quadrature_vec.resize(1);
      quadrature_vec[0] = QGauss<1>(this->level_info[level].degree + 1);

      matrix_free->reinit(
        mapping, dof_handler_vec, constraint_vec, quadrature_vec, additional_data);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return matrix_free;
  }

  void
  initialize_dof_handler_and_constraints(bool const                           operator_is_singular,
                                         PeriodicFacePairs *                  periodic_face_pairs,
                                         FiniteElement<dim> const &           fe,
                                         parallel::Triangulation<dim> const * tria,
                                         Map const *                          dirichlet_bc)
  {
    BASE::initialize_dof_handler_and_constraints(
      operator_is_singular, periodic_face_pairs, fe, tria, dirichlet_bc);

    if(operator_data.type_velocity_field == TypeVelocityField::Numerical)
    {
      FESystem<dim> fe_velocity(FE_DGQ<dim>(fe.degree), dim);
      Map           dirichlet_bc_velocity;
      this->do_initialize_dof_handler_and_constraints(false,
                                                      *periodic_face_pairs,
                                                      fe_velocity,
                                                      tria,
                                                      dirichlet_bc_velocity,
                                                      this->level_info,
                                                      this->p_levels,
                                                      this->dof_handlers_velocity,
                                                      this->constrained_dofs_velocity,
                                                      this->constraints_velocity);
    }
  }

  void
  initialize_transfer_operators()
  {
    BASE::initialize_transfer_operators();

    if(operator_data.type_velocity_field == TypeVelocityField::Numerical)
      this->transfers_velocity.template reinit<MultigridNumber>(this->matrix_free_objects,
                                                                this->constraints_velocity,
                                                                this->constrained_dofs_velocity,
                                                                1);
  }

  /*
   *  This function updates the multigrid preconditioner.
   */
  virtual void
  update(LinearOperatorBase const * pde_operator_in)
  {
    ConvectionDiffusionOperator<dim, Number> const * pde_operator =
      dynamic_cast<ConvectionDiffusionOperator<dim, Number> const *>(pde_operator_in);

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

      // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
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

      update_operators(pde_operator->get_evaluation_time(),
                       pde_operator->get_scaling_factor_time_derivative_term(),
                       vector_multigrid_type_ptr);
    }
    else
    {
      update_operators(pde_operator->get_evaluation_time(),
                       pde_operator->get_scaling_factor_time_derivative_term());
    }

    update_smoothers();

    this->update_coarse_solver(operator_data.operator_is_singular);
  }

private:
  /*
   *  This function updates mg_matrices
   *  To do this, three functions are called:
   *   - set_evaluation_time
   *   - set_scaling_factor_time_derivative_term
   */
  void
  update_operators(double const &       evaluation_time,
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
    // copy velocity to finest level
    this->get_operator(this->fine_level)->set_velocity(velocity);

    // interpolate velocity from fine to coarse level
    for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
    {
      auto & vector_fine_level   = this->get_operator(level - 0)->get_velocity();
      auto   vector_coarse_level = this->get_operator(level - 1)->get_velocity();
      transfers_velocity.interpolate(level, vector_coarse_level, vector_fine_level);
      this->get_operator(level - 1)->set_velocity(vector_coarse_level);
    }
  }

  /*
   *  This function updates the evaluation time.
   *  In order to update operators[level] this function has to be called.
   *  (This is due to the fact that the velocity field of the convective term
   *  is a function of the time.)
   */
  void
  set_evaluation_time(double const & evaluation_time)
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
      this->get_operator(level)->set_evaluation_time(evaluation_time);
  }

  /*
   *  This function updates scaling_factor_time_derivative_term.
   *  In order to update operators[level] this function has to be called.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_time_derivative_term(double const & scaling_factor)
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
      this->get_operator(level)->set_scaling_factor_time_derivative_term(scaling_factor);
  }

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that operators[level] have
   *  been updated.
   */
  void
  update_smoothers()
  {
    // Skip coarsest level
    for(unsigned int level = this->coarse_level + 1; level <= this->fine_level; ++level)
      this->update_smoother(level);
  }

  std::shared_ptr<PDEOperator>
  get_operator(unsigned int level)
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  MGTransferMF_MGLevelObject<dim, VectorTypeMG> transfers_velocity;

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>     dof_handlers_velocity;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>         constrained_dofs_velocity;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>> constraints_velocity;

  ConvectionDiffusionOperatorData<dim> operator_data;
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */
