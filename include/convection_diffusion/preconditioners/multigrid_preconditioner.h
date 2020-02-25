/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include "../../operators/mapping_flags.h"
#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/operators/combined_operator.h"

namespace ConvDiff
{
/*
 *  Multigrid preconditioner for scalar convection-diffusion equation
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
private:
  typedef Operator<dim, Number>          PDEOperator;
  typedef Operator<dim, MultigridNumber> PDEOperatorMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>            MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorMG> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

  typedef typename MatrixFree<dim, MultigridNumber>::AdditionalData MatrixFreeData;

public:
  MultigridPreconditioner(MPI_Comm const & mpi_comm)
    : Base(mpi_comm),
      pde_operator(nullptr),
      mg_operator_type(MultigridOperatorType::Undefined),
      mesh_is_moving(false)
  {
  }

  virtual ~MultigridPreconditioner(){};

  void
  initialize(MultigridData const &                    mg_data,
             parallel::TriangulationBase<dim> const * tria,
             FiniteElement<dim> const &               fe,
             Mapping<dim> const &                     mapping,
             PDEOperator const &                      pde_operator,
             MultigridOperatorType const &            mg_operator_type,
             bool const                               mesh_is_moving,
             Map const *                              dirichlet_bc        = nullptr,
             PeriodicFacePairs *                      periodic_face_pairs = nullptr)
  {
    this->pde_operator     = &pde_operator;
    this->mg_operator_type = mg_operator_type;

    this->mesh_is_moving = mesh_is_moving;

    data                                           = this->pde_operator->get_data();
    data.dof_index                                 = 0;
    data.convective_kernel_data.dof_index_velocity = 1;
    data.quad_index                                = 0;

    // When solving the reaction-convection-diffusion equations, it might be possible
    // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
    // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
    // reaction-convection-diffusion operator. Accordingly, we have to reset which
    // operators should be "active" for the multigrid preconditioner, independently of
    // the actual equation type that is solved.
    AssertThrow(this->mg_operator_type != MultigridOperatorType::Undefined,
                ExcMessage("Invalid parameter mg_operator_type."));

    if(this->mg_operator_type == MultigridOperatorType::ReactionDiffusion)
    {
      // deactivate convective term for multigrid preconditioner
      data.convective_problem = false;
      data.diffusive_problem  = true;
    }
    else if(this->mg_operator_type == MultigridOperatorType::ReactionConvection)
    {
      data.convective_problem = true;
      // deactivate viscous term for multigrid preconditioner
      data.diffusive_problem = false;
    }
    else if(this->mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      data.convective_problem = true;
      data.diffusive_problem  = true;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    Base::initialize(
      mg_data, tria, fe, mapping, data.operator_is_singular, dirichlet_bc, periodic_face_pairs);
  }

  /*
   *  This function updates the multigrid preconditioner.
   */
  void
  update() override
  {
    if(mesh_is_moving)
    {
      this->update_matrix_free();
    }

    update_operators();

    update_smoothers();

    this->update_coarse_solver(data.operator_is_singular);
  }

  // TODO multigrid preconditioner is used as filtering operation here
  void
  project_and_prolongate(VectorType & solution_fine) const
  {
    unsigned int smoothing_levels = std::min((unsigned int)1, this->n_levels - 1);

    solution.resize(0, this->n_levels - 1);

    // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
    VectorTypeMG         vector_fine;
    VectorTypeMG const * vector_fine_ptr;
    if(std::is_same<MultigridNumber, Number>::value)
    {
      vector_fine_ptr = reinterpret_cast<VectorTypeMG const *>(&solution_fine);
    }
    else
    {
      vector_fine     = solution_fine;
      vector_fine_ptr = &vector_fine;
    }

    for(unsigned int level = this->fine_level - smoothing_levels; level <= this->fine_level;
        ++level)
    {
      AssertThrow(this->get_operator(level).get() != 0, ExcMessage("invalid pointer"));
      this->get_operator(level)->initialize_dof_vector(solution[level]);
      if(level == this->fine_level)
        solution[level] = *vector_fine_ptr;
    }

    do_project_and_prolongate(this->operators.max_level(), smoothing_levels);

    solution_fine = solution[this->operators.max_level()];
  }

private:
  void
  initialize_matrix_free() override
  {
    if(mesh_is_moving)
    {
      matrix_free_data_update.resize(0, this->n_levels - 1);
    }

    quadrature.resize(0, this->n_levels - 1);

    Base::initialize_matrix_free();
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  do_initialize_matrix_free(unsigned int const level) override
  {
    typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;

    additional_data.mg_level              = this->level_info[level].h_level();
    additional_data.tasks_parallel_scheme = MatrixFree<dim, MultigridNumber>::AdditionalData::none;

    MappingFlags flags;
    if(data.unsteady_problem)
      flags = flags || MassMatrixKernel<dim, Number>::get_mapping_flags();
    if(data.convective_problem)
      flags = flags || Operators::ConvectiveKernel<dim, Number>::get_mapping_flags();
    if(data.diffusive_problem)
      flags = flags || Operators::DiffusiveKernel<dim, Number>::get_mapping_flags();

    additional_data.mapping_update_flags = flags.cells;
    if(this->level_info[level].is_dg())
    {
      additional_data.mapping_update_flags_inner_faces    = flags.inner_faces;
      additional_data.mapping_update_flags_boundary_faces = flags.boundary_faces;
    }

    if(data.use_cell_based_loops && this->level_info[level].is_dg())
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &this->dof_handlers[level]->get_triangulation());
      Categorization::do_cell_based_loops(*tria,
                                          additional_data,
                                          this->level_info[level].h_level());
    }

    if(mesh_is_moving)
    {
      matrix_free_data_update[level] = additional_data;
      matrix_free_data_update[level].initialize_indices =
        false; // connectivity of elements stays the same
      matrix_free_data_update[level].initialize_mapping = true;
    }

    quadrature[level] = QGauss<1>(this->level_info[level].degree() + 1);

    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);
    if(data.convective_kernel_data.velocity_type == TypeVelocityField::Function)
    {
      matrix_free->reinit(*this->mapping,
                          *this->dof_handlers[level],
                          *this->constraints[level],
                          quadrature[level],
                          additional_data);
    }
    // we need two dof-handlers in case the velocity field comes from the fluid solver.
    else if(data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
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
      quadrature_vec[0] = quadrature[level];

      matrix_free->reinit(
        *this->mapping, dof_handler_vec, constraint_vec, quadrature_vec, additional_data);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return matrix_free;
  }

  void
  do_update_matrix_free(unsigned int const level) override
  {
    if(data.convective_kernel_data.velocity_type == TypeVelocityField::Function)
    {
      this->matrix_free_objects[level]->reinit(*this->mapping,
                                               *this->dof_handlers[level],
                                               *this->constraints[level],
                                               quadrature[level],
                                               matrix_free_data_update[level]);
    }
    // we need two dof-handlers in case the velocity field comes from the fluid solver.
    else if(data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
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
      quadrature_vec[0] = quadrature[level];

      this->matrix_free_objects[level]->reinit(*this->mapping,
                                               dof_handler_vec,
                                               constraint_vec,
                                               quadrature_vec,
                                               matrix_free_data_update[level]);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<PDEOperatorMG> pde_operator_level(new PDEOperatorMG());

    pde_operator_level->reinit(*this->matrix_free_objects[level], *this->constraints[level], data);

    // make sure that scaling factor of time derivative term has been set before the smoothers are
    // initialized
    pde_operator_level->set_scaling_factor_mass_matrix(
      pde_operator->get_scaling_factor_mass_matrix());

    // initialize MGOperator which is a wrapper around the PDEOperatorMG
    std::shared_ptr<MGOperator> mg_operator_level(new MGOperator(pde_operator_level));

    return mg_operator_level;
  }

  void
  initialize_dof_handler_and_constraints(bool const                 operator_is_singular,
                                         PeriodicFacePairs *        periodic_face_pairs,
                                         FiniteElement<dim> const & fe,
                                         parallel::TriangulationBase<dim> const * tria,
                                         Map const *                              dirichlet_bc)
  {
    Base::initialize_dof_handler_and_constraints(
      operator_is_singular, periodic_face_pairs, fe, tria, dirichlet_bc);

    if(data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
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
    Base::initialize_transfer_operators();

    if(data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
      this->transfers_velocity.template reinit<MultigridNumber>(this->matrix_free_objects,
                                                                this->constraints_velocity,
                                                                this->constrained_dofs_velocity,
                                                                1);
  }

  void
  do_project_and_prolongate(unsigned int const level, unsigned int const smoothing_levels) const
  {
    // call coarse grid solver
    if(level == this->fine_level - smoothing_levels)
    {
      // do nothing

      return;
    }

    // restriction
    this->transfers.interpolate(level, solution[level - 1], solution[level]);

    // coarse grid correction
    do_project_and_prolongate(level - 1, smoothing_levels);

    // prolongation
    this->transfers.prolongate(level, solution[level], solution[level - 1]);
  }

  /*
   *  This function updates the operators on all levels
   */
  void
  update_operators()
  {
    TypeVelocityField velocity_type = pde_operator->get_data().convective_kernel_data.velocity_type;

    if(velocity_type == TypeVelocityField::DoFVector &&
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

      update_operators(pde_operator->get_time(),
                       pde_operator->get_scaling_factor_mass_matrix(),
                       vector_multigrid_type_ptr);
    }
    else
    {
      update_operators(pde_operator->get_time(), pde_operator->get_scaling_factor_mass_matrix());
    }
  }

  void
  update_operators(double const &       time,
                   double const &       scaling_factor_time_derivative_term,
                   VectorTypeMG const * velocity = nullptr)
  {
    if(mesh_is_moving)
    {
      update_operators_after_mesh_movement();
    }

    set_time(time);
    set_scaling_factor_mass_matrix(scaling_factor_time_derivative_term);

    if(velocity != nullptr)
      set_velocity(*velocity);
  }

  /*
   * This function updates the velocity field for all levels.
   * In order to update mg_matrices[level] this function has to be called.
   */
  void
  set_velocity(VectorTypeMG const & velocity)
  {
    // copy velocity to finest level
    this->get_operator(this->fine_level)->set_velocity_copy(velocity);

    // interpolate velocity from fine to coarse level
    for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
    {
      auto & vector_fine_level   = this->get_operator(level - 0)->get_velocity();
      auto   vector_coarse_level = this->get_operator(level - 1)->get_velocity();
      transfers_velocity.interpolate(level, vector_coarse_level, vector_fine_level);
      this->get_operator(level - 1)->set_velocity_copy(vector_coarse_level);
    }
  }

  /*
   * This function performs the updates that are necessary after the mesh has been moved
   * and after matrix_free has been updated.
   */
  void
  update_operators_after_mesh_movement()
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    {
      this->get_operator(level)->update_after_mesh_movement();
    }
  }

  /*
   *  This function sets the current the time.
   *  In order to update operators[level] this function has to be called.
   *  (This is due to the fact that the velocity field of the convective term
   *  is a function of the time.)
   */
  void
  set_time(double const & time)
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
      this->get_operator(level)->set_time(time);
  }

  /*
   *  This function updates scaling_factor_time_derivative_term.
   *  In order to update operators[level] this function has to be called.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_mass_matrix(double const & scaling_factor)
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
      this->get_operator(level)->set_scaling_factor_mass_matrix(scaling_factor);
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

  std::shared_ptr<PDEOperatorMG>
  get_operator(unsigned int level) const
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  MGTransferMF_MGLevelObject<dim, VectorTypeMG> transfers_velocity;

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>     dof_handlers_velocity;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>         constrained_dofs_velocity;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>> constraints_velocity;

  OperatorData<dim> data;

  PDEOperator const * pde_operator;

  MultigridOperatorType mg_operator_type;

  // TODO multigrid preconditioner is used as filtering operation here
  mutable MGLevelObject<VectorTypeMG> solution;

  // moving mesh
  MGLevelObject<MatrixFreeData> matrix_free_data_update;

  MGLevelObject<Quadrature<1>> quadrature;

  bool mesh_is_moving;
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */
