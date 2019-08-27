/*
 * multigrid_preconditioner_projection.h
 *
 *  Created on: Jun 25, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_PROJECTION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_PROJECTION_H_

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/operators/projection_operator.h"

namespace IncNS
{
/*
 * Multigrid preconditioner for projection operator of the incompressible Navier-Stokes equations.
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditionerProjection
  : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
private:
  typedef ProjectionOperator<dim, Number>                      PDEOperatorNumber;
  typedef ProjectionOperator<dim, MultigridNumber>             PDEOperator;
  typedef MultigridOperatorBase<dim, MultigridNumber>          MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperator> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

public:
  MultigridPreconditionerProjection() : pde_operator(nullptr)
  {
  }

  virtual ~MultigridPreconditionerProjection(){};

  void
  initialize(MultigridData const &                mg_data,
             parallel::TriangulationBase<dim> const * tria,
             FiniteElement<dim> const &           fe,
             Mapping<dim> const &                 mapping,
             PDEOperatorNumber const &            pde_operator,
             Map const *                          dirichlet_bc        = nullptr,
             PeriodicFacePairs *                  periodic_face_pairs = nullptr)
  {
    this->pde_operator = &pde_operator;

    data            = this->pde_operator->get_data();
    data.dof_index  = 0;
    data.quad_index = 0;

    Base::initialize(mg_data,
                     tria,
                     fe,
                     mapping,
                     false /*operator_is_singular*/,
                     dirichlet_bc,
                     periodic_face_pairs);
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping)
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    // additional data
    typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;

    additional_data.level_mg_handler      = this->level_info[level].h_level();
    additional_data.tasks_parallel_scheme = MatrixFree<dim, MultigridNumber>::AdditionalData::none;

    MappingFlags flags;
    flags = flags || MassMatrixKernel<dim, Number>::get_mapping_flags();
    if(data.use_divergence_penalty)
      flags = flags || Operators::DivergencePenaltyKernel<dim, Number>::get_mapping_flags();
    if(data.use_continuity_penalty)
      flags = flags || Operators::ContinuityPenaltyKernel<dim, Number>::get_mapping_flags();

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

    QGauss<1> quadrature(this->level_info[level].degree() + 1);
    matrix_free->reinit(
      mapping, *this->dof_handlers[level], *this->constraints[level], quadrature, additional_data);

    return matrix_free;
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<PDEOperator> pde_operator_level(new PDEOperator());

    // The polynomial degree changes in case of p-multigrid, so we have to adapt the kernel_data
    // objects.
    Operators::DivergencePenaltyKernelData div_kernel_data =
      this->pde_operator->get_divergence_kernel_data();
    div_kernel_data.degree = this->level_info[level].degree();

    Operators::ContinuityPenaltyKernelData conti_kernel_data =
      this->pde_operator->get_continuity_kernel_data();
    conti_kernel_data.degree = this->level_info[level].degree();

    pde_operator_level->reinit(*this->matrix_free_objects[level],
                               *this->constraints[level],
                               data,
                               div_kernel_data,
                               conti_kernel_data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator_level));

    return mg_operator;
  }

  /*
   * This function updates the multigrid preconditioner.
   */
  virtual void
  update()
  {
    update_operators();

    update_smoothers();

    // singular operators do not occur for this operator
    this->update_coarse_solver(false /* operator_is_singular */);
  }

private:
  /*
   * This function updates the multigrid operators for all levels
   */
  void
  update_operators()
  {
    double const time_step_size = pde_operator->get_time_step_size();

    VectorType const & velocity = pde_operator->get_velocity();

    // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
    VectorTypeMG         velocity_multigrid_type_copy;
    VectorTypeMG const * velocity_multigrid_type_ptr;
    if(std::is_same<MultigridNumber, Number>::value)
    {
      velocity_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&velocity);
    }
    else
    {
      velocity_multigrid_type_copy = velocity;
      velocity_multigrid_type_ptr  = &velocity_multigrid_type_copy;
    }

    // update operator
    this->get_operator(this->fine_level)->update(*velocity_multigrid_type_ptr, time_step_size);

    // we store only two vectors since the velocity is no longer needed after having updated the
    // operators
    VectorTypeMG velocity_fine_level = *velocity_multigrid_type_ptr;
    VectorTypeMG velocity_coarse_level;

    for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
    {
      // interpolate velocity from fine to coarse level
      this->get_operator(level - 1)->initialize_dof_vector(velocity_coarse_level);
      this->transfers.interpolate(level, velocity_coarse_level, velocity_fine_level);

      // update operator
      this->get_operator(level - 1)->update(velocity_coarse_level, time_step_size);

      // current coarse level becomes the fine level in the next iteration
      this->get_operator(level - 1)->initialize_dof_vector(velocity_fine_level);
      velocity_fine_level.copy_locally_owned_data_from(velocity_coarse_level);
    }
  }

  /*
   * This function updates the smoother for all multigrid levels.
   * The prerequisite to call this function is that the multigrid operators have been updated.
   */
  void
  update_smoothers()
  {
    // Skip coarsest level
    for(unsigned int level = this->coarse_level + 1; level <= this->fine_level; ++level)
    {
      this->update_smoother(level);
    }
  }

  std::shared_ptr<PDEOperator>
  get_operator(unsigned int level)
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  ProjectionOperatorData data;

  PDEOperatorNumber const * pde_operator;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_PROJECTION_H_ \
        */
