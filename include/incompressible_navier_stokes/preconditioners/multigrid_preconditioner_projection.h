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

  typedef typename MatrixFree<dim, MultigridNumber>::AdditionalData MatrixFreeData;

public:
  MultigridPreconditionerProjection() : pde_operator(nullptr), mesh_is_moving(false)
  {
  }

  void
  initialize(MultigridData const &                    mg_data,
             parallel::TriangulationBase<dim> const * tria,
             FiniteElement<dim> const &               fe,
             Mapping<dim> const &                     mapping,
             PDEOperatorNumber const &                pde_operator,
             bool const                               mesh_is_moving,
             Map const *                              dirichlet_bc        = nullptr,
             PeriodicFacePairs *                      periodic_face_pairs = nullptr)
  {
    this->pde_operator = &pde_operator;

    data            = this->pde_operator->get_data();
    data.dof_index  = 0;
    data.quad_index = 0;

    this->mesh_is_moving = mesh_is_moving;

    Base::initialize(mg_data,
                     tria,
                     fe,
                     mapping,
                     false /*operator_is_singular*/,
                     dirichlet_bc,
                     periodic_face_pairs);
  }

  /*
   * This function updates the multigrid preconditioner.
   */
  void
  update() override
  {
    if(mesh_is_moving)
    {
      this->update_matrix_free();
    }

    update_operators();

    this->update_smoothers();

    // singular operators do not occur for this operator
    this->update_coarse_solver(false /* operator_is_singular */);
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
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    MatrixFreeData additional_data;

    additional_data.mg_level              = this->level_info[level].h_level();
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

    if(mesh_is_moving)
    {
      matrix_free_data_update[level] = additional_data;
      matrix_free_data_update[level].initialize_indices =
        false; // connectivity of elements stays the same
      matrix_free_data_update[level].initialize_mapping = true;
    }

    quadrature[level] = QGauss<1>(this->level_info[level].degree() + 1);
    matrix_free->reinit(*this->mapping,
                        *this->dof_handlers[level],
                        *this->constraints[level],
                        quadrature[level],
                        additional_data);

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

  void
  do_update_matrix_free(unsigned int const level) override
  {
    this->matrix_free_objects[level]->reinit(*this->mapping,
                                             *this->dof_handlers[level],
                                             *this->constraints[level],
                                             quadrature[level],
                                             matrix_free_data_update[level]);
  }

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

  std::shared_ptr<PDEOperator>
  get_operator(unsigned int level)
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  ProjectionOperatorData<dim> data;

  PDEOperatorNumber const * pde_operator;

  MGLevelObject<MatrixFreeData> matrix_free_data_update;

  MGLevelObject<Quadrature<1>> quadrature;

  bool mesh_is_moving;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_PROJECTION_H_ \
        */
