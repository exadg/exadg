/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/laplace_operator.h"

namespace Poisson
{
/*
 *  Multigrid preconditioner for scalar Laplace operator.
 */
template<int dim, typename Number, typename MultigridNumber, int n_components>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef LaplaceOperator<dim, MultigridNumber, n_components> Laplace;

  typedef MultigridOperatorBase<dim, MultigridNumber>      MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, Laplace> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;

  typedef typename MatrixFree<dim, MultigridNumber>::AdditionalData MatrixFreeData;

  MultigridPreconditioner(MPI_Comm const & mpi_comm) : Base(mpi_comm), mesh_is_moving(false)
  {
  }

  void
  initialize(MultigridData const &                    mg_data,
             const parallel::TriangulationBase<dim> * tria,
             const FiniteElement<dim> &               fe,
             Mapping<dim> const &                     mapping,
             LaplaceOperatorData<dim> const &         data_in,
             bool const                               mesh_is_moving,
             Map const *                              dirichlet_bc        = nullptr,
             PeriodicFacePairs *                      periodic_face_pairs = nullptr)
  {
    data            = data_in;
    data.dof_index  = 0;
    data.quad_index = 0;

    this->mesh_is_moving = mesh_is_moving;

    Base::initialize(
      mg_data, tria, fe, mapping, data.operator_is_singular, dirichlet_bc, periodic_face_pairs);
  }

  void
  update() override
  {
    // update of this multigrid preconditioner is only needed
    // if the mesh is moving
    if(mesh_is_moving)
    {
      this->update_matrix_free();

      update_operators_after_mesh_movement();

      this->update_smoothers();

      // singular operators do not occur for this operator
      this->update_coarse_solver(data.operator_is_singular);
    }
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
    additional_data.mg_level = this->level_info[level].h_level();

    MappingFlags flags = Operators::LaplaceKernel<dim, Number>::get_mapping_flags();
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
    std::shared_ptr<Laplace> pde_operator(new Laplace());

    pde_operator->reinit(*this->matrix_free_objects[level], *this->constraints[level], data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

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
   * This function performs the updates that are necessary after the mesh has been moved
   * and after matrix_free has been updated.
   */
  void
  update_operators_after_mesh_movement()
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    {
      get_operator(level)->update_after_mesh_movement();
    }
  }

  std::shared_ptr<Laplace>
  get_operator(unsigned int level)
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  LaplaceOperatorData<dim> data;

  MGLevelObject<MatrixFreeData> matrix_free_data_update;

  MGLevelObject<Quadrature<1>> quadrature;

  bool mesh_is_moving;
};

} // namespace Poisson


#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_ */
