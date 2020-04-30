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
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  typedef LaplaceOperator<dim, MultigridNumber, n_components> Laplace;

  typedef MultigridOperatorBase<dim, MultigridNumber>      MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, Laplace> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;

public:
  MultigridPreconditioner(MPI_Comm const & mpi_comm)
    : Base(mpi_comm), is_dg(true), mesh_is_moving(false)
  {
  }

  void
  initialize(MultigridData const &                    mg_data,
             const parallel::TriangulationBase<dim> * tria,
             const FiniteElement<dim> &               fe,
             Mapping<dim> const &                     mapping,
             LaplaceOperatorData<rank, dim> const &   data_in,
             bool const                               mesh_is_moving,
             Map const *                              dirichlet_bc        = nullptr,
             PeriodicFacePairs *                      periodic_face_pairs = nullptr)
  {
    data = data_in;

    is_dg = (fe.dofs_per_vertex == 0);

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
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level)
  {
    matrix_free_data.data.mg_level = this->level_info[level].h_level();

    matrix_free_data.append_mapping_flags(
      Operators::LaplaceKernel<dim, Number>::get_mapping_flags(this->level_info[level].is_dg(),
                                                               this->level_info[level].is_dg()));


    if(data.use_cell_based_loops && this->level_info[level].is_dg())
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &this->dof_handlers[level]->get_triangulation());
      Categorization::do_cell_based_loops(*tria,
                                          matrix_free_data.data,
                                          this->level_info[level].h_level());
    }

    matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "laplace_dof_handler");
    matrix_free_data.insert_constraint(&(*this->constraints[level]), "laplace_dof_handler");
    matrix_free_data.insert_quadrature(QGauss<1>(this->level_info[level].degree() + 1),
                                       "laplace_quadrature");
  }

  /*
   * Has to be overwritten since we want to use ComponentMask here
   */
  void
  initialize_constrained_dofs(DoFHandler<dim> const & dof_handler,
                              MGConstrainedDoFs &     constrained_dofs,
                              Map const &             dirichlet_bc) override
  {
    // TODO: use the same code as for CG case below (which currently segfaults
    // if used for DG case as well)
    if(is_dg)
    {
      std::set<types::boundary_id> dirichlet_boundary;
      for(auto & it : dirichlet_bc)
        dirichlet_boundary.insert(it.first);
      constrained_dofs.initialize(dof_handler);
      constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
    }
    else
    {
      // We use data.bc->dirichlet_bc since we also need dirichlet_bc_component_mask,
      // but the argument dirichlet_bc could be used as well

      constrained_dofs.initialize(dof_handler);
      for(auto it : data.bc->dirichlet_bc)
      {
        std::set<types::boundary_id> dirichlet_boundary;
        dirichlet_boundary.insert(it.first);

        ComponentMask mask    = ComponentMask();
        auto          it_mask = data.bc->dirichlet_bc_component_mask.find(it.first);
        if(it_mask != data.bc->dirichlet_bc_component_mask.end())
          mask = it_mask->second;

        constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary, mask);
      }
    }
  }


  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<Laplace> pde_operator(new Laplace());

    data.dof_index  = this->matrix_free_data_objects[level]->get_dof_index("laplace_dof_handler");
    data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("laplace_quadrature");

    pde_operator->initialize(*this->matrix_free_objects[level], *this->constraints[level], data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

    return mg_operator;
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

  LaplaceOperatorData<rank, dim> data;

  bool is_dg;

  bool mesh_is_moving;
};

} // namespace Poisson


#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_ */
