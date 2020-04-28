/*
 * multigrid_preconditioner.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_STRUCTURE_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"

#include "../../structure/spatial_discretization/operators/linear_operator.h"

namespace Structure
{
template<int dim, typename Number, typename MultigridNumber, bool nonlinear>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef LinearOperator<dim, Number>          PDEOperatorLinear;
  typedef LinearOperator<dim, MultigridNumber> PDEOperatorLinearMG;

  typedef NonLinearOperator<dim, Number>          PDEOperatorNonlinear;
  typedef NonLinearOperator<dim, MultigridNumber> PDEOperatorNonlinearMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>                     MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorLinearMG>    MGOperatorLinear;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorNonlinearMG> MGOperatorNonlinear;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

  MultigridPreconditioner(MPI_Comm const & mpi_comm) : Base(mpi_comm), pde_operator(nullptr)
  {
  }

  void
  initialize(MultigridData const &                       mg_data,
             parallel::TriangulationBase<dim> const *    tria,
             FiniteElement<dim> const &                  fe,
             Mapping<dim> const &                        mapping,
             ElasticityOperatorBase<dim, Number> const & pde_operator,
             Map const *                                 dirichlet_bc        = nullptr,
             PeriodicFacePairs *                         periodic_face_pairs = nullptr)
  {
    this->pde_operator = &pde_operator;

    data = this->pde_operator->get_data();

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
    if(nonlinear)
    {
      update_operators();

      this->update_smoothers();

      // singular operators do not occur for this operator
      this->update_coarse_solver(false /* operator_is_singular */);
    }
    else
    {
      AssertThrow(
        false,
        ExcMessage("Update of multigrid preconditioner is not implemented for linear elasticity."));
    }
  }

private:
  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level)
  {
    matrix_free_data.data.mg_level = this->level_info[level].h_level();
    matrix_free_data.data.tasks_parallel_scheme =
      MatrixFree<dim, MultigridNumber>::AdditionalData::none;

    if(nonlinear)
      matrix_free_data.append_mapping_flags(PDEOperatorNonlinear::get_mapping_flags());
    else // linear
      matrix_free_data.append_mapping_flags(PDEOperatorLinear::get_mapping_flags());

    matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "elasticity_dof_handler");
    matrix_free_data.insert_constraint(&(*this->constraints[level]), "elasticity_dof_handler");
    matrix_free_data.insert_quadrature(QGauss<1>(this->level_info[level].degree() + 1),
                                       "elasticity_quadrature");
  }

  /*
   * Has to be overwritten since we want to use ComponentMask here
   */
  void
  initialize_constrained_dofs(DoFHandler<dim> const & dof_handler,
                              MGConstrainedDoFs &     constrained_dofs,
                              Map const &             dirichlet_bc) override
  {
    // We use data.bc->dirichlet_bc since we also need dirichlet_bc_component_mask,
    // but the argument dirichlet_bc could be used as well
    (void)dirichlet_bc;

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

  /*
   * This function updates the multigrid operators for all levels
   */
  void
  update_operators()
  {
    PDEOperatorNonlinear const * pde_operator_nonlinear =
      dynamic_cast<PDEOperatorNonlinear const *>(pde_operator);

    VectorType const & vector_linearization = pde_operator_nonlinear->get_solution_linearization();

    // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
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

    set_solution_linearization(*vector_multigrid_type_ptr);
  }

  /*
   * This function updates solution_linearization.
   * In order to update operators[level] this function has to be called.
   */
  void
  set_solution_linearization(VectorTypeMG const & vector_linearization)
  {
    // copy velocity to finest level
    this->get_operator_nonlinear(this->fine_level)
      ->set_solution_linearization(vector_linearization);

    // interpolate velocity from fine to coarse level
    for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
    {
      auto & vector_fine_level =
        this->get_operator_nonlinear(level - 0)->get_solution_linearization();
      auto vector_coarse_level =
        this->get_operator_nonlinear(level - 1)->get_solution_linearization();
      this->transfers.interpolate(level, vector_coarse_level, vector_fine_level);
      this->get_operator_nonlinear(level - 1)->set_solution_linearization(vector_coarse_level);
    }
  }

  std::shared_ptr<PDEOperatorNonlinearMG>
  get_operator_nonlinear(unsigned int level)
  {
    std::shared_ptr<MGOperatorNonlinear> mg_operator =
      std::dynamic_pointer_cast<MGOperatorNonlinear>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    std::shared_ptr<MGOperatorBase> mg_operator_level;

    data.dof_index = this->matrix_free_data_objects[level]->get_dof_index("elasticity_dof_handler");
    data.quad_index =
      this->matrix_free_data_objects[level]->get_quad_index("elasticity_quadrature");

    if(nonlinear)
    {
      std::shared_ptr<PDEOperatorNonlinearMG> pde_operator_level(new PDEOperatorNonlinearMG());
      pde_operator_level->reinit(*this->matrix_free_objects[level],
                                 *this->constraints[level],
                                 data);

      mg_operator_level.reset(new MGOperatorNonlinear(pde_operator_level));
    }
    else // linear
    {
      std::shared_ptr<PDEOperatorLinearMG> pde_operator_level(new PDEOperatorLinearMG());
      pde_operator_level->reinit(*this->matrix_free_objects[level],
                                 *this->constraints[level],
                                 data);

      mg_operator_level.reset(new MGOperatorLinear(pde_operator_level));
    }

    return mg_operator_level;
  }

private:
  OperatorData<dim> data;

  ElasticityOperatorBase<dim, Number> const * pde_operator;
};

} // namespace Structure

#endif
