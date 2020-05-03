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
#include "../../structure/spatial_discretization/operators/nonlinear_operator.h"

namespace Structure
{
template<int dim, typename Number>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

public:
  typedef LinearOperator<dim, Number>          PDEOperatorLinear;
  typedef LinearOperator<dim, MultigridNumber> PDEOperatorLinearMG;

  typedef NonLinearOperator<dim, Number>          PDEOperatorNonlinear;
  typedef NonLinearOperator<dim, MultigridNumber> PDEOperatorNonlinearMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>                     MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorLinearMG>    MGOperatorLinear;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorNonlinearMG> MGOperatorNonlinear;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

  MultigridPreconditioner(MPI_Comm const & mpi_comm);

  void
  initialize(MultigridData const &                       mg_data,
             parallel::TriangulationBase<dim> const *    tria,
             FiniteElement<dim> const &                  fe,
             Mapping<dim> const &                        mapping,
             ElasticityOperatorBase<dim, Number> const & pde_operator,
             bool const                                  nonlinear_operator,
             Map const *                                 dirichlet_bc        = nullptr,
             PeriodicFacePairs *                         periodic_face_pairs = nullptr);

  /*
   * This function updates the multigrid preconditioner.
   */
  void
  update() override;

private:
  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level) override;

  /*
   * Has to be overwritten since we want to use ComponentMask here
   */
  void
  initialize_constrained_dofs(DoFHandler<dim> const & dof_handler,
                              MGConstrainedDoFs &     constrained_dofs,
                              Map const &             dirichlet_bc) override;

  /*
   * This function updates the multigrid operators for all levels
   */
  void
  update_operators();

  /*
   * This function updates solution_linearization.
   * In order to update operators[level] this function has to be called.
   */
  void
  set_solution_linearization(VectorTypeMG const & vector_linearization);

  std::shared_ptr<PDEOperatorNonlinearMG>
  get_operator_nonlinear(unsigned int level);

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level) override;

private:
  OperatorData<dim> data;

  ElasticityOperatorBase<dim, Number> const * pde_operator;

  bool nonlinear;
};

} // namespace Structure

#endif
