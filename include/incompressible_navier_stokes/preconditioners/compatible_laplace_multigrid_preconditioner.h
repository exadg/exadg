/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_

#include <vector>

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "compatible_laplace_operator.h"

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

/*
 *  Multigrid preconditioner for compatible Laplace operator.
 */
template<int dim, typename Number>
class CompatibleLaplaceMultigridPreconditioner : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

private:
  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

  typedef CompatibleLaplaceOperator<dim, MultigridNumber> PDEOperatorMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>            MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorMG> MGOperator;

public:
  CompatibleLaplaceMultigridPreconditioner(MPI_Comm const & mpi_comm);

  void
  initialize(MultigridData const &                      mg_data,
             const parallel::TriangulationBase<dim> *   tria,
             const FiniteElement<dim> &                 fe,
             Mapping<dim> const &                       mapping,
             CompatibleLaplaceOperatorData<dim> const & data_in,
             bool const                                 mesh_is_moving,
             Map const *                                dirichlet_bc        = nullptr,
             PeriodicFacePairs *                        periodic_face_pairs = nullptr);

  void
  update() override;

private:
  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level) override;

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level) override;

  void
  initialize_dof_handler_and_constraints(bool const                 operator_is_singular,
                                         PeriodicFacePairs *        periodic_face_pairs,
                                         FiniteElement<dim> const & fe,
                                         parallel::TriangulationBase<dim> const * tria,
                                         Map const * dirichlet_bc) override;

  std::shared_ptr<PDEOperatorMG>
  get_operator(unsigned int level);

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>     dof_handlers_velocity;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>         constrained_dofs_velocity;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>> constraints_velocity;

  CompatibleLaplaceOperatorData<dim> data;

  bool mesh_is_moving;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_ \
        */
