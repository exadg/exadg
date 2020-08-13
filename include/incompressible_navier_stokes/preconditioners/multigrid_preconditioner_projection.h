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

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

/*
 * Multigrid preconditioner for projection operator of the incompressible Navier-Stokes equations.
 */
template<int dim, typename Number>
class MultigridPreconditionerProjection : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

private:
  typedef ProjectionOperator<dim, Number>                        PDEOperator;
  typedef ProjectionOperator<dim, MultigridNumber>               PDEOperatorMG;
  typedef MultigridOperatorBase<dim, MultigridNumber>            MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorMG> MGOperator;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

public:
  MultigridPreconditionerProjection(MPI_Comm const & mpi_comm);

  void
  initialize(MultigridData const &                    mg_data,
             parallel::TriangulationBase<dim> const * tria,
             FiniteElement<dim> const &               fe,
             Mapping<dim> const &                     mapping,
             PDEOperator const &                      pde_operator,
             bool const                               mesh_is_moving,
             Map const *                              dirichlet_bc        = nullptr,
             PeriodicFacePairs *                      periodic_face_pairs = nullptr);

  /*
   * This function updates the multigrid preconditioner.
   */
  void
  update() override;

private:
  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level) override;

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level) override;

  /*
   * This function updates the multigrid operators for all levels
   */
  void
  update_operators();

  std::shared_ptr<PDEOperatorMG>
  get_operator(unsigned int level);

  ProjectionOperatorData<dim> data;

  PDEOperator const * pde_operator;

  bool mesh_is_moving;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_PROJECTION_H_ \
        */
