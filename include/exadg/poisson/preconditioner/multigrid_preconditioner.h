/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_

#include <exadg/operators/multigrid_operator.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

/*
 *  Multigrid preconditioner for Laplace operator.
 */
template<int dim, typename Number, int n_components>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;

  typedef LaplaceOperator<dim, MultigridNumber, n_components> Laplace;

  typedef MultigridOperatorBase<dim, MultigridNumber>      MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, Laplace> MGOperator;

public:
  MultigridPreconditioner(MPI_Comm const & mpi_comm);

  void
  initialize(MultigridData const &                    mg_data,
             const parallel::TriangulationBase<dim> * tria,
             const FiniteElement<dim> &               fe,
             Mapping<dim> const &                     mapping,
             LaplaceOperatorData<rank, dim> const &   data_in,
             bool const                               mesh_is_moving,
             Map const *                              dirichlet_bc        = nullptr,
             PeriodicFacePairs *                      periodic_face_pairs = nullptr);

  void
  update() override;

private:
  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level,
                        unsigned int const                     h_level);

  /*
   * Has to be overwritten since we want to use ComponentMask here
   */
  void
  initialize_constrained_dofs(DoFHandler<dim> const & dof_handler,
                              MGConstrainedDoFs &     constrained_dofs,
                              Map const &             dirichlet_bc) override;


  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level) override;

  /*
   * This function performs the updates that are necessary after the mesh has been moved
   * and after matrix_free has been updated.
   */
  void
  update_operators_after_mesh_movement();

  std::shared_ptr<Laplace>
  get_operator(unsigned int level);

  LaplaceOperatorData<rank, dim> data;

  bool is_dg;

  bool mesh_is_moving;
};

} // namespace Poisson
} // namespace ExaDG

#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_ */
