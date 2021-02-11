/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include <exadg/convection_diffusion/spatial_discretization/operators/combined_operator.h>
#include <exadg/operators/multigrid_operator.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

/*
 *  Multigrid preconditioner for scalar convection-diffusion equation.
 */
template<int dim, typename Number>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

private:
  typedef Operator<dim, Number>          PDEOperator;
  typedef Operator<dim, MultigridNumber> PDEOperatorMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>            MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorMG> MGOperator;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

public:
  MultigridPreconditioner(MPI_Comm const & mpi_comm);

  virtual ~MultigridPreconditioner(){};

  /*
   *  This function initializes the multigrid preconditioner.
   */
  void
  initialize(MultigridData const &                    mg_data,
             parallel::TriangulationBase<dim> const * tria,
             FiniteElement<dim> const &               fe,
             Mapping<dim> const &                     mapping,
             PDEOperator const &                      pde_operator,
             MultigridOperatorType const &            mg_operator_type,
             bool const                               mesh_is_moving,
             Map const *                              dirichlet_bc        = nullptr,
             PeriodicFacePairs *                      periodic_face_pairs = nullptr);

  /*
   *  This function updates the multigrid preconditioner.
   */
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

  void
  initialize_transfer_operators() override;

  /*
   *  This function updates the operators on all levels
   */
  void
  update_operators();

  /*
   * This function updates the velocity field for all levels.
   * In order to update mg_matrices[level] this function has to be called.
   */
  void
  set_velocity(VectorTypeMG const & velocity);

  /*
   * This function performs the updates that are necessary after the mesh has been moved
   * and after matrix_free has been updated.
   */
  void
  update_operators_after_mesh_movement();

  /*
   *  This function sets the current the time.
   *  In order to update operators[level] this function has to be called.
   *  (This is due to the fact that the velocity field of the convective term
   *  is a function of the time.)
   */
  void
  set_time(double const & time);

  /*
   *  This function updates the scaling factor of the mass operator.
   *  In order to update operators[level] this function has to be called.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_mass_operator(double const & scaling_factor);

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that operators[level] have
   *  been updated.
   */
  void
  update_smoothers();

  std::shared_ptr<PDEOperatorMG>
  get_operator(unsigned int level) const;

  MGTransfer_MGLevelObject<dim, MultigridNumber, VectorTypeMG> transfers_velocity;

  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>              dof_handlers_velocity;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>                  constrained_dofs_velocity;
  MGLevelObject<std::shared_ptr<AffineConstraints<MultigridNumber>>> constraints_velocity;

  OperatorData<dim> data;

  PDEOperator const * pde_operator;

  MultigridOperatorType mg_operator_type;

  bool mesh_is_moving;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */
