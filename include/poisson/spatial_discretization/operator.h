/*
 * poisson_operation.h
 *
 *  Created on: 2016
 *      Author: Fehn/Munch
 */

#ifndef INCLUDE_LAPLACE_DG_LAPLACE_OPERATION_H_
#define INCLUDE_LAPLACE_DG_LAPLACE_OPERATION_H_

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/vector_tools.h>

// operators
#include "../../convection_diffusion/spatial_discretization/operators/rhs_operator.h"
#include "../../operators/inverse_mass_matrix.h"
#include "laplace_operator.h"

// solvers/preconditioners
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"

// user interface
#include "../user_interface/analytical_solution.h"
#include "../user_interface/boundary_descriptor.h"
#include "../user_interface/field_functions.h"
#include "../user_interface/input_parameters.h"

// postprocessor
#include "../../convection_diffusion/postprocessor/postprocessor_base.h"

namespace Poisson
{
template<int dim, typename Number>
class DGOperator : public dealii::Subscriptor
{
public:
  typedef float MultigridNumber;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

  DGOperator(parallel::Triangulation<dim> const &                      triangulation,
             Poisson::InputParameters const &                          param_in,
             std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> postprocessor_in);

  void
  setup(PeriodicFaces const                                     periodic_face_pairs,
        std::shared_ptr<Poisson::BoundaryDescriptor<dim>> const boundary_descriptor_in,
        std::shared_ptr<Poisson::FieldFunctions<dim>> const     field_functions_in);

  void
  setup_solver();

  void
  initialize_dof_vector(VectorType & src) const;

  /*
   * Prescribe initial conditions using a specified analytical function.
   */
  void
  prescribe_initial_conditions(VectorType & src) const;

  void
  rhs(VectorType & dst, double const evaluation_time = 0.0) const;

  unsigned int
  solve(VectorType & sol, VectorType const & rhs) const;

  Mapping<dim> const &
  get_mapping() const;

  DoFHandler<dim> const &
  get_dof_handler() const;

  types::global_dof_index
  get_number_of_dofs() const;

  void
  do_postprocessing(VectorType const & solution) const;

private:
  void
  create_dofs();

  void
  initialize_matrix_free();

  void
  setup_operators();

  void
  setup_postprocessor();

  Poisson::InputParameters const & param;

  // DG
  FE_DGQ<dim> fe_dgq;

  // FE (continuous elements)
  FE_Q<dim> fe_q;

  unsigned int                          mapping_degree;
  std::shared_ptr<MappingQGeneric<dim>> mapping;

  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraint_matrix;

  MatrixFree<dim, Number> matrix_free;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;

  ConvDiff::RHSOperator<dim, Number> rhs_operator;

  LaplaceOperator<dim, Number>                laplace_operator;
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> iterative_solver;

  /*
   * Postprocessor.
   */
  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> postprocessor;
};
} // namespace Poisson

#endif
