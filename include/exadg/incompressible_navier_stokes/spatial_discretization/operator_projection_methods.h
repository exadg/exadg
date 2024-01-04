/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PROJECTION_METHODS_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PROJECTION_METHODS_H_

#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Base class for projection-type incompressible Navier-Stokes solvers such as the high-order dual
 * splitting (velocity-correction) scheme or pressure correction schemes.
 */
template<int dim, typename Number>
class OperatorProjectionMethods : public SpatialOperatorBase<dim, Number>
{
protected:
  typedef SpatialOperatorBase<dim, Number> Base;

  typedef typename Base::VectorType       VectorType;
  typedef typename Base::MultigridPoisson MultigridPoisson;

public:
  /*
   * Constructor.
   */
  OperatorProjectionMethods(
    std::shared_ptr<Grid<dim> const>                      grid,
    std::shared_ptr<dealii::Mapping<dim> const>           mapping,
    std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings,
    std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor,
    std::shared_ptr<FieldFunctions<dim> const>            field_functions,
    Parameters const &                                    parameters,
    std::string const &                                   field,
    MPI_Comm const &                                      mpi_comm);

  /*
   * Destructor.
   */
  virtual ~OperatorProjectionMethods();

protected:
  /*
   * Calls setup() function of base class and additionally initializes the pressure Poisson operator
   * needed for projection-type methods.
   */
  void
  setup_derived() override;

  void
  setup_preconditioners_and_solvers() override;

public:
  void
  update_after_grid_motion(bool const update_matrix_free) override;

  /*
   * This function evaluates the rhs-contribution of the viscous term and adds the result to the
   * dst-vector.
   */
  void
  do_rhs_add_viscous_term(VectorType & dst, double const time) const;

  /*
   * Pressure Poisson equation: This function evaluates the inhomogeneous parts of boundary face
   * integrals of the negative Laplace operator and adds the result to the dst-vector.
   */
  void
  do_rhs_ppe_laplace_add(VectorType & dst, double const & time) const;

  /*
   * This function solves the pressure Poisson equation and returns the number of iterations.
   */
  unsigned int
  do_solve_pressure(VectorType &       dst,
                    VectorType const & src,
                    bool const         update_preconditioner) const;

  /*
   * This function applies the projection operator (used for throughput measurements).
   */
  void
  apply_projection_operator(VectorType & dst, VectorType const & src) const;

  /*
   * This function applies the Laplace operator (used for throughput measurements).
   */
  void
  apply_laplace_operator(VectorType & dst, VectorType const & src) const;

protected:
  // Pressure Poisson equation (operator, preconditioner, solver).
  Poisson::LaplaceOperator<dim, Number, 1> laplace_operator;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner_pressure_poisson;

  std::shared_ptr<Krylov::SolverBase<VectorType>> pressure_poisson_solver;

private:
  void
  initialize_laplace_operator();

  /*
   * Setup functions called during setup of pressure Poisson solver.
   */
  void
  setup_preconditioner_pressure_poisson();

  void
  setup_solver_pressure_poisson();
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_PROJECTION_METHODS_H_ \
        */
