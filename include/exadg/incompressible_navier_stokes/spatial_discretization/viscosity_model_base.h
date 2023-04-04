/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VISCOSITY_MODEL_BASE_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VISCOSITY_MODEL_BASE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/viscous_operator.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/incompressible_navier_stokes/user_interface/viscosity_model_data.h>
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace IncNS
{
/*
 *  Variable viscosity models.
 */
template<int dim, typename Number>
class ViscosityModelBase : public dealii::Subscriptor
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  /*
   * Constructor.
   */
  ViscosityModelBase();

  /*
   * Destructor.
   */
  virtual ~ViscosityModelBase(){};

  /*
   * Initialization function of base class.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &                matrix_free_in,
             std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel_in,
             unsigned int                                           dof_index_velocity_in,
             unsigned int                                           quad_index_velocity_linear_in);

  /*
   *  Pure virtual function for *setting* the viscosity to viscosity_newtonian_limit.
   */
  virtual void
  set_viscosity(VectorType const & velocity) const = 0;

  /*
   *  Pure virtual function for *adding to* the viscosity taking the currently stored viscosity as a
   * basis.
   */
  virtual void
  add_viscosity(VectorType const & velocity) const = 0;

protected:
  unsigned int dof_index_velocity;
  unsigned int quad_index_velocity_linear;
  double       viscosity_newtonian_limit;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  std::shared_ptr<Operators::ViscousKernel<dim, Number>> viscous_kernel;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_VISCOSITY_MODEL_BASE_H_ \
        */
