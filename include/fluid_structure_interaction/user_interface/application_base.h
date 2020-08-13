/*
 * application_base.h
 *
 *  Created on: 01.04.2020
 *      Author: fehn
 */

#ifndef INCLUDE_FLUID_STRUCTURE_INTERACTION_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_FLUID_STRUCTURE_INTERACTION_USER_INTERFACE_APPLICATION_BASE_H_


// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>


// Fluid

// user interface
#include "../../incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../../incompressible_navier_stokes/user_interface/field_functions.h"
#include "../../incompressible_navier_stokes/user_interface/input_parameters.h"

// postprocessor
#include "../../incompressible_navier_stokes/postprocessor/postprocessor.h"

// Fluid - mesh movement
#include "../../convection_diffusion/user_interface/boundary_descriptor.h"
#include "../../poisson/user_interface/analytical_solution.h"
#include "../../poisson/user_interface/field_functions.h"
#include "../../poisson/user_interface/input_parameters.h"


// Structure

// user interface
#include "../../structure/user_interface/boundary_descriptor.h"
#include "../../structure/user_interface/field_functions.h"
#include "../../structure/user_interface/input_parameters.h"
#include "../../structure/user_interface/material_descriptor.h"

// material
#include "../../structure/material/library/st_venant_kirchhoff.h"

// postprocessor
#include "../../structure/postprocessor/postprocessor.h"

namespace ExaDG
{
namespace FSI
{
using namespace dealii;

template<int dim, typename Number>
class ApplicationBase
{
public:
  virtual void
  add_parameters(ParameterHandler & prm)
  {
    (void)prm;

    // can be overwritten by derived classes and is for example necessary
    // in order to generate a default input file
  }

  ApplicationBase(std::string parameter_file) : parameter_file(parameter_file)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  // fluid
  virtual void
  set_input_parameters_fluid(IncNS::InputParameters & parameters) = 0;

  virtual void
  create_grid_fluid(
    std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
      periodic_faces) = 0;

  // currently required for test cases with analytical mesh movement
  virtual std::shared_ptr<Function<dim>>
  set_mesh_movement_function_fluid()
  {
    std::shared_ptr<Function<dim>> mesh_motion;
    mesh_motion.reset(new Functions::ZeroFunction<dim>(dim));

    return mesh_motion;
  }

  virtual void
  set_boundary_conditions_fluid(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure) = 0;

  virtual void
  set_field_functions_fluid(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions) = 0;

  virtual std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor_fluid(unsigned int const degree, MPI_Comm const & mpi_comm) = 0;

  // Moving mesh

  // Poisson type mesh smoothing
  virtual void
  set_input_parameters_ale(Poisson::InputParameters & parameters) = 0;

  virtual void set_boundary_conditions_ale(
    std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> boundary_descriptor) = 0;

  virtual void
  set_field_functions_ale(std::shared_ptr<Poisson::FieldFunctions<dim>> field_functions) = 0;

  // elasticity type mesh smoothing
  virtual void
  set_input_parameters_ale(Structure::InputParameters & parameters) = 0;

  virtual void
  set_boundary_conditions_ale(
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor) = 0;

  virtual void
  set_material_ale(Structure::MaterialDescriptor & material_descriptor) = 0;

  virtual void
  set_field_functions_ale(std::shared_ptr<Structure::FieldFunctions<dim>> field_functions) = 0;

  // Structure
  virtual void
  set_input_parameters_structure(Structure::InputParameters & parameters) = 0;

  virtual void
  create_grid_structure(
    std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
      periodic_faces) = 0;

  virtual void
  set_boundary_conditions_structure(
    std::shared_ptr<Structure::BoundaryDescriptor<dim>> boundary_descriptor) = 0;

  virtual void
  set_material_structure(Structure::MaterialDescriptor & material_descriptor) = 0;

  virtual void
  set_field_functions_structure(
    std::shared_ptr<Structure::FieldFunctions<dim>> field_functions) = 0;

  virtual std::shared_ptr<Structure::PostProcessor<dim, Number>>
  construct_postprocessor_structure(unsigned int const degree, MPI_Comm const & mpi_comm) = 0;

protected:
  std::string parameter_file;
};

} // namespace FSI
} // namespace ExaDG

#endif /* INCLUDE_FLUID_STRUCTURE_INTERACTION_USER_INTERFACE_APPLICATION_BASE_H_ */
