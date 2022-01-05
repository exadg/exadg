/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_SOLVER_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_SOLVER_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG
#include <exadg/configuration/config.h>
#include <exadg/fluid_structure_interaction/user_interface/declare_get_application.h>
#include <exadg/fluid_structure_interaction_precice/driver.h>
#include <exadg/fluid_structure_interaction_precice/driver_fluid.h>
#include <exadg/fluid_structure_interaction_precice/driver_solid.h>
#include <exadg/utilities/general_parameters.h>

namespace ExaDG
{
struct ResolutionParameters
{
  ResolutionParameters()
  {
  }

  ResolutionParameters(const std::string & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("SpatialResolution");
      prm.add_parameter("DegreeFluid",
                        degree_fluid,
                        "Polynomial degree of fluid (velocity).",
                        Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("DegreeStructure",
                        degree_structure,
                        "Polynomial degree of structural problem.",
                        Patterns::Integer(1,EXADG_DEGREE_MAX),
                        true);
      prm.add_parameter("RefineFluid",
                        refine_fluid,
                        "Number of mesh refinements (fluid).",
                        Patterns::Integer(0,20),
                        true);
      prm.add_parameter("RefineStructure",
                        refine_structure,
                        "Number of mesh refinements (structure).",
                        Patterns::Integer(0,20),
                        true);
    prm.leave_subsection();
    // clang-format on
  }

  unsigned int degree_fluid = 3, degree_structure = 3;

  unsigned int refine_fluid = 0, refine_structure = 0;
};

template<int dim, typename Number>
void
run(std::string const &          input_file,
    ResolutionParameters const & resolution,
    MPI_Comm const &             mpi_comm,
    bool const                   is_test)
{
  Timer timer;
  timer.restart();

  bool const is_solid = false;

  std::shared_ptr<FSI::Driver<dim, Number>> driver;

  if(is_solid)
    driver = std::make_shared<FSI::DriverSolid<dim, Number>>(input_file, mpi_comm, is_test);
  else
    driver = std::make_shared<FSI::DriverFluid<dim, Number>>(input_file, mpi_comm, is_test);

  std::shared_ptr<FSI::ApplicationBase<dim, Number>> application =
    FSI::get_application<dim, Number>(input_file, mpi_comm);

  driver->setup(application,
                resolution.degree_fluid,
                resolution.degree_structure,
                resolution.refine_fluid,
                resolution.refine_structure);

  driver->solve();
}
} // namespace ExaDG

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;

  if(argc == 1)
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      // clang-format off
      std::cout << "To run the program, use:      ./solver input_file" << std::endl
                << "To setup the input file, use: ./solver input_file --help" << std::endl;
      // clang-format on
    }

    return 0;
  }
  else if(argc >= 2)
  {
    input_file = std::string(argv[1]);

    if(argc == 3 and std::string(argv[2]) == "--help")
    {
      Assert(false, dealii::StandardExceptions::ExcNotImplemented());

      return 0;
    }
  }

  ExaDG::GeneralParameters    general(input_file);
  ExaDG::ResolutionParameters resolution(input_file);

  // run the simulation
  if(general.dim == 2 && general.precision == "double")
    ExaDG::run<2, double>(input_file, resolution, mpi_comm, general.is_test);
  else if(general.dim == 3 && general.precision == "double")
    ExaDG::run<3, double>(input_file, resolution, mpi_comm, general.is_test);
  else
    AssertThrow(false, dealii::ExcMessage("Only dim = 2|3 and precision=double implemented."));

  return 0;
}

#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_SOLVER_H_ */
