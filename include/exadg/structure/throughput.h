/*
 * throughput.h
 *
 *  Created on: Jan 19, 2021
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_THROUGHPUT_H_
#define INCLUDE_EXADG_STRUCTURE_THROUGHPUT_H_

// likwid
#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

// ExaDG

// driver
#include <exadg/structure/driver.h>

// utilities
#include <exadg/utilities/parameter_study.h>
#include <exadg/utilities/throughput_study.h>

namespace ExaDG
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<Structure::ApplicationBase<dim, Number>>
get_application(std::string input_file);

template<int dim, typename Number>
void
add_parameters_application(dealii::ParameterHandler & prm, std::string const & input_file);

void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  ParameterStudy parameter_study;
  parameter_study.add_parameters(prm);

  ThroughputStudy throughput_study;
  throughput_study.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  typedef double     Number;

  add_parameters_application<Dim, Number>(prm, input_file);

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}

template<int dim, typename Number>
void
run(ThroughputStudy const & throughput,
    std::string const &     input_file,
    unsigned int const      degree,
    unsigned int const      refine_space,
    unsigned int const      n_cells_1d,
    MPI_Comm const &        mpi_comm)
{
  std::shared_ptr<Structure::Driver<dim, Number>> driver;
  driver.reset(new Structure::Driver<dim, Number>(mpi_comm));

  std::shared_ptr<Structure::ApplicationBase<dim, Number>> application =
    get_application<dim, Number>(input_file);

  application->set_subdivisions_hypercube(n_cells_1d);

  driver->setup(application, degree, refine_space, 0 /* refine time */, true);

  std::tuple<unsigned int, types::global_dof_index, double> wall_time =
    driver->apply_operator(degree,
                           throughput.operator_type,
                           throughput.n_repetitions_inner,
                           throughput.n_repetitions_outer);

  throughput.wall_times.push_back(wall_time);
}
} // namespace ExaDG

int
main(int argc, char ** argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;

  if(argc == 1 or (argc == 2 and std::string(argv[1]) == "--help"))
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::cout << "To run the program, use:      ./throughput input_file" << std::endl
                << "To create an input file, use: ./throughput --create_input_file input_file"
                << std::endl;
    }

    return 0;
  }
  else if(argc >= 2)
  {
    input_file = std::string(argv[argc - 1]);
  }

  if(argc == 3 and std::string(argv[1]) == "--create_input_file")
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      ExaDG::create_input_file(input_file);

    return 0;
  }

  ExaDG::ParameterStudy  study(input_file);
  ExaDG::ThroughputStudy throughput(input_file);

  // fill resolution vector depending on the operator_type
  study.fill_resolution_vector(&ExaDG::Structure::get_dofs_per_element, input_file);

  // loop over resolutions vector and run simulations
  for(auto iter = study.resolutions.begin(); iter != study.resolutions.end(); ++iter)
  {
    unsigned int const degree       = std::get<0>(*iter);
    unsigned int const refine_space = std::get<1>(*iter);
    unsigned int const n_cells_1d   = std::get<2>(*iter);

    if(study.dim == 2 && study.precision == "float")
      ExaDG::run<2, float>(throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm);
    else if(study.dim == 2 && study.precision == "double")
      ExaDG::run<2, double>(throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm);
    else if(study.dim == 3 && study.precision == "float")
      ExaDG::run<3, float>(throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm);
    else if(study.dim == 3 && study.precision == "double")
      ExaDG::run<3, double>(throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm);
    else
      AssertThrow(false,
                  dealii::ExcMessage("Only dim = 2|3 and precision = float|double implemented."));
  }

  throughput.print_results(mpi_comm);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}

#endif /* INCLUDE_EXADG_STRUCTURE_THROUGHPUT_H_ */
