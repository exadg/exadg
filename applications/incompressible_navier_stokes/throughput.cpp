/*
 * incompressible_navier_stokes_performance.cc
 *
 *  Created on: May 5, 2017
 *      Author: fehn
 */

#include "../include/incompressible_navier_stokes/driver.h"

// infrastructure for parameter studies and throughput measurements
#include "../include/utilities/parameter_study.h"
#include "../include/utilities/throughput_study.h"

// applications
#include "applications/periodic_box/periodic_box.h"

namespace ExaDG
{
class ApplicationSelector
{
public:
  template<int dim, typename Number>
  std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    parse_name_parameter(input_file);

    std::shared_ptr<IncNS::ApplicationBase<dim, Number>> app;
    if(name == "PeriodicBox")
      app.reset(new IncNS::PeriodicBox::Application<dim, Number>(input_file));
    else
      AssertThrow(false, ExcMessage("This application does not exist!"));

    return app;
  }

  template<int dim, typename Number>
  void
  add_parameters(dealii::ParameterHandler & prm, std::string const & input_file)
  {
    // if application is known, also add application-specific parameters
    try
    {
      std::shared_ptr<IncNS::ApplicationBase<dim, Number>> app =
        get_application<dim, Number>(input_file);

      add_name_parameter(prm);
      app->add_parameters(prm);
    }
    catch(...) // if application is unknown, only add name of application to parameters
    {
      add_name_parameter(prm);
    }
  }

private:
  void
  add_name_parameter(ParameterHandler & prm)
  {
    prm.enter_subsection("Application");
    prm.add_parameter("Name", name, "Name of application.");
    prm.leave_subsection();
  }

  void
  parse_name_parameter(std::string input_file)
  {
    dealii::ParameterHandler prm;
    add_name_parameter(prm);
    prm.parse_input(input_file, "", true, true);
  }

  std::string name = "MyApp";
};


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

  ApplicationSelector selector;
  selector.add_parameters<Dim, Number>(prm, input_file);

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
  std::shared_ptr<IncNS::Driver<dim, Number>> driver;
  driver.reset(new IncNS::Driver<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<IncNS::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);
  application->set_subdivisions_hypercube(n_cells_1d);

  unsigned int const refine_time = 0; // not used
  driver->setup(application, degree, refine_space, refine_time, true);


  std::tuple<unsigned int, types::global_dof_index, double> wall_time =
    driver->apply_operator(throughput.operator_type,
                           throughput.n_repetitions_inner,
                           throughput.n_repetitions_outer);

  throughput.wall_times.push_back(wall_time);
}
} // namespace ExaDG

int
main(int argc, char ** argv)
{
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
  study.fill_resolution_vector(&ExaDG::IncNS::get_dofs_per_element, throughput.operator_type);

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
                  dealii::ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
  }

  throughput.print_results(mpi_comm);

  return 0;
}
