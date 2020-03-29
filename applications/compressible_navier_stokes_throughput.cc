/*
 * compressible_navier_stokes_performance.cc
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// driver
#include "../include/compressible_navier_stokes/driver.h"

// infrastructure for parameter studies and throughput measurements
#include "../include/functionalities/parameter_study.h"
#include "../include/functionalities/throughput_study.h"

// specify the test case that has to be solved
#include "compressible_navier_stokes_test_cases/taylor_green/taylor_green.h"

class ApplicationSelector
{
public:
  template<int dim, typename Number>
  void
  add_parameters(dealii::ParameterHandler & prm, std::string name_of_application = "")
  {
    // application is unknown -> only add name of application to parameters
    if(name_of_application.length() == 0)
    {
      this->add_name_parameter(prm);
    }
    else // application is known -> add also application-specific parameters
    {
      name = name_of_application;
      this->add_name_parameter(prm);

      std::shared_ptr<CompNS::ApplicationBase<dim, Number>> app;
      if(name == "TaylorGreen")
        app.reset(new CompNS::TaylorGreen::Application<dim, Number>());
      else
        AssertThrow(false, ExcMessage("This application does not exist!"));

      app->add_parameters(prm);
    }
  }

  template<int dim, typename Number>
  std::shared_ptr<CompNS::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    dealii::ParameterHandler prm;
    this->add_name_parameter(prm);
    parse_input(input_file, prm, true, true);

    std::shared_ptr<CompNS::ApplicationBase<dim, Number>> app;
    if(name == "TaylorGreen")
      app.reset(new CompNS::TaylorGreen::Application<dim, Number>(input_file));
    else
      AssertThrow(false, ExcMessage("This application does not exist!"));

    return app;
  }

private:
  void
  add_name_parameter(ParameterHandler & prm)
  {
    prm.enter_subsection("Application");
    prm.add_parameter("Name", name, "Name of application.");
    prm.leave_subsection();
  }

  std::string name = "MyApp";
};

void
create_input_file(std::string const & name_of_application = "")
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
  selector.add_parameters<Dim, Number>(prm, name_of_application);

  prm.print_parameters(std::cout, dealii::ParameterHandler::OutputStyle::JSON, false);
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
  std::shared_ptr<CompNS::Driver<dim, Number>> driver;
  driver.reset(new CompNS::Driver<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<CompNS::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);
  application->set_subdivisions_hypercube(n_cells_1d);

  unsigned int const refine_time = 0; // not used
  driver->setup(application, degree, refine_space, refine_time);


  std::tuple<unsigned int, types::global_dof_index, double> wall_time =
    driver->apply_operator(throughput.operator_type,
                           throughput.n_repetitions_inner,
                           throughput.n_repetitions_outer);

  throughput.wall_times.push_back(wall_time);
}

int
main(int argc, char ** argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    MPI_Comm mpi_comm(MPI_COMM_WORLD);

    // check if parameter file is provided

    // ./compressible_navier_stokes_throughput
    AssertThrow(argc > 1, ExcMessage("No parameter file has been provided!"));

    // ./compressible_navier_stokes_throughput --help
    if(argc == 2 && std::string(argv[1]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        create_input_file();

      return 0;
    }
    // ./compressible_navier_stokes_throughput --help NameOfApplication
    else if(argc == 3 && std::string(argv[1]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        create_input_file(argv[2]);

      return 0;
    }

    // the second argument is the input-file
    // ./compressible_navier_stokes_throughput InputFile
    std::string     input_file = std::string(argv[1]);
    ParameterStudy  study(input_file);
    ThroughputStudy throughput(input_file);

    // fill resolution vector depending on the operator_type
    study.fill_resolution_vector(&CompNS::get_dofs_per_element, throughput.operator_type);

    // loop over resolutions vector and run simulations
    for(auto iter = study.resolutions.begin(); iter != study.resolutions.end(); ++iter)
    {
      unsigned int const degree       = std::get<0>(*iter);
      unsigned int const refine_space = std::get<1>(*iter);
      unsigned int const n_cells_1d   = std::get<2>(*iter);

      if(study.dim == 2 && study.precision == "float")
        run<2, float>(throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm);
      else if(study.dim == 2 && study.precision == "double")
        run<2, double>(throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm);
      else if(study.dim == 3 && study.precision == "float")
        run<3, float>(throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm);
      else if(study.dim == 3 && study.precision == "double")
        run<3, double>(throughput, input_file, degree, refine_space, n_cells_1d, mpi_comm);
      else
        AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
    }

    throughput.print_results(mpi_comm);
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }


#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
