/*
 * incompressible_navier_stokes.cc
 *
 *  Created on: Oct 10, 2016
 *      Author: fehn
 */

// driver
#include "../include/incompressible_navier_stokes/driver.h"

// infrastructure for convergence studies
#include "../include/utilities/convergence_study.h"

// applications
#include "incompressible_navier_stokes_test_cases/template/template.h"

// Stokes flow - analytical solutions
#include "incompressible_navier_stokes_test_cases/stokes_curl_flow/stokes_curl_flow.h"
#include "incompressible_navier_stokes_test_cases/stokes_guermond/stokes_guermond.h"
#include "incompressible_navier_stokes_test_cases/stokes_shahbazi/stokes_shahbazi.h"

// Navier-Stokes flow - analytical solutions
#include "incompressible_navier_stokes_test_cases/beltrami/beltrami.h"
#include "incompressible_navier_stokes_test_cases/cavity/cavity.h"
#include "incompressible_navier_stokes_test_cases/couette/couette.h"
#include "incompressible_navier_stokes_test_cases/free_stream/free_stream.h"
#include "incompressible_navier_stokes_test_cases/kovasznay/kovasznay.h"
#include "incompressible_navier_stokes_test_cases/orr_sommerfeld/orr_sommerfeld.h"
#include "incompressible_navier_stokes_test_cases/poiseuille/poiseuille.h"
#include "incompressible_navier_stokes_test_cases/unstable_beltrami/unstable_beltrami.h"
#include "incompressible_navier_stokes_test_cases/vortex/vortex.h"
#include "incompressible_navier_stokes_test_cases/vortex_periodic/vortex_periodic.h"

// more complex applications and turbulence
#include "incompressible_navier_stokes_test_cases/fda/fda_nozzle_benchmark.h"
#include "incompressible_navier_stokes_test_cases/flow_past_cylinder/flow_past_cylinder.h"
#include "incompressible_navier_stokes_test_cases/kelvin_helmholtz/kelvin_helmholtz.h"
#include "incompressible_navier_stokes_test_cases/periodic_hill/periodic_hill.h"
#include "incompressible_navier_stokes_test_cases/shear_layer/shear_layer.h"
#include "incompressible_navier_stokes_test_cases/taylor_green_vortex/taylor_green_vortex.h"
#include "incompressible_navier_stokes_test_cases/tum/tum.h"
#include "incompressible_navier_stokes_test_cases/turbulent_channel/turbulent_channel.h"

// incompressible flow with scalar transport (but can also be used for pure fluid simulations)
#include "../include/incompressible_flow_with_transport/user_interface/application_base.h"
#include "incompressible_flow_with_transport_test_cases/lung/lung.h"

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

      std::shared_ptr<IncNS::ApplicationBase<dim, Number>> app;
      if(name == "Template")
        app.reset(new IncNS::Template::Application<dim, Number>());
      else if(name == "StokesGuermond")
        app.reset(new IncNS::StokesGuermond::Application<dim, Number>());
      else if(name == "StokesShahbazi")
        app.reset(new IncNS::StokesShahbazi::Application<dim, Number>());
      else if(name == "StokesCurlFlow")
        app.reset(new IncNS::StokesCurlFlow::Application<dim, Number>());
      else if(name == "FreeStream")
        app.reset(new IncNS::FreeStream::Application<dim, Number>());
      else if(name == "Couette")
        app.reset(new IncNS::Couette::Application<dim, Number>());
      else if(name == "Poiseuille")
        app.reset(new IncNS::Poiseuille::Application<dim, Number>());
      else if(name == "Cavity")
        app.reset(new IncNS::Cavity::Application<dim, Number>());
      else if(name == "Kovasznay")
        app.reset(new IncNS::Kovasznay::Application<dim, Number>());
      else if(name == "VortexPeriodic")
        app.reset(new IncNS::VortexPeriodic::Application<dim, Number>());
      else if(name == "Vortex")
        app.reset(new IncNS::Vortex::Application<dim, Number>());
      else if(name == "OrrSommerfeld")
        app.reset(new IncNS::OrrSommerfeld::Application<dim, Number>());
      else if(name == "Beltrami")
        app.reset(new IncNS::Beltrami::Application<dim, Number>());
      else if(name == "UnstableBeltrami")
        app.reset(new IncNS::UnstableBeltrami::Application<dim, Number>());
      else if(name == "KelvinHelmholtz")
        app.reset(new IncNS::KelvinHelmholtz::Application<dim, Number>());
      else if(name == "ShearLayer")
        app.reset(new IncNS::ShearLayer::Application<dim, Number>());
      else if(name == "TUM")
        app.reset(new IncNS::TUM::Application<dim, Number>());
      else if(name == "FlowPastCylinder")
        app.reset(new IncNS::FlowPastCylinder::Application<dim, Number>());
      else if(name == "TaylorGreenVortex")
        app.reset(new IncNS::TaylorGreenVortex::Application<dim, Number>());
      else if(name == "TurbulentChannel")
        app.reset(new IncNS::TurbulentChannel::Application<dim, Number>());
      else if(name == "PeriodicHill")
        app.reset(new IncNS::PeriodicHill::Application<dim, Number>());
      else if(name == "FDA")
        app.reset(new IncNS::FDA::Application<dim, Number>());
      else if(name == "Lung")
        app.reset(new FTI::Lung::Application<dim, Number>());
      else
        AssertThrow(false, ExcMessage("This application does not exist!"));

      app->add_parameters(prm);
    }
  }

  template<int dim, typename Number>
  std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    dealii::ParameterHandler prm;
    this->add_name_parameter(prm);
    parse_input(input_file, prm, true, true);

    std::shared_ptr<IncNS::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new IncNS::Template::Application<dim, Number>(input_file));
    else if(name == "StokesGuermond")
      app.reset(new IncNS::StokesGuermond::Application<dim, Number>(input_file));
    else if(name == "StokesShahbazi")
      app.reset(new IncNS::StokesShahbazi::Application<dim, Number>(input_file));
    else if(name == "StokesCurlFlow")
      app.reset(new IncNS::StokesCurlFlow::Application<dim, Number>(input_file));
    else if(name == "FreeStream")
      app.reset(new IncNS::FreeStream::Application<dim, Number>(input_file));
    else if(name == "Couette")
      app.reset(new IncNS::Couette::Application<dim, Number>(input_file));
    else if(name == "Poiseuille")
      app.reset(new IncNS::Poiseuille::Application<dim, Number>(input_file));
    else if(name == "Cavity")
      app.reset(new IncNS::Cavity::Application<dim, Number>(input_file));
    else if(name == "Kovasznay")
      app.reset(new IncNS::Kovasznay::Application<dim, Number>(input_file));
    else if(name == "VortexPeriodic")
      app.reset(new IncNS::VortexPeriodic::Application<dim, Number>(input_file));
    else if(name == "Vortex")
      app.reset(new IncNS::Vortex::Application<dim, Number>(input_file));
    else if(name == "OrrSommerfeld")
      app.reset(new IncNS::OrrSommerfeld::Application<dim, Number>(input_file));
    else if(name == "Beltrami")
      app.reset(new IncNS::Beltrami::Application<dim, Number>(input_file));
    else if(name == "UnstableBeltrami")
      app.reset(new IncNS::UnstableBeltrami::Application<dim, Number>(input_file));
    else if(name == "KelvinHelmholtz")
      app.reset(new IncNS::KelvinHelmholtz::Application<dim, Number>(input_file));
    else if(name == "ShearLayer")
      app.reset(new IncNS::ShearLayer::Application<dim, Number>(input_file));
    else if(name == "TUM")
      app.reset(new IncNS::TUM::Application<dim, Number>(input_file));
    else if(name == "FlowPastCylinder")
      app.reset(new IncNS::FlowPastCylinder::Application<dim, Number>(input_file));
    else if(name == "TaylorGreenVortex")
      app.reset(new IncNS::TaylorGreenVortex::Application<dim, Number>(input_file));
    else if(name == "TurbulentChannel")
      app.reset(new IncNS::TurbulentChannel::Application<dim, Number>(input_file));
    else if(name == "PeriodicHill")
      app.reset(new IncNS::PeriodicHill::Application<dim, Number>(input_file));
    else if(name == "FDA")
      app.reset(new IncNS::FDA::Application<dim, Number>(input_file));
    else if(name == "Lung")
      app.reset(new FTI::Lung::Application<dim, Number>(input_file));
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

  ConvergenceStudy study;
  study.add_parameters(prm);

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
run(std::string const & input_file,
    unsigned int const  degree,
    unsigned int const  refine_space,
    unsigned int const  refine_time,
    MPI_Comm const &    mpi_comm)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<IncNS::Driver<dim, Number>> driver;
  driver.reset(new IncNS::Driver<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<IncNS::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  driver->setup(application, degree, refine_space, refine_time);

  driver->solve();

  driver->print_statistics(timer.wall_time());
}

//#define USE_SUB_COMMUNICATOR

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  // new communicator
  MPI_Comm sub_comm;

#ifdef USE_SUB_COMMUNICATOR
  // use stride of n cores
  const int n = 48; // 24;

  const int rank     = Utilities::MPI::this_mpi_process(mpi_comm);
  const int size     = Utilities::MPI::n_mpi_processes(mpi_comm);
  const int flag     = 1;
  const int new_rank = rank + (rank % n) * size;

  // split default communicator into two groups
  MPI_Comm_split(mpi_comm, flag, new_rank, &sub_comm);

  if(rank == 0)
    std::cout << std::endl << "Created sub communicator with stride of " << n << std::endl;
#else
  sub_comm = mpi_comm;
#endif

  // check if parameter file is provided

  // ./incompressible_navier_stokes
  AssertThrow(argc > 1, ExcMessage("No parameter file has been provided!"));

  // ./incompressible_navier_stokes --help
  if(argc == 2 && std::string(argv[1]) == "--help")
  {
    if(dealii::Utilities::MPI::this_mpi_process(sub_comm) == 0)
      create_input_file();

    return 0;
  }
  // ./incompressible_navier_stokes --help NameOfApplication
  else if(argc == 3 && std::string(argv[1]) == "--help")
  {
    if(dealii::Utilities::MPI::this_mpi_process(sub_comm) == 0)
      create_input_file(argv[2]);

    return 0;
  }

  // the second argument is the input-file
  // ./incompressible_navier_stokes InputFile
  std::string      input_file = std::string(argv[1]);
  ConvergenceStudy study(input_file);

  // k-refinement
  for(unsigned int degree = study.degree_min; degree <= study.degree_max; ++degree)
  {
    // h-refinement
    for(unsigned int refine_space = study.refine_space_min; refine_space <= study.refine_space_max;
        ++refine_space)
    {
      // dt-refinement
      for(unsigned int refine_time = study.refine_time_min; refine_time <= study.refine_time_max;
          ++refine_time)
      {
        // run the simulation
        if(study.dim == 2 && study.precision == "float")
          run<2, float>(input_file, degree, refine_space, refine_time, sub_comm);
        else if(study.dim == 2 && study.precision == "double")
          run<2, double>(input_file, degree, refine_space, refine_time, sub_comm);
        else if(study.dim == 3 && study.precision == "float")
          run<3, float>(input_file, degree, refine_space, refine_time, sub_comm);
        else if(study.dim == 3 && study.precision == "double")
          run<3, double>(input_file, degree, refine_space, refine_time, sub_comm);
        else
          AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
      }
    }
  }

#ifdef USE_SUB_COMMUNICATOR
  // free communicator
  MPI_Comm_free(&sub_comm);
#endif

  return 0;
}
