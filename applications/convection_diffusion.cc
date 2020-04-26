/*
 * convection_diffusion.cc
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

// driver
#include "../include/convection_diffusion/driver.h"

// infrastructure for convergence studies
#include "../include/utilities/convergence_study.h"

// applications
#include "convection_diffusion_test_cases/template/application.h"

// applications - convection
#include "convection_diffusion_test_cases/deforming_hill/deforming_hill.h"
#include "convection_diffusion_test_cases/rotating_hill/rotating_hill.h"
#include "convection_diffusion_test_cases/sine_wave/sine_wave.h"

// applications - diffusion
#include "convection_diffusion_test_cases/decaying_hill/decaying_hill.h"

// applications - convection-diffusion
#include "convection_diffusion_test_cases/boundary_layer/boundary_layer.h"
#include "convection_diffusion_test_cases/const_rhs_const_or_circular_wind/const_rhs.h"

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

      std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>> app;
      if(name == "Template")
        app.reset(new ConvDiff::Template::Application<dim, Number>());
      else if(name == "SineWave")
        app.reset(new ConvDiff::SineWave::Application<dim, Number>());
      else if(name == "DeformingHill")
        app.reset(new ConvDiff::DeformingHill::Application<dim, Number>());
      else if(name == "RotatingHill")
        app.reset(new ConvDiff::RotatingHill::Application<dim, Number>());
      else if(name == "DecayingHill")
        app.reset(new ConvDiff::DecayingHill::Application<dim, Number>());
      else if(name == "BoundaryLayer")
        app.reset(new ConvDiff::BoundaryLayer::Application<dim, Number>());
      else if(name == "ConstRHS")
        app.reset(new ConvDiff::ConstRHS::Application<dim, Number>());
      else
        AssertThrow(false, ExcMessage("This application does not exist!"));

      app->add_parameters(prm);
    }
  }

  template<int dim, typename Number>
  std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>>
  get_application(std::string input_file)
  {
    dealii::ParameterHandler prm;
    this->add_name_parameter(prm);
    parse_input(input_file, prm, true, true);

    std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>> app;
    if(name == "Template")
      app.reset(new ConvDiff::Template::Application<dim, Number>(input_file));
    else if(name == "SineWave")
      app.reset(new ConvDiff::SineWave::Application<dim, Number>(input_file));
    else if(name == "DeformingHill")
      app.reset(new ConvDiff::DeformingHill::Application<dim, Number>(input_file));
    else if(name == "RotatingHill")
      app.reset(new ConvDiff::RotatingHill::Application<dim, Number>(input_file));
    else if(name == "DecayingHill")
      app.reset(new ConvDiff::DecayingHill::Application<dim, Number>(input_file));
    else if(name == "BoundaryLayer")
      app.reset(new ConvDiff::BoundaryLayer::Application<dim, Number>(input_file));
    else if(name == "ConstRHS")
      app.reset(new ConvDiff::ConstRHS::Application<dim, Number>(input_file));
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

  std::shared_ptr<ConvDiff::Driver<dim, Number>> solver;
  solver.reset(new ConvDiff::Driver<dim, Number>(mpi_comm));

  ApplicationSelector selector;

  std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>> application =
    selector.get_application<dim, Number>(input_file);

  solver->setup(application, degree, refine_space, refine_time);

  solver->solve();

  solver->print_statistics(timer.wall_time());
}

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  // check if parameter file is provided

  // ./convection_diffusion
  AssertThrow(argc > 1, ExcMessage("No parameter file has been provided!"));

  // ./convection_diffusion --help
  if(argc == 2 && std::string(argv[1]) == "--help")
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      create_input_file();

    return 0;
  }
  // ./convection_diffusion --help NameOfApplication
  else if(argc == 3 && std::string(argv[1]) == "--help")
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      create_input_file(argv[2]);

    return 0;
  }

  // the second argument is the input-file
  // ./convection_diffusion InputFile
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
          run<2, float>(input_file, degree, refine_space, refine_time, mpi_comm);
        else if(study.dim == 2 && study.precision == "double")
          run<2, double>(input_file, degree, refine_space, refine_time, mpi_comm);
        else if(study.dim == 3 && study.precision == "float")
          run<3, float>(input_file, degree, refine_space, refine_time, mpi_comm);
        else if(study.dim == 3 && study.precision == "double")
          run<3, double>(input_file, degree, refine_space, refine_time, mpi_comm);
        else
          AssertThrow(false, ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
      }
    }
  }

  return 0;
}
