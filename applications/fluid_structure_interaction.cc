/*
 * fluid_structure_interaction.cc
 *
 *  Created on: Feb 25, 2020
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// IncNS: postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor_base.h"

// IncNS: spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_pressure_correction.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/interface.h"

// IncNS: temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/driver_steady_problems.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// IncNS: Parameters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"

// Poisson: spatial discretization
#include "../include/poisson/spatial_discretization/operator.h"

// Poisson: user interface, etc.
#include "../include/poisson/user_interface/analytical_solution.h"
#include "../include/poisson/user_interface/field_functions.h"
#include "../include/poisson/user_interface/input_parameters.h"

#include "../include/convection_diffusion/user_interface/boundary_descriptor.h"

// general functionalities
#include "../include/functionalities/interface_coupling.h"
#include "../include/functionalities/mapping_degree.h"
#include "../include/functionalities/matrix_free_wrapper.h"
#include "../include/functionalities/moving_mesh.h"
#include "../include/functionalities/print_general_infos.h"
#include "../include/functionalities/verify_boundary_conditions.h"

using namespace dealii;
// using namespace IncNS;

// specify the flow problem that has to be solved

// template
#include "fluid_structure_interaction_test_cases/template.h"

//#include "fluid_structure_interaction_test_cases/vortex.h"
//#include "fluid_structure_interaction_test_cases/bending_wall.h"
//#include "fluid_structure_interaction_test_cases/cylinder_with_flag.h"

template<typename Number>
class ProblemBase
{
public:
  virtual ~ProblemBase()
  {
  }

  virtual void
  setup(IncNS::InputParameters const &   fluid_param,
        Poisson::InputParameters const & poisson_param) = 0;

  virtual void
  solve() const = 0;

  virtual void
  analyze_computing_times() const = 0;
};

template<int dim, typename Number>
class Problem : public ProblemBase<Number>
{
public:
  Problem(MPI_Comm const & comm);

  void
  setup(IncNS::InputParameters const & fluid_param, Poisson::InputParameters const & poisson_param);

  void
  solve() const;

  void
  analyze_computing_times() const;

private:
  void
  print_header() const;

  /****************************************** FLUID *******************************************/

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> fluid_triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    fluid_periodic_faces;

  // solve mesh deformation by a Poisson problem
  Poisson::InputParameters poisson_param;

  std::shared_ptr<Poisson::FieldFunctions<dim>>         poisson_field_functions;
  std::shared_ptr<ConvDiff::BoundaryDescriptor<1, dim>> poisson_boundary_descriptor;

  // static mesh for Poisson problem
  std::shared_ptr<Mesh<dim>> poisson_mesh;

  std::shared_ptr<MatrixFreeWrapper<dim, Number>>      poisson_matrix_free_wrapper;
  std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson_operator;

  IncNS::InputParameters fluid_param;

  std::shared_ptr<IncNS::FieldFunctions<dim>>      fluid_field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> fluid_boundary_descriptor_velocity;
  std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> fluid_boundary_descriptor_pressure;

  // moving mesh for fluid problem
  std::shared_ptr<Mesh<dim>>                   fluid_mesh;
  std::shared_ptr<MovingMeshBase<dim, Number>> fluid_moving_mesh;

  std::shared_ptr<MatrixFreeWrapper<dim, Number>> fluid_matrix_free_wrapper;

  // Spatial discretization
  typedef IncNS::DGNavierStokesBase<dim, Number>               DGBase;
  typedef IncNS::DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef IncNS::DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef IncNS::DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase>               fluid_operator;
  std::shared_ptr<DGCoupled>            fluid_operator_coupled;
  std::shared_ptr<DGDualSplitting>      fluid_operator_dual_splitting;
  std::shared_ptr<DGPressureCorrection> fluid_operator_pressure_correction;

  // Temporal discretization
  typedef IncNS::TimeIntBDF<dim, Number>                   TimeInt;
  typedef IncNS::TimeIntBDFCoupled<dim, Number>            TimeIntCoupled;
  typedef IncNS::TimeIntBDFDualSplitting<dim, Number>      TimeIntDualSplitting;
  typedef IncNS::TimeIntBDFPressureCorrection<dim, Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> fluid_time_integrator;

  // Postprocessor
  typedef IncNS::PostProcessorBase<dim, Number> Postprocessor;
  std::shared_ptr<Postprocessor>                fluid_postprocessor;

  /****************************************** FLUID *******************************************/


  /**************************************** STRUCTURE *****************************************/

  // TODO

  /**************************************** STRUCTURE *****************************************/


  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/

  // TODO
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> structure_to_fluid;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> structure_to_moving_mesh;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> fluid_to_structure;

  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/


  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;
  mutable double ale_update_time;
};

template<int dim, typename Number>
Problem<dim, Number>::Problem(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
    overall_time(0.0),
    setup_time(0.0),
    ale_update_time(0.0)
{
}

template<int dim, typename Number>
void
Problem<dim, Number>::print_header() const
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "           High-order solver for fluid-structure interaction problems            " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Problem<dim, Number>::setup(IncNS::InputParameters const &   fluid_param_in,
                            Poisson::InputParameters const & poisson_param_in)
{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  // input parameters
  fluid_param = fluid_param_in;
  fluid_param.check_input_parameters(pcout);
  fluid_param.print(pcout, "List of input parameters for incompressible flow solver:");

  poisson_param = poisson_param_in;
  poisson_param.check_input_parameters();
  poisson_param.print(pcout, "List of input parameters for Poisson solver (moving mesh):");

  // Some FSI specific Asserts
  AssertThrow(fluid_param.problem_type == IncNS::ProblemType::Unsteady,
              ExcMessage("Invalid parameter in context of fluid-structure interaction."));
  AssertThrow(fluid_param.ale_formulation == true,
              ExcMessage("Invalid parameter in context of fluid-structure interaction."));

  // triangulation
  if(fluid_param.triangulation_type == TriangulationType::Distributed)
  {
    fluid_triangulation.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(fluid_param.triangulation_type == TriangulationType::FullyDistributed)
  {
    fluid_triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  create_grid_and_set_boundary_ids(fluid_triangulation,
                                   fluid_param.h_refinements,
                                   fluid_periodic_faces);
  print_grid_data(pcout, fluid_param.h_refinements, *fluid_triangulation);

  // field functions and boundary conditions

  // fluid
  fluid_boundary_descriptor_velocity.reset(new IncNS::BoundaryDescriptorU<dim>());
  fluid_boundary_descriptor_pressure.reset(new IncNS::BoundaryDescriptorP<dim>());
  IncNS::set_boundary_conditions(fluid_boundary_descriptor_velocity,
                                 fluid_boundary_descriptor_pressure);
  verify_boundary_conditions(*fluid_boundary_descriptor_velocity,
                             *fluid_triangulation,
                             fluid_periodic_faces);
  verify_boundary_conditions(*fluid_boundary_descriptor_pressure,
                             *fluid_triangulation,
                             fluid_periodic_faces);

  fluid_field_functions.reset(new IncNS::FieldFunctions<dim>());
  IncNS::set_field_functions(fluid_field_functions);

  // poisson
  poisson_boundary_descriptor.reset(new ConvDiff::BoundaryDescriptor<1, dim>());
  Poisson::set_boundary_conditions(poisson_boundary_descriptor);
  verify_boundary_conditions(*poisson_boundary_descriptor,
                             *fluid_triangulation,
                             fluid_periodic_faces);

  poisson_field_functions.reset(new Poisson::FieldFunctions<dim>());
  Poisson::set_field_functions(poisson_field_functions);

  AssertThrow(poisson_param.right_hand_side == false,
              ExcMessage("Parameter does not make sense in context of FSI."));

  // mapping for Poisson solver (static mesh)
  unsigned int const mapping_degree =
    get_mapping_degree(poisson_param.mapping, poisson_param.degree);
  poisson_mesh.reset(new Mesh<dim>(mapping_degree));

  // initialize Poisson operator
  poisson_operator.reset(new Poisson::Operator<dim, Number, dim>(*fluid_triangulation,
                                                                 poisson_mesh->get_mapping(),
                                                                 fluid_periodic_faces,
                                                                 poisson_boundary_descriptor,
                                                                 poisson_field_functions,
                                                                 poisson_param,
                                                                 mpi_comm));

  // initialize matrix_free
  poisson_matrix_free_wrapper.reset(
    new MatrixFreeWrapper<dim, Number>(poisson_mesh->get_mapping()));
  poisson_matrix_free_wrapper->append_data_structures(*poisson_operator);
  poisson_matrix_free_wrapper->reinit(poisson_param.enable_cell_based_face_loops,
                                      fluid_triangulation);

  poisson_operator->setup(poisson_matrix_free_wrapper);
  poisson_operator->setup_solver();

  // mapping for fluid problem (moving mesh)
  {
    unsigned int const mapping_degree =
      get_mapping_degree(fluid_param.mapping, fluid_param.degree_u);

    // TODO
    if(fluid_param.ale_formulation) // moving mesh
    {
      fluid_moving_mesh.reset(new MovingMeshPoisson<dim, Number>(
        *fluid_triangulation, mapping_degree, mpi_comm, poisson_operator, fluid_param.start_time));

      //      std::shared_ptr<Function<dim>> mesh_motion = set_mesh_movement_function<dim>();
      //      fluid_moving_mesh.reset(new MovingMeshAnalytical<dim, Number>(
      //        *fluid_triangulation, mapping_degree, fluid_param.degree_u, mpi_comm, mesh_motion,
      //        fluid_param.start_time));

      fluid_mesh = fluid_moving_mesh;
    }
    else // static mesh
    {
      fluid_mesh.reset(new Mesh<dim>(mapping_degree));
    }
  }

  // initialize fluid_operator
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    fluid_operator_coupled.reset(new DGCoupled(*fluid_triangulation,
                                               fluid_mesh->get_mapping(),
                                               fluid_periodic_faces,
                                               fluid_boundary_descriptor_velocity,
                                               fluid_boundary_descriptor_pressure,
                                               fluid_field_functions,
                                               fluid_param,
                                               mpi_comm));

    fluid_operator = fluid_operator_coupled;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    fluid_operator_dual_splitting.reset(new DGDualSplitting(*fluid_triangulation,
                                                            fluid_mesh->get_mapping(),
                                                            fluid_periodic_faces,
                                                            fluid_boundary_descriptor_velocity,
                                                            fluid_boundary_descriptor_pressure,
                                                            fluid_field_functions,
                                                            fluid_param,
                                                            mpi_comm));

    fluid_operator = fluid_operator_dual_splitting;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    fluid_operator_pressure_correction.reset(
      new DGPressureCorrection(*fluid_triangulation,
                               fluid_mesh->get_mapping(),
                               fluid_periodic_faces,
                               fluid_boundary_descriptor_velocity,
                               fluid_boundary_descriptor_pressure,
                               fluid_field_functions,
                               fluid_param,
                               mpi_comm));

    fluid_operator = fluid_operator_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize matrix_free
  fluid_matrix_free_wrapper.reset(new MatrixFreeWrapper<dim, Number>(fluid_mesh->get_mapping()));
  fluid_matrix_free_wrapper->append_data_structures(*fluid_operator);
  fluid_matrix_free_wrapper->reinit(fluid_param.use_cell_based_face_loops, fluid_triangulation);

  // setup Navier-Stokes operator
  fluid_operator->setup(fluid_matrix_free_wrapper);

  // setup postprocessor
  fluid_postprocessor = IncNS::construct_postprocessor<dim, Number>(fluid_param, mpi_comm);
  fluid_postprocessor->setup(*fluid_operator);

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  AssertThrow(fluid_param.solver_type == IncNS::SolverType::Unsteady,
              ExcMessage("Invalid parameter in context of fluid-structure interaction."));

  // initialize fluid_operator
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    fluid_time_integrator.reset(new TimeIntCoupled(fluid_operator_coupled,
                                                   fluid_param,
                                                   mpi_comm,
                                                   fluid_postprocessor,
                                                   fluid_moving_mesh,
                                                   fluid_matrix_free_wrapper));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    fluid_time_integrator.reset(new TimeIntDualSplitting(fluid_operator_dual_splitting,
                                                         fluid_param,
                                                         mpi_comm,
                                                         fluid_postprocessor,
                                                         fluid_moving_mesh,
                                                         fluid_matrix_free_wrapper));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    fluid_time_integrator.reset(new TimeIntPressureCorrection(fluid_operator_pressure_correction,
                                                              fluid_param,
                                                              mpi_comm,
                                                              fluid_postprocessor,
                                                              fluid_moving_mesh,
                                                              fluid_matrix_free_wrapper));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  fluid_time_integrator->setup(fluid_param.restarted_simulation);

  fluid_operator->setup_solvers(fluid_time_integrator->get_scaling_factor_time_derivative_term(),
                                fluid_time_integrator->get_velocity());

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::solve() const
{
  do
  {
    fluid_time_integrator->advance_one_timestep_pre_solve();

    // partitioned iteration
    unsigned int const N_ITER_MAX = 1;
    for(unsigned int iter = 0; iter < N_ITER_MAX; ++iter)
    {
      // TODO
      LinearAlgebra::distributed::Vector<Number> vec_displacements, vec_velocity;
      if(iter == 0)
      {
        // extrapolate structural displacements and fluid velocity at interface
      }
      else
      {
        // use structural displacements of last iteration and compute
        // fluid velocity at interface using the new structural displacements
      }

      // move the mesh and update dependent data structures
      Timer timer;
      timer.restart();

      // structure_to_moving_mesh->update_data(vec_displacements);
      fluid_moving_mesh->move_mesh(fluid_time_integrator->get_next_time());
      fluid_matrix_free_wrapper->update_mapping();
      fluid_operator->update_after_mesh_movement();
      fluid_time_integrator->ale_update();

      ale_update_time += timer.wall_time();

      // structure_to_fluid->update_data(vec_velocity);
      fluid_time_integrator->advance_one_timestep_solve();

      // TODO update stress boundary condition for solid

      // TODO solve structural problem
    }

    fluid_time_integrator->advance_one_timestep_post_solve();
  } while(!fluid_time_integrator->finished());

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for fluid-structure interaction solver:" << std::endl;

  // Iterations
  {
    this->pcout << std::endl << "Average number of iterations:" << std::endl;

    std::vector<std::string> names;
    std::vector<double>      iterations;

    fluid_time_integrator->get_iterations(names, iterations);

    unsigned int length = 1;
    for(unsigned int i = 0; i < names.size(); ++i)
    {
      length = length > names[i].length() ? length : names[i].length();
    }

    for(unsigned int i = 0; i < iterations.size(); ++i)
    {
      this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
                  << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                  << std::endl;
    }
  }

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(overall_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  // wall times
  this->pcout << std::endl << "Wall times:" << std::endl;

  std::vector<std::string> names;
  std::vector<double>      computing_times;

  fluid_time_integrator->get_wall_times(names, computing_times);

  unsigned int length = 1;
  for(unsigned int i = 0; i < names.size(); ++i)
  {
    length = length > names[i].length() ? length : names[i].length();
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(computing_times[i], mpi_comm);
    this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  Utilities::MPI::MinMaxAvg setup_time_data = Utilities::MPI::min_max_avg(setup_time, mpi_comm);
  double const              setup_time_avg  = setup_time_data.avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  Utilities::MPI::MinMaxAvg ale_time_data = Utilities::MPI::min_max_avg(ale_update_time, mpi_comm);
  double const              ale_time_avg  = ale_time_data.avg;
  if(this->fluid_param.ale_formulation)
  {
    this->pcout << "  " << std::setw(length + 2) << std::left << "ALE update"
                << std::setprecision(2) << std::scientific << std::setw(10) << std::right
                << ale_time_avg << " s  " << std::setprecision(2) << std::fixed << std::setw(6)
                << std::right << ale_time_avg / overall_time_avg * 100 << " %" << std::endl;
  }

  double other = overall_time_avg - sum_of_substeps - setup_time_avg;
  if(this->fluid_param.ale_formulation)
    other -= ale_time_avg;

  this->pcout << "  " << std::setw(length + 2) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length + 2) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  this->pcout << std::endl
              << "Computational costs (including setup + postprocessing):" << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl;

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs = this->fluid_operator->get_number_of_dofs();

  unsigned int N_time_steps      = fluid_time_integrator->get_number_of_time_steps();
  double const time_per_timestep = overall_time_avg / (double)N_time_steps;
  this->pcout << std::endl
              << "Throughput per time step (including setup + postprocessing):" << std::endl
              << "  Degrees of freedom      = " << DoFs << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Time steps              = " << std::left << N_time_steps << std::endl
              << "  Wall time per time step = " << std::scientific << std::setprecision(2)
              << time_per_timestep << " s" << std::endl
              << "  Throughput              = " << std::scientific << std::setprecision(2)
              << DoFs / (time_per_timestep * N_mpi_processes) << " DoFs/s/core" << std::endl;


  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    MPI_Comm mpi_comm(MPI_COMM_WORLD);

    // set parameters
    IncNS::InputParameters fluid_param;
    IncNS::set_input_parameters(fluid_param);

    Poisson::InputParameters poisson_param;
    Poisson::set_input_parameters(poisson_param);

    // check parameters in case of restart
    if(fluid_param.restarted_simulation)
    {
      AssertThrow(REFINE_SPACE_MIN == REFINE_SPACE_MAX,
                  ExcMessage("Spatial refinement not possible in combination with restart!"));

      AssertThrow(REFINE_TIME_MIN == REFINE_TIME_MAX,
                  ExcMessage("Temporal refinement not possible in combination with restart!"));
    }

    // h-refinement
    for(unsigned int h_refinements = REFINE_SPACE_MIN; h_refinements <= REFINE_SPACE_MAX;
        ++h_refinements)
    {
      // dt-refinement
      for(unsigned int dt_refinements = REFINE_TIME_MIN; dt_refinements <= REFINE_TIME_MAX;
          ++dt_refinements)
      {
        // reset parameters
        fluid_param.h_refinements  = h_refinements;
        fluid_param.dt_refinements = dt_refinements;

        poisson_param.h_refinements = h_refinements;

        // setup problem and run simulation
        typedef double                       Number;
        std::shared_ptr<ProblemBase<Number>> problem;

        if(fluid_param.dim == 2)
          problem.reset(new Problem<2, Number>(mpi_comm));
        else if(fluid_param.dim == 3)
          problem.reset(new Problem<3, Number>(mpi_comm));
        else
          AssertThrow(false, ExcMessage("Only dim=2 and dim=3 implemented."));

        problem->setup(fluid_param, poisson_param);

        problem->solve();

        problem->analyze_computing_times();
      }
    }
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
  return 0;
}
