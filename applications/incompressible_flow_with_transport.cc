/*
 * incompressible_flow_with_transport.cc
 *
 *  Created on: Nov 6, 2018
 *      Author: fehn
 */

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// CONVECTION-DIFFUSION

// postprocessor
#include "convection_diffusion/postprocessor/postprocessor_base.h"

// spatial discretization
#include "convection_diffusion/spatial_discretization/dg_operator.h"

// time integration
#include "convection_diffusion/time_integration/time_int_bdf.h"
#include "convection_diffusion/time_integration/time_int_explicit_runge_kutta.h"

// user interface, etc.
#include "convection_diffusion/user_interface/boundary_descriptor.h"
#include "convection_diffusion/user_interface/field_functions.h"
#include "convection_diffusion/user_interface/input_parameters.h"

// NAVIER-STOKES

// postprocessor
#include "incompressible_navier_stokes/postprocessor/postprocessor.h"

// spatial discretization
#include "incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h"
#include "incompressible_navier_stokes/spatial_discretization/dg_dual_splitting.h"
#include "incompressible_navier_stokes/spatial_discretization/dg_pressure_correction.h"
#include "incompressible_navier_stokes/spatial_discretization/interface.h"

// temporal discretization
#include "incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// Parameters, BCs, etc.
#include "incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "incompressible_navier_stokes/user_interface/field_functions.h"
#include "incompressible_navier_stokes/user_interface/input_parameters.h"

// general functionalities
#include "functionalities/mapping_degree.h"
#include "functionalities/matrix_free_wrapper.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"

using namespace dealii;

// specify the test case that has to be solved

// template
#include "incompressible_flow_with_transport_test_cases/template.h"

// passive scalar
//#include "incompressible_flow_with_transport_test_cases/cavity.h"
//#include "incompressible_flow_with_transport_test_cases/lung.h"

// natural convection (active scalar)
//#include "incompressible_flow_with_transport_test_cases/cavity_natural_convection.h"
//#include "incompressible_flow_with_transport_test_cases/rayleigh_bernard.h"
//#include "incompressible_flow_with_transport_test_cases/rising_bubble.h"
//#include "incompressible_flow_with_transport_test_cases/mantle_convection.h"

template<typename Number>
class ProblemBase
{
public:
  virtual ~ProblemBase()
  {
  }

  virtual void
  setup(IncNS::InputParameters const &                 fluid_param_in,
        std::vector<ConvDiff::InputParameters> const & scalar_param_in) = 0;

  virtual void
  solve() const = 0;

  virtual void
  analyze_computing_times() const = 0;
};

template<int dim, typename Number = double>
class Problem : public ProblemBase<Number>
{
public:
  Problem(MPI_Comm const & mpi_comm, unsigned int const n_scalars);

  void
  setup(IncNS::InputParameters const &                 fluid_param_in,
        std::vector<ConvDiff::InputParameters> const & scalar_param_in);

  void
  solve() const;

  void
  analyze_computing_times() const;

private:
  // GENERAL (FLUID + TRANSPORT)
  void
  print_header() const;

  void
  communicate_scalar_to_fluid() const;

  void
  communicate_fluid_to_all_scalars() const;

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  double
  analyze_computing_times_fluid(double const overall_time) const;

  void
  analyze_iterations_fluid() const;

  double
  analyze_computing_times_transport(double const overall_time) const;

  void
  analyze_iterations_transport() const;

  MPI_Comm const & mpi_comm;

  ConditionalOStream pcout;

  /*
   * Mesh: triangulation, mapping
   */
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  // mapping (static and moving meshes)
  std::shared_ptr<Mesh<dim>>                         mesh;
  std::shared_ptr<MovingMeshAnalytical<dim, Number>> moving_mesh;

  // number of scalar quantities
  unsigned int const n_scalars;

  bool use_adaptive_time_stepping;

  // INCOMPRESSIBLE NAVIER-STOKES
  std::shared_ptr<IncNS::FieldFunctions<dim>>      fluid_field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> fluid_boundary_descriptor_velocity;
  std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> fluid_boundary_descriptor_pressure;

  IncNS::InputParameters fluid_param;

  //  MatrixFree
  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper;

  typedef IncNS::DGNavierStokesBase<dim, Number>               DGBase;
  typedef IncNS::DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef IncNS::DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef IncNS::DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase>               navier_stokes_operator;
  std::shared_ptr<DGCoupled>            navier_stokes_operator_coupled;
  std::shared_ptr<DGDualSplitting>      navier_stokes_operator_dual_splitting;
  std::shared_ptr<DGPressureCorrection> navier_stokes_operator_pressure_correction;

  typedef IncNS::PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> fluid_postprocessor;

  typedef IncNS::TimeIntBDF<dim, Number>                   TimeInt;
  typedef IncNS::TimeIntBDFCoupled<dim, Number>            TimeIntCoupled;
  typedef IncNS::TimeIntBDFDualSplitting<dim, Number>      TimeIntDualSplitting;
  typedef IncNS::TimeIntBDFPressureCorrection<dim, Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> fluid_time_integrator;

  // SCALAR TRANSPORT

  std::vector<ConvDiff::InputParameters> scalar_param;

  std::vector<std::shared_ptr<ConvDiff::FieldFunctions<dim>>>     scalar_field_functions;
  std::vector<std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>>> scalar_boundary_descriptor;

  std::vector<std::shared_ptr<ConvDiff::DGOperator<dim, Number>>> conv_diff_operator;

  std::vector<std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>> scalar_postprocessor;

  std::vector<std::shared_ptr<TimeIntBase>> scalar_time_integrator;

  mutable LinearAlgebra::distributed::Vector<Number> temperature;

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;

  unsigned int const length = 15;
};

template<int dim, typename Number>
Problem<dim, Number>::Problem(MPI_Comm const & comm, unsigned int const n_scalars)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    n_scalars(n_scalars),
    use_adaptive_time_stepping(false),
    overall_time(0.0),
    setup_time(0.0)
{
  scalar_param.resize(n_scalars);
  scalar_field_functions.resize(n_scalars);
  scalar_boundary_descriptor.resize(n_scalars);

  conv_diff_operator.resize(n_scalars);
  scalar_postprocessor.resize(n_scalars);
  scalar_time_integrator.resize(n_scalars);
}

template<int dim, typename Number>
void
Problem<dim, Number>::print_header() const
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                unsteady, incompressible Navier-Stokes equations                 " << std::endl
  << "                             with scalar transport.                              " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Problem<dim, Number>::setup(IncNS::InputParameters const &                 fluid_param_in,
                            std::vector<ConvDiff::InputParameters> const & scalar_param_in)
{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  // parameters (fluid + scalar)
  fluid_param = fluid_param_in;
  fluid_param.check_input_parameters(pcout);
  fluid_param.print(pcout, "List of input parameters for fluid solver:");

  scalar_param = scalar_param_in;
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_param[i].check_input_parameters();
    AssertThrow(scalar_param[i].problem_type == ConvDiff::ProblemType::Unsteady,
                ExcMessage("ProblemType must be unsteady!"));

    scalar_param[i].print(pcout,
                          "List of input parameters for scalar quantity " +
                            Utilities::to_string(i) + ":");
  }

  // triangulation
  if(fluid_param.triangulation_type == TriangulationType::Distributed)
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(scalar_param[i].triangulation_type == TriangulationType::Distributed,
                  ExcMessage(
                    "Parameter triangulation_type is different for fluid field and scalar field"));
    }

    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(fluid_param.triangulation_type == TriangulationType::FullyDistributed)
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(scalar_param[i].triangulation_type == TriangulationType::FullyDistributed,
                  ExcMessage(
                    "Parameter triangulation_type is different for fluid field and scalar field"));
    }

    triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  create_grid_and_set_boundary_ids(triangulation, fluid_param.h_refinements, periodic_faces);
  print_grid_data(pcout, fluid_param.h_refinements, *triangulation);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(fluid_param.mapping, fluid_param.degree_u);

  if(fluid_param.ale_formulation) // moving mesh
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(scalar_param[i].ale_formulation == true,
                  ExcMessage(
                    "Parameter ale_formulation is different for fluid field and scalar field"));
    }

    std::shared_ptr<Function<dim>> mesh_motion;
    mesh_motion = set_mesh_movement_function<dim>();
    moving_mesh.reset(new MovingMeshAnalytical<dim, Number>(*triangulation,
                                                            mapping_degree,
                                                            fluid_param.degree_u,
                                                            mpi_comm,
                                                            mesh_motion,
                                                            fluid_param.start_time));

    mesh = moving_mesh;
  }
  else // static mesh
  {
    mesh.reset(new Mesh<dim>(mapping_degree));
  }

  // boundary conditions
  fluid_boundary_descriptor_velocity.reset(new IncNS::BoundaryDescriptorU<dim>());
  fluid_boundary_descriptor_pressure.reset(new IncNS::BoundaryDescriptorP<dim>());

  IncNS::set_boundary_conditions(fluid_boundary_descriptor_velocity,
                                 fluid_boundary_descriptor_pressure);

  // field functions
  fluid_field_functions.reset(new IncNS::FieldFunctions<dim>());
  IncNS::set_field_functions(fluid_field_functions);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    // boundary conditions
    scalar_boundary_descriptor[i].reset(new ConvDiff::BoundaryDescriptor<dim>());

    ConvDiff::set_boundary_conditions(scalar_boundary_descriptor[i], i);

    // field functions
    scalar_field_functions[i].reset(new ConvDiff::FieldFunctions<dim>());
    ConvDiff::set_field_functions(scalar_field_functions[i], i);
  }


  // initialize navier_stokes_operator
  AssertThrow(fluid_param.solver_type == IncNS::SolverType::Unsteady,
              ExcMessage("This is an unsteady solver. Check input parameters."));

  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operator_coupled.reset(new DGCoupled(*triangulation,
                                                       mesh->get_mapping(),
                                                       periodic_faces,
                                                       fluid_boundary_descriptor_velocity,
                                                       fluid_boundary_descriptor_pressure,
                                                       fluid_field_functions,
                                                       fluid_param,
                                                       mpi_comm));

    navier_stokes_operator = navier_stokes_operator_coupled;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operator_dual_splitting.reset(
      new DGDualSplitting(*triangulation,
                          mesh->get_mapping(),
                          periodic_faces,
                          fluid_boundary_descriptor_velocity,
                          fluid_boundary_descriptor_pressure,
                          fluid_field_functions,
                          fluid_param,
                          mpi_comm));

    navier_stokes_operator = navier_stokes_operator_dual_splitting;
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operator_pressure_correction.reset(
      new DGPressureCorrection(*triangulation,
                               mesh->get_mapping(),
                               periodic_faces,
                               fluid_boundary_descriptor_velocity,
                               fluid_boundary_descriptor_pressure,
                               fluid_field_functions,
                               fluid_param,
                               mpi_comm));

    navier_stokes_operator = navier_stokes_operator_pressure_correction;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  // initialize convection-diffusion operator
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    conv_diff_operator[i].reset(new ConvDiff::DGOperator<dim, Number>(*triangulation,
                                                                      mesh->get_mapping(),
                                                                      periodic_faces,
                                                                      scalar_boundary_descriptor[i],
                                                                      scalar_field_functions[i],
                                                                      scalar_param[i],
                                                                      mpi_comm));
  }

  // initialize matrix_free
  matrix_free_wrapper.reset(new MatrixFreeWrapper<dim, Number>(mesh->get_mapping()));
  matrix_free_wrapper->append_data_structures(*navier_stokes_operator, "fluid");
  for(unsigned int i = 0; i < n_scalars; ++i)
    matrix_free_wrapper->append_data_structures(*conv_diff_operator[i],
                                                "scalar" + std::to_string(i));
  matrix_free_wrapper->reinit(fluid_param.use_cell_based_face_loops, triangulation);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    AssertThrow(
      scalar_param[i].use_cell_based_face_loops == fluid_param.use_cell_based_face_loops,
      ExcMessage(
        "Parameter use_cell_based_face_loops should be the same for fluid and scalar transport."));
  }

  // setup Navier-Stokes operator
  if(fluid_param.boussinesq_term)
  {
    // assume that the first scalar field with index 0 is the active scalar that
    // couples to the incompressible Navier-Stokes equations
    navier_stokes_operator->setup(matrix_free_wrapper, conv_diff_operator[0]->get_dof_name());
  }
  else
  {
    navier_stokes_operator->setup(matrix_free_wrapper);
  }

  // setup convection-diffusion operator
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    conv_diff_operator[i]->setup(matrix_free_wrapper,
                                 navier_stokes_operator->get_dof_name_velocity());
  }

  // setup postprocessor
  fluid_postprocessor = IncNS::construct_postprocessor<dim, Number>(fluid_param, mpi_comm);
  fluid_postprocessor->setup(*navier_stokes_operator);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_postprocessor[i] =
      ConvDiff::construct_postprocessor<dim, Number>(scalar_param[i], mpi_comm, i);
    scalar_postprocessor[i]->setup(conv_diff_operator[i]->get_dof_handler(), mesh->get_mapping());
  }

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  if(this->fluid_param.temporal_discretization == IncNS::TemporalDiscretization::BDFCoupledSolution)
  {
    fluid_time_integrator.reset(new TimeIntCoupled(navier_stokes_operator_coupled,
                                                   fluid_param,
                                                   mpi_comm,
                                                   fluid_postprocessor,
                                                   moving_mesh,
                                                   matrix_free_wrapper));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFDualSplittingScheme)
  {
    fluid_time_integrator.reset(new TimeIntDualSplitting(navier_stokes_operator_dual_splitting,
                                                         fluid_param,
                                                         mpi_comm,
                                                         fluid_postprocessor,
                                                         moving_mesh,
                                                         matrix_free_wrapper));
  }
  else if(this->fluid_param.temporal_discretization ==
          IncNS::TemporalDiscretization::BDFPressureCorrection)
  {
    fluid_time_integrator.reset(
      new TimeIntPressureCorrection(navier_stokes_operator_pressure_correction,
                                    fluid_param,
                                    mpi_comm,
                                    fluid_postprocessor,
                                    moving_mesh,
                                    matrix_free_wrapper));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  fluid_time_integrator->setup(fluid_param.restarted_simulation);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    // initialize time integrator
    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      scalar_time_integrator[i].reset(new ConvDiff::TimeIntExplRK<Number>(
        conv_diff_operator[i], scalar_param[i], mpi_comm, scalar_postprocessor[i]));
    }
    else if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      scalar_time_integrator[i].reset(new ConvDiff::TimeIntBDF<dim, Number>(conv_diff_operator[i],
                                                                            scalar_param[i],
                                                                            mpi_comm,
                                                                            scalar_postprocessor[i],
                                                                            moving_mesh,
                                                                            matrix_free_wrapper));
    }
    else
    {
      AssertThrow(scalar_param[i].temporal_discretization ==
                      ConvDiff::TemporalDiscretization::ExplRK ||
                    scalar_param[i].temporal_discretization ==
                      ConvDiff::TemporalDiscretization::BDF,
                  ExcMessage("Specified time integration scheme is not implemented!"));
    }

    if(scalar_param[i].restarted_simulation == false)
    {
      // The parameter start_with_low_order has to be true.
      // This is due to the fact that the setup function of the time integrator initializes
      // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
      // start_with_low_order == false. However, the combined time step size
      // is not known at this point since we have to first communicate the time step size
      // in order to find the minimum time step size. Overwriting the time step size would
      // imply that the time step sizes are non constant for a time integrator, but the
      // time integration constants are initialized based on the assumption of constant
      // time step size. Hence, the easiest way to avoid these kind of
      // inconsistencies is to preclude the case start_with_low_order == false.
      // In case of a restart, start_with_low_order = false is possible since it has been
      // enforced that all previous time steps are identical for fluid and scalar transport.
      AssertThrow(fluid_param.start_with_low_order == true &&
                    scalar_param[i].start_with_low_order == true,
                  ExcMessage("start_with_low_order has to be true for this solver."));
    }

    scalar_time_integrator[i]->setup(scalar_param[i].restarted_simulation);

    // adaptive time stepping
    if(fluid_param.adaptive_time_stepping == true)
    {
      AssertThrow(
        scalar_param[i].adaptive_time_stepping == true,
        ExcMessage(
          "Adaptive time stepping has to be used for both fluid and scalar transport solvers."));

      use_adaptive_time_stepping = true;
    }
  }

  // setup solvers once time integrator has been initialized
  navier_stokes_operator->setup_solvers(
    fluid_time_integrator->get_scaling_factor_time_derivative_term(),
    fluid_time_integrator->get_velocity());

  // setup solvers in case of BDF time integration (solution of linear systems of equations)
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    AssertThrow(scalar_param[i].analytical_velocity_field == false,
                ExcMessage(
                  "An analytical velocity field can not be used for this coupled solver."));

    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> scalar_time_integrator_BDF =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      double const scaling_factor =
        scalar_time_integrator_BDF->get_scaling_factor_time_derivative_term();

      LinearAlgebra::distributed::Vector<Number> vector;
      navier_stokes_operator->initialize_vector_velocity(vector);
      LinearAlgebra::distributed::Vector<Number> const * velocity = &vector;

      conv_diff_operator[i]->setup_solver(scaling_factor, velocity);
    }
    else
    {
      AssertThrow(scalar_param[i].temporal_discretization ==
                    ConvDiff::TemporalDiscretization::ExplRK,
                  ExcMessage("Not implemented."));
    }
  }

  // Boussinesq term
  // assume that the first scalar quantity with index 0 is the active scalar coupled to
  // the incompressible Navier-Stokes equations via the Boussinesq term
  if(fluid_param.boussinesq_term)
    conv_diff_operator[0]->initialize_dof_vector(temperature);

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Problem<dim, Number>::set_start_time() const
{
  // Setup time integrator and get time step size
  double const fluid_time = fluid_time_integrator->get_time();

  double time = fluid_time;

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    double const scalar_time = scalar_time_integrator[i]->get_time();
    time                     = std::min(time, scalar_time);
  }

  // Set the same start time for both solvers

  // fluid
  fluid_time_integrator->reset_time(time);

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_time_integrator[i]->reset_time(time);
  }
}

template<int dim, typename Number>
void
Problem<dim, Number>::synchronize_time_step_size() const
{
  double const EPSILON = 1.e-10;

  // Setup time integrator and get time step size
  double time_step_size_fluid = std::numeric_limits<double>::max();

  // fluid
  if(fluid_time_integrator->get_time() > fluid_param.start_time - EPSILON)
    time_step_size_fluid = fluid_time_integrator->get_time_step_size();

  double time_step_size = time_step_size_fluid;

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    double time_step_size_scalar = std::numeric_limits<double>::max();
    if(scalar_time_integrator[i]->get_time() > scalar_param[i].start_time - EPSILON)
      time_step_size_scalar = scalar_time_integrator[i]->get_time_step_size();

    time_step_size = std::min(time_step_size, time_step_size_scalar);
  }

  if(use_adaptive_time_stepping == false)
  {
    // decrease time_step in order to exactly hit end_time
    time_step_size = (fluid_param.end_time - fluid_param.start_time) /
                     (1 + int((fluid_param.end_time - fluid_param.start_time) / time_step_size));

    pcout << std::endl << "Combined time step size dt = " << time_step_size << std::endl;
  }

  // Set the same time step size for both solvers

  // fluid
  fluid_time_integrator->set_current_time_step_size(time_step_size);

  // scalar transport
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_time_integrator[i]->set_current_time_step_size(time_step_size);
  }
}

template<int dim, typename Number>
void
Problem<dim, Number>::communicate_scalar_to_fluid() const
{
  // We need to communicate between fluid solver and scalar transport solver, i.e., ask the
  // scalar transport solver (scalar 0 by definition) for the temperature and hand it over to the
  // fluid solver.
  if(fluid_param.boussinesq_term)
  {
    // assume that the first scalar quantity with index 0 is the active scalar coupled to
    // the incompressible Navier-Stokes equations via the Boussinesq term
    if(scalar_param[0].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      std::shared_ptr<ConvDiff::TimeIntExplRK<Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntExplRK<Number>>(scalar_time_integrator[0]);
      time_int_scalar->extrapolate_solution(temperature);
    }
    else if(scalar_param[0].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[0]);
      time_int_scalar->extrapolate_solution(temperature);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    navier_stokes_operator->set_temperature(temperature);
  }
}

template<int dim, typename Number>
void
Problem<dim, Number>::communicate_fluid_to_all_scalars() const
{
  // We need to communicate between fluid solver and scalar transport solver, i.e., ask the
  // fluid solver for the velocity field and hand it over to all scalar transport solvers.
  std::vector<LinearAlgebra::distributed::Vector<Number> const *> velocities;
  std::vector<double>                                             times;
  fluid_time_integrator->get_velocities_and_times_np(velocities, times);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      std::shared_ptr<ConvDiff::TimeIntExplRK<Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntExplRK<Number>>(scalar_time_integrator[i]);
      time_int_scalar->set_velocities_and_times(velocities, times);
    }
    else if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      time_int_scalar->set_velocities_and_times(velocities, times);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
Problem<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // time loop
  bool finished = false;
  do
  {
    /*
     * pre solve
     */
    fluid_time_integrator->advance_one_timestep_pre_solve();
    for(unsigned int i = 0; i < n_scalars; ++i)
      scalar_time_integrator[i]->advance_one_timestep_pre_solve();

    /*
     * ALE
     */

    // move the mesh and update dependent data structures
    moving_mesh->move_mesh(fluid_time_integrator->get_next_time());
    matrix_free_wrapper->update_geometry();
    navier_stokes_operator->update_after_mesh_movement();
    for(unsigned int i = 0; i < n_scalars; ++i)
      conv_diff_operator[i]->update_after_mesh_movement();
    fluid_time_integrator->ale_update();
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_bdf =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      time_int_bdf->ale_update();
    }

    /*
     *  solve
     */

    // Communicate scalar -> fluid
    communicate_scalar_to_fluid();

    // fluid: advance one time step
    fluid_time_integrator->advance_one_timestep_solve();

    // Communicate fluid -> all scalars
    communicate_fluid_to_all_scalars();

    // scalar transport: advance one time step
    for(unsigned int i = 0; i < n_scalars; ++i)
      scalar_time_integrator[i]->advance_one_timestep_solve();

    /*
     * post solve
     */
    fluid_time_integrator->advance_one_timestep_post_solve();
    for(unsigned int i = 0; i < n_scalars; ++i)
      scalar_time_integrator[i]->advance_one_timestep_post_solve();

    // Both solvers have already calculated the new, adaptive time step size individually in
    // function advance_one_timestep(). Here, we have to synchronize the time step size.
    if(use_adaptive_time_stepping == true)
      synchronize_time_step_size();

    finished = fluid_time_integrator->finished();
    for(unsigned int i = 0; i < n_scalars; ++i)
      finished = finished && scalar_time_integrator[i]->finished();
  } while(!finished);

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
double
Problem<dim, Number>::analyze_computing_times_fluid(double const overall_time_avg) const
{
  this->pcout << std::endl << "Incompressible Navier-Stokes solver:" << std::endl;

  // wall times
  std::vector<std::string> names;
  std::vector<double>      computing_times;

  if(fluid_param.solver_type == IncNS::SolverType::Unsteady)
  {
    this->fluid_time_integrator->get_wall_times(names, computing_times);
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(computing_times[i], mpi_comm);
    this->pcout << "  " << std::setw(length) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  return sum_of_substeps;
}

template<int dim, typename Number>
void
Problem<dim, Number>::analyze_iterations_fluid() const
{
  this->pcout << std::endl << "Incompressible Navier-Stokes solver:" << std::endl;

  // Iterations
  if(fluid_param.solver_type == IncNS::SolverType::Unsteady)
  {
    std::vector<std::string> names;
    std::vector<double>      iterations;

    this->fluid_time_integrator->get_iterations(names, iterations);

    for(unsigned int i = 0; i < iterations.size(); ++i)
    {
      this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
                  << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                  << std::endl;
    }
  }
}

template<int dim, typename Number>
double
Problem<dim, Number>::analyze_computing_times_transport(double const overall_time_avg) const
{
  double wall_time_summed_over_all_scalars = 0.0;

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    this->pcout << std::endl << "Convection-diffusion solver for scalar " << i << ":" << std::endl;

    // wall times
    std::vector<std::string> names;
    std::vector<double>      computing_times;

    if(scalar_param[i].problem_type == ConvDiff::ProblemType::Unsteady)
    {
      this->scalar_time_integrator[i]->get_wall_times(names, computing_times);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    double sum_of_substeps = 0.0;
    for(unsigned int i = 0; i < computing_times.size(); ++i)
    {
      Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(computing_times[i], mpi_comm);
      this->pcout << "  " << std::setw(length) << std::left << names[i] << std::setprecision(2)
                  << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                  << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                  << data.avg / overall_time_avg * 100 << " %" << std::endl;

      sum_of_substeps += data.avg;
    }

    wall_time_summed_over_all_scalars += sum_of_substeps;
  }

  return wall_time_summed_over_all_scalars;
}

template<int dim, typename Number>
void
Problem<dim, Number>::analyze_iterations_transport() const
{
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    this->pcout << std::endl << "Convection-diffusion solver for scalar " << i << ":" << std::endl;

    // Iterations are only relevant for BDF time integrator
    if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::BDF)
    {
      // Iterations
      if(scalar_param[i].problem_type == ConvDiff::ProblemType::Unsteady)
      {
        std::vector<std::string> names;
        std::vector<double>      iterations;

        std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_integrator_bdf =
          std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
        time_integrator_bdf->get_iterations(names, iterations);

        for(unsigned int i = 0; i < iterations.size(); ++i)
        {
          this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::fixed
                      << std::setprecision(2) << std::right << std::setw(6) << iterations[i]
                      << std::endl;
        }
      }
    }
    else if(scalar_param[i].temporal_discretization == ConvDiff::TemporalDiscretization::ExplRK)
    {
      this->pcout << "  Explicit solver (no systems of equations have to be solved)" << std::endl;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
Problem<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  // Iterations
  this->pcout << std::endl << "Average number of iterations:" << std::endl;

  analyze_iterations_fluid();
  analyze_iterations_transport();

  // Wall times

  this->pcout << std::endl << "Wall times:" << std::endl;
  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(overall_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  double const time_fluid_avg  = analyze_computing_times_fluid(overall_time_avg);
  double const time_scalar_avg = analyze_computing_times_transport(overall_time_avg);

  this->pcout << std::endl;

  Utilities::MPI::MinMaxAvg setup_time_data = Utilities::MPI::min_max_avg(setup_time, mpi_comm);
  double const              setup_time_avg  = setup_time_data.avg;
  this->pcout << "  " << std::setw(length) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  double const other = overall_time_avg - time_fluid_avg - time_scalar_avg - setup_time_avg;
  this->pcout << "  " << std::setw(length) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  this->pcout << std::endl
              << "Computational costs (fluid + transport, including setup + postprocessing):"
              << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl;

  // Throughput in DoFs/s per time step per core
  types::global_dof_index DoFs = this->navier_stokes_operator->get_number_of_dofs();

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    DoFs += this->conv_diff_operator[i]->get_number_of_dofs();
  }

  if(fluid_param.solver_type == IncNS::SolverType::Unsteady)
  {
    unsigned int N_time_steps      = this->fluid_time_integrator->get_number_of_time_steps();
    double const time_per_timestep = overall_time_avg / (double)N_time_steps;
    this->pcout << std::endl
                << "Throughput per time step (fluid + transport, including setup + postprocessing):"
                << std::endl
                << "  Degrees of freedom      = " << DoFs << std::endl
                << "  Wall time               = " << std::scientific << std::setprecision(2)
                << overall_time_avg << " s" << std::endl
                << "  Time steps              = " << std::left << N_time_steps << std::endl
                << "  Wall time per time step = " << std::scientific << std::setprecision(2)
                << time_per_timestep << " s" << std::endl
                << "  Throughput              = " << std::scientific << std::setprecision(2)
                << DoFs / (time_per_timestep * N_mpi_processes) << " DoFs/s/core" << std::endl;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }


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

    IncNS::InputParameters fluid_param;
    set_input_parameters(fluid_param);

    std::vector<ConvDiff::InputParameters> scalar_param;
    scalar_param.resize(N_SCALARS);
    for(unsigned int i = 0; i < N_SCALARS; ++i)
    {
      set_input_parameters(scalar_param[i], i);

      AssertThrow(fluid_param.dim == scalar_param[i].dim, ExcMessage("Invalid parameters!"));
      AssertThrow(fluid_param.h_refinements == scalar_param[i].h_refinements,
                  ExcMessage("Invalid parameters!"));
      AssertThrow(fluid_param.restarted_simulation == scalar_param[i].restarted_simulation,
                  ExcMessage("Invalid parameters!"));
    }

    // setup problem and run simulation
    typedef double                       Number;
    std::shared_ptr<ProblemBase<Number>> problem;

    if(fluid_param.dim == 2)
      problem.reset(new Problem<2, Number>(mpi_comm, N_SCALARS));
    else if(fluid_param.dim == 3)
      problem.reset(new Problem<3, Number>(mpi_comm, N_SCALARS));
    else
      AssertThrow(false, ExcMessage("Only dim=2 and dim=3 implemented."));

    problem->setup(fluid_param, scalar_param);

    problem->solve();

    problem->analyze_computing_times();
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
