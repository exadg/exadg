/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KELVIN_HELMHOLTZ_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KELVIN_HELMHOLTZ_H_

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity(double const DELTA_0, double const U_INF, double const C_N)
    : Function<dim>(dim, 0.0), DELTA_0(DELTA_0), U_INF(U_INF), C_N(C_N)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double const x1 = p[0];
    double const x2 = p[1];
    double const PI = numbers::PI;

    // clang-format off
    double result = 0.0;
    if(component == 0)
    {
      double const dpsidx2 = - C_N*U_INF*std::exp(-std::pow(x2-0.5,2.0)/std::pow(DELTA_0,2.0))*2.0*(x2-0.5)/std::pow(DELTA_0,2.0)
                               *(std::cos(8.0*PI*x1) + std::cos(20.0*PI*x1));

      result = U_INF*std::tanh((2.0*x2-1.0)/DELTA_0) + dpsidx2;
    }
    else if(component == 1)
    {
      double const dpsidx1 = - C_N*U_INF*std::exp(-std::pow(x2-0.5,2.0)/std::pow(DELTA_0,2.0))
                                *PI*(8.0*std::sin(8.0*PI*x1)+20.0*std::sin(20.0*PI*x1));

      result = - dpsidx1;
    }
    // clang-format on

    return result;
  }

private:
  double const DELTA_0, U_INF, C_N;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const Re           = 1.0e4;
  double const L            = 1.0;
  double const DELTA_0      = 1. / 28.;
  double const U_INF        = 1.0;
  double const C_N          = 1.0e-3;
  double const max_velocity = U_INF;
  double const viscosity    = U_INF * DELTA_0 / Re;
  double const T            = DELTA_0 / U_INF;

  double const start_time = 0.0;
  double const end_time   = 400.0 * T;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type             = ProblemType::Unsteady;
    param.equation_type            = EquationType::NavierStokes;
    param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    param.right_hand_side          = false;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.max_velocity                    = max_velocity;
    param.cfl                             = 0.1;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.time_step_size                  = 1.0e-2;
    param.max_number_of_time_steps        = 1e8;
    param.order_time_integrator           = 2; // 1; // 2; // 3;
    param.start_with_low_order            = true;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 200;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-6, 100);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::Multigrid;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-14, 1.e-6);

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-20, 1.e-6, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-10, 1.e-6);

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 200);

    // preconditioning linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = true;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix; // Multigrid;
    param.multigrid_data_velocity_block.smoother_data.smoother =
      MultigridSmoother::Jacobi; // Jacobi; //Chebyshev; //GMRES;
    param.multigrid_data_velocity_block.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi; // PointJacobi; //BlockJacobi;
    param.multigrid_data_velocity_block.smoother_data.iterations        = 5;
    param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree)
  {
    AssertThrow(dim == 2, ExcMessage("This application is only implemented for dim=2."));

    std::vector<unsigned int> repetitions({1, 1});
    Point<dim>                point1(0.0, 0.0), point2(L, L);
    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);

    // periodicity in x-direction
    for(auto cell : triangulation->active_cell_iterators())
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if(std::fabs(cell->face(f)->center()(0) - 0.0) < 1e-12)
          cell->face(f)->set_boundary_id(1);
        if(std::fabs(cell->face(f)->center()(0) - L) < 1e-12)
          cell->face(f)->set_boundary_id(2);
      }
    }

    auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
    GridTools::collect_periodic_faces(*tria, 1, 2, 0, periodic_faces);
    triangulation->add_periodicity(periodic_faces);

    triangulation->refine_global(n_refine_space);

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor_velocity->symmetry_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));

    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
  }


  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(DELTA_0, U_INF, C_N));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output              = this->write_output;
    pp_data.output_data.directory                 = this->output_directory + "vtu/";
    pp_data.output_data.filename                  = this->output_name;
    pp_data.output_data.start_time                = start_time;
    pp_data.output_data.interval_time             = (end_time - start_time) / 200;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.degree                    = degree;

    // kinetic energy
    pp_data.kinetic_energy_data.calculate                  = true;
    pp_data.kinetic_energy_data.calculate_every_time_steps = 1;
    pp_data.kinetic_energy_data.viscosity                  = viscosity;
    pp_data.kinetic_energy_data.directory                  = this->output_directory;
    pp_data.kinetic_energy_data.filename                   = this->output_name;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace IncNS

template<int dim, typename Number>
std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<IncNS::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_KELVIN_HELMHOLTZ_H_ */
