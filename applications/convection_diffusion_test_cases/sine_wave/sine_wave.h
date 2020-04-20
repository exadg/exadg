/*
 * sine_wave.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_

// test case for a purely convective problem
// sine wave that is advected from left to right by a constant velocity field

#include "../../grid_tools/mesh_movement_functions.h"

namespace ConvDiff
{
namespace SineWave
{
template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(const unsigned int n_components = 1, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double t = this->get_time();

    double result = std::sin(numbers::PI * (p[0] - t));

    return result;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application() : ApplicationBase<dim, Number>("")
  {
  }

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    parse_input(input_file, prm, true, true);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("OutputDirectory",  output_directory, "Directory where output is written.");
      prm.add_parameter("OutputName",       output_name,      "Name of output files.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string output_directory = "output/vtu/", output_name = "test";

  double const start_time = 0.0;
  double const end_time   = 8.0;

  double const left  = -1.0;
  double const right = +1.0;

  bool const ale = false;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::Convection;
    param.analytical_velocity_field   = true;
    param.right_hand_side             = false;
    param.ale_formulation             = ale;
    param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;

    // PHYSICAL QUANTITIES
    param.start_time  = start_time;
    param.end_time    = end_time;
    param.diffusivity = 0.0;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization = TemporalDiscretization::ExplRK;
    param.time_integrator_rk      = TimeIntegratorRK::ExplRK3Stage7Reg2;
    param.treatment_of_convective_term =
      TreatmentOfConvectiveTerm::Explicit; // ExplicitOIF; //Explicit;
    param.time_integrator_oif           = TimeIntegratorRK::ExplRK3Stage7Reg2;
    param.adaptive_time_stepping        = true;
    param.order_time_integrator         = 2;
    param.start_with_low_order          = false;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.time_step_size                = 1.0e-1;
    param.cfl                           = 0.2;
    param.cfl_oif                       = param.cfl / 1.0;
    param.diffusion_number              = 0.01;

    // SPATIAL DISCRETIZATION

    // triangulation
    param.triangulation_type = TriangulationType::Distributed;

    // polynomial degree
    param.mapping = MappingType::Affine;

    // convective term
    param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    param.IP_factor = 1.0;

    // SOLVER
    param.solver         = Solver::GMRES;
    param.solver_data    = SolverData(1e4, 1.e-20, 1.e-6, 100);
    param.preconditioner = Preconditioner::InverseMassMatrix;
    // use default parameters of multigrid preconditioner

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 20;

    // NUMERICAL PARAMETERS
    param.use_combined_operator                   = true;
    param.store_analytical_velocity_in_dof_vector = false;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    // hypercube volume is [left,right]^dim
    GridGenerator::hyper_cube(*triangulation, left, right);

    // set boundary id of 1 at right boundary (outflow)
    for(auto cell : *triangulation)
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if((std::fabs(cell.face(face_number)->center()(0) - right) < 1e-12))
          cell.face(face_number)->set_boundary_id(1);
      }
    }
    triangulation->refine_global(n_refine_space);
  }

  std::shared_ptr<Function<dim>>
  set_mesh_movement_function() override
  {
    std::shared_ptr<Function<dim>> mesh_motion;

    MeshMovementData<dim> data;
    data.temporal                       = MeshMovementAdvanceInTime::Sin;
    data.shape                          = MeshMovementShape::SineZeroAtBoundary; // SineAligned;
    data.dimensions[0]                  = std::abs(right - left);
    data.dimensions[1]                  = std::abs(right - left);
    data.amplitude                      = 0.08 * (right - left); // A_max = (right-left)/(2*pi)
    data.period                         = end_time;
    data.t_start                        = 0.0;
    data.t_end                          = end_time;
    data.spatial_number_of_oscillations = 1.0;
    mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));

    return mesh_motion;
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));
    boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }


  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Solution<dim>());
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    std::vector<double> velocity = std::vector<double>(dim, 0.0);
    velocity[0]                  = 1.0;
    field_functions->velocity.reset(new Functions::ConstantFunction<dim>(velocity));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output         = true;
    pp_data.output_data.output_folder        = output_directory;
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 100;
    pp_data.output_data.degree               = degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(1));
    pp_data.error_data.error_calc_start_time    = start_time;
    pp_data.error_data.error_calc_interval_time = (end_time - start_time) / 20;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace SineWave
} // namespace ConvDiff

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_ */
