/*
 * decaying_hill.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DECAYING_HILL_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DECAYING_HILL_H_

namespace ConvDiff
{
namespace DecayingHill
{
enum class BoundaryConditionType
{
  HomogeneousDBC,
  HomogeneousNBC,
  HomogeneousNBCWithRHS
};

BoundaryConditionType const BOUNDARY_TYPE = BoundaryConditionType::HomogeneousDBC;

bool const RIGHT_HAND_SIDE =
  (BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS) ? true : false;

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(const double diffusivity) : Function<dim>(1, 0.0), diffusivity(diffusivity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double t      = this->get_time();
    double result = 1.0;

    if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC)
    {
      for(int d = 0; d < dim; d++)
        result *= std::cos(p[d] * numbers::PI / 2.0);
      result *= std::exp(-0.5 * diffusivity * pow(numbers::PI, 2.0) * t);
    }
    else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC)
    {
      for(int d = 0; d < dim; d++)
        result *= std::cos(p[d] * numbers::PI);
      result *= std::exp(-2.0 * diffusivity * pow(numbers::PI, 2.0) * t);
    }
    else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
    {
      for(int d = 0; d < dim; d++)
        result *= std::cos(p[d] * numbers::PI) + 1.0;
      result *= std::exp(-2.0 * diffusivity * pow(numbers::PI, 2.0) * t);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return result;
  }

private:
  const double diffusivity;
};

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const double diffusivity) : Function<dim>(1, 0.0), diffusivity(diffusivity)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double t      = this->get_time();
    double result = 0.0;

    if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC ||
       BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC)
    {
      // do nothing, rhs=0
    }
    else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
    {
      for(int d = 0; d < dim; ++d)
        result += std::cos(p[d] * numbers::PI) + 1;
      result *= -std::pow(numbers::PI, 2.0) * diffusivity *
                std::exp(-2.0 * diffusivity * pow(numbers::PI, 2.0) * t);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return result;
  }

private:
  const double diffusivity;
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

  double const left = -1.0, right = 1.0;

  double const diffusivity = 1.0e-1;

  double const start_time = 0.0;
  double const end_time   = 1.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.dim             = 2;
    param.problem_type    = ProblemType::Unsteady;
    param.equation_type   = EquationType::Diffusion;
    param.right_hand_side = RIGHT_HAND_SIDE;

    // PHYSICAL QUANTITIES
    param.start_time  = start_time;
    param.end_time    = end_time;
    param.diffusivity = diffusivity;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::BDF;
    param.time_integrator_rk            = TimeIntegratorRK::ExplRK3Stage7Reg2;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    param.order_time_integrator         = 3;
    param.start_with_low_order          = false;
    param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    param.time_step_size                = 1.0e-3;
    param.cfl                           = 0.1;
    param.diffusion_number              = 0.04; // 0.01;

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
    param.solver =
      Solver::CG; // use FGMRES for elementwise iterative block Jacobi type preconditioners
    param.solver_data      = SolverData(1e4, 1.e-20, 1.e-6, 100);
    param.preconditioner   = Preconditioner::InverseMassMatrix; // BlockJacobi; //Multigrid;
    param.mg_operator_type = MultigridOperatorType::ReactionDiffusion;
    param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev; // GMRES;
    param.implement_block_diagonal_preconditioner_matrix_free = true;
    param.solver_block_diagonal                               = Elementwise::Solver::CG;
    param.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
    param.solver_data_block_diagonal    = SolverData(1000, 1.e-12, 1.e-2, 1000);
    param.use_cell_based_face_loops     = true;
    param.update_preconditioner         = false;

    // output of solver information
    param.solver_info_data.interval_time = end_time - start_time;

    // NUMERICAL PARAMETERS
    param.use_combined_operator = true;
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

    triangulation->refine_global(n_refine_space);
  }

  std::shared_ptr<Function<dim>>
  set_mesh_movement_function()
  {
    std::shared_ptr<Function<dim>> mesh_motion;
    mesh_motion.reset(new Functions::ZeroFunction<dim>(dim));

    return mesh_motion;
  }

  void set_boundary_conditions(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<0, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC)
    {
      boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>(diffusivity)));
    }
    else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC ||
            BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
    {
      boundary_descriptor->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  void
  set_field_functions(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Solution<dim>(diffusivity));
    field_functions->right_hand_side.reset(new RightHandSide<dim>(diffusivity));
    field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(ConvDiff::InputParameters const & param, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output         = true;
    pp_data.output_data.output_folder        = output_directory;
    pp_data.output_data.output_name          = output_name;
    pp_data.output_data.output_start_time    = param.start_time;
    pp_data.output_data.output_interval_time = (param.end_time - param.start_time) / 20;
    pp_data.output_data.write_higher_order   = false;
    //  pp_data.output_data.degree = param.degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(diffusivity));
    pp_data.error_data.error_calc_start_time    = param.start_time;
    pp_data.error_data.error_calc_interval_time = (param.end_time - param.start_time) / 20;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace DecayingHill
} // namespace ConvDiff

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DECAYING_HILL_H_ */
