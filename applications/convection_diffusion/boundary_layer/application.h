/*
 * boundary_layer.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_H_

// prescribe value of solution at left and right boundary
// Neumann boundaries at upper and lower boundary
// use constant advection velocity from left to right -> boundary layer

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

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
    double phi_l = 1.0, phi_r = 0.0;
    double U = 1.0, L = 2.0;
    double Pe = U * L / diffusivity;

    double result = phi_l + (phi_r - phi_l) * (std::exp(Pe * p[0] / L) - std::exp(-Pe / 2.0)) /
                              (std::exp(Pe / 2.0) - std::exp(-Pe / 2.0));

    return result;
  }

private:
  double const diffusivity;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const left = -1.0, right = 1.0;

  double const diffusivity = 1.0e-1;

  double const start_time = 0.0;
  double const end_time   = 1.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type              = ProblemType::Steady;
    param.equation_type             = EquationType::ConvectionDiffusion;
    param.right_hand_side           = false;
    param.analytical_velocity_field = true;

    // PHYSICAL QUANTITIES
    param.start_time  = start_time;
    param.end_time    = end_time;
    param.diffusivity = diffusivity;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::BDF;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    param.order_time_integrator         = 2;
    param.start_with_low_order          = true;
    param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    param.time_step_size                = 1.0e-1;
    param.cfl                           = 0.2;
    param.diffusion_number              = 0.01;

    // SPATIAL DISCRETIZATION

    // triangulation
    param.triangulation_type = TriangulationType::Distributed;

    // mapping
    param.mapping = MappingType::Affine;

    // convective term
    param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    param.IP_factor = 1.0;

    // SOLVER
    param.use_cell_based_face_loops = true;
    param.solver                    = Solver::GMRES;
    param.solver_data               = SolverData(1e4, 1.e-20, 1.e-8, 100);
    param.preconditioner            = Preconditioner::Multigrid; // PointJacobi;
    param.mg_operator_type          = MultigridOperatorType::ReactionConvectionDiffusion;
    param.multigrid_data.type       = MultigridType::phMG;
    // MG smoother
    param.multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi; // Chebyshev;
    // MG smoother data
    param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
    param.multigrid_data.smoother_data.iterations     = 5;

    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    param.update_preconditioner = false;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 20;

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

    // set boundary indicator
    for(auto cell : *triangulation)
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if((std::fabs(cell.face(face_number)->center()(1) - left) < 1e-12) ||
           (std::fabs(cell.face(face_number)->center()(1) - right) < 1e-12) ||
           ((dim == 3) && ((std::fabs(cell.face(face_number)->center()(2) - left) < 1e-12) ||
                           (std::fabs(cell.face(face_number)->center()(2) - right) < 1e-12))))
          cell.face(face_number)->set_boundary_id(1);

        // TODO Neumann boundary condition at right boundary
        //      if (std::fabs(cell.face(face_number)->center()(0) - right) < 1e-12)
        //        cell->face(face_number)->set_boundary_id (2);
      }
    }

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>(diffusivity)));
    boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor->neumann_bc.insert(pair(2, new Functions::ConstantFunction<dim>(-10.0, 1)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    std::vector<double> velocity = std::vector<double>(dim, 0.0);
    velocity[0]                  = 1.0;
    field_functions->velocity.reset(new Functions::ConstantFunction<dim>(velocity));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.degree               = degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(diffusivity));
    pp_data.error_data.error_calc_start_time    = start_time;
    pp_data.error_data.error_calc_interval_time = (end_time - start_time) / 20;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace ConvDiff

template<int dim, typename Number>
std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>>(
    new ConvDiff::Application<dim, Number>(input_file));
}

} // namespace ExaDG


#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_H_ */
