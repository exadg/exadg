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

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COUETTE_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COUETTE_H_

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

// problem specific parameters
double const DYN_VISCOSITY = 1.0e-2;
double const GAMMA         = 1.4;
double const LAMBDA        = 0.0262;
double const R             = 287.058;
double const U_0           = 1.0;
double const PRESSURE      = 1.0e5;
double const GAS_CONSTANT  = 287.058;
double const T_0           = 273.0;
double const RHO_0         = PRESSURE / (R * T_0);

double const H = 3.0;
double const L = 2.0 * H;

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(unsigned int const n_components = dim + 2, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double const T =
      T_0 + DYN_VISCOSITY * U_0 * U_0 / (2.0 * LAMBDA) * (p[1] * p[1] / (H * H) - 1.0);
    double const rho = PRESSURE / (R * T);
    double const u   = U_0 / H * p[1];

    double result = 0.0;

    if(component == 0)
      result = rho;
    else if(component == 1)
      result = rho * u;
    else if(component == 2)
      result = 0.0;
    else if(component == 1 + dim)
      result = rho * (0.5 * u * u + R * T / (GAMMA - 1.0));

    return result;
  }
};

template<int dim>
class VelocityBC : public Function<dim>
{
public:
  VelocityBC(unsigned int const n_components = dim, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;

    if(component == 0)
      result = U_0 * p[1] / H;
    else if(component == 1)
      result = 0.0;

    return result;
  }
};

template<int dim>
class DensityBC : public Function<dim>
{
public:
  DensityBC(double const time = 0.) : Function<dim>(1, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    (void)component;

    double const T =
      T_0 + DYN_VISCOSITY * U_0 * U_0 / (2.0 * LAMBDA) * (p[1] * p[1] / (H * H) - 1.0);

    double const rho = PRESSURE / (R * T);

    return rho;
  }
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

  double const start_time = 0.0;
  double const end_time   = 10.0;

  void
  set_input_parameters(InputParameters & param) final
  {
    // MATHEMATICAL MODEL
    param.equation_type   = EquationType::NavierStokes;
    param.right_hand_side = true;

    // PHYSICAL QUANTITIES
    param.start_time            = start_time;
    param.end_time              = end_time;
    param.dynamic_viscosity     = DYN_VISCOSITY;
    param.reference_density     = RHO_0;
    param.heat_capacity_ratio   = GAMMA;
    param.thermal_conductivity  = LAMBDA;
    param.specific_gas_constant = R;
    param.max_temperature       = T_0;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::ExplRK;
    param.order_time_integrator         = 2;
    param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
    param.time_step_size                = 1.0e-3;
    param.max_velocity                  = U_0;
    param.cfl_number                    = 0.1;
    param.diffusion_number              = 0.01;
    param.exponent_fe_degree_cfl        = 2.0;
    param.exponent_fe_degree_viscous    = 4.0;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    param.triangulation_type    = TriangulationType::Distributed;
    param.mapping               = MappingType::Isoparametric;
    param.n_q_points_convective = QuadratureRule::Standard;
    param.n_q_points_viscous    = QuadratureRule::Standard;

    // viscous term
    param.IP_factor = 1.0;

    // NUMERICAL PARAMETERS
    param.use_combined_operator = false;
  }

  std::shared_ptr<Grid<dim>>
  create_grid(GridData const & data, MPI_Comm const & mpi_comm) final
  {
    std::shared_ptr<Grid<dim>> grid = std::make_shared<Grid<dim>>(data, mpi_comm);

    std::vector<unsigned int> repetitions({2, 1});
    Point<dim>                point1(0.0, 0.0), point2(L, H);
    GridGenerator::subdivided_hyper_rectangle(*grid->triangulation, repetitions, point1, point2);

    // indicator
    // fixed wall = 0
    // moving wall = 1
    /*
     *             indicator = 1
     *   ___________________________________
     *   |             --->                 |
     *   |                                  |
     *   | <---- periodic B.C.  --------->  |
     *   |                                  |
     *   |                                  |
     *   |__________________________________|
     *             indicator = 0
     */
    typename Triangulation<dim>::cell_iterator cell = grid->triangulation->begin(),
                                               endc = grid->triangulation->end();
    for(; cell != endc; ++cell)
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if(std::fabs(cell->face(face_number)->center()(1) - point2[1]) < 1e-12)
        {
          cell->face(face_number)->set_boundary_id(1);
        }
        else if(std::fabs(cell->face(face_number)->center()(1) - 0.0) < 1e-12)
        {
          cell->face(face_number)->set_boundary_id(0);
        }
        else if(std::fabs(cell->face(face_number)->center()(0) - 0.0) < 1e-12)
        {
          cell->face(face_number)->set_boundary_id(0 + 10);
        }
        else if(std::fabs(cell->face(face_number)->center()(0) - point2[0]) < 1e-12)
        {
          cell->face(face_number)->set_boundary_id(1 + 10);
        }
      }
    }

    auto tria = dynamic_cast<Triangulation<dim> *>(&*grid->triangulation);
    GridTools::collect_periodic_faces(*tria, 0 + 10, 1 + 10, 0, grid->periodic_faces);
    grid->triangulation->add_periodicity(grid->periodic_faces);

    grid->triangulation->refine_global(data.n_refine_global);

    return grid;
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_density,
                          std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_pressure,
                          std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, EnergyBoundaryVariable>         pair_variable;

    // density
    boundary_descriptor_density->dirichlet_bc.insert(pair(0, new DensityBC<dim>()));
    boundary_descriptor_density->dirichlet_bc.insert(pair(1, new DensityBC<dim>()));
    //  boundary_descriptor_density->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(1)));
    //  boundary_descriptor_density->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(1)));

    // velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new VelocityBC<dim>()));
    boundary_descriptor_velocity->dirichlet_bc.insert(pair(1, new VelocityBC<dim>()));

    // pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));

    // energy: prescribe temperature
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(0, EnergyBoundaryVariable::Temperature));
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(1, EnergyBoundaryVariable::Temperature));

    boundary_descriptor_energy->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_energy->dirichlet_bc.insert(
      pair(1, new Functions::ConstantFunction<dim>(T_0, 1)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions) final
  {
    field_functions->initial_solution.reset(new Solution<dim>());
    field_functions->right_hand_side_density.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->right_hand_side_energy.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output      = this->write_output;
    pp_data.output_data.directory         = this->output_directory + "vtu/";
    pp_data.output_data.filename          = this->output_name;
    pp_data.output_data.write_pressure    = true;
    pp_data.output_data.write_velocity    = true;
    pp_data.output_data.write_temperature = true;
    pp_data.output_data.write_vorticity   = true;
    pp_data.output_data.write_divergence  = true;
    pp_data.output_data.start_time        = start_time;
    pp_data.output_data.interval_time     = (end_time - start_time) / 10;
    pp_data.output_data.degree            = degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());
    pp_data.error_data.error_calc_start_time    = start_time;
    pp_data.error_data.error_calc_interval_time = (end_time - start_time) / 10;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace CompNS

template<int dim, typename Number>
std::shared_ptr<CompNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<CompNS::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COUETTE_H_ */
