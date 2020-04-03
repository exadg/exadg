/*
 * template.h
 *
 *  Created on: 22.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_

namespace CompNS
{
namespace Template
{
// Example of a user defined function
template<int dim>
class MyFunction : public Function<dim>
{
public:
  MyFunction(const unsigned int n_components = 1, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    (void)p;
    (void)component;

    return 0.0;
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

  std::string output_directory = "output/compressible_flow/manufactured_solution/",
              output_name      = "test";

  void
  set_input_parameters(InputParameters & param)
  {
    (void)param;

    // Here, set all parameters differing from their default values as initialized in
    // CompNS::InputParameters::InputParameters()
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    // to avoid warnings (unused variable) use ...
    (void)triangulation;
    (void)n_refine_space;
    (void)periodic_faces;
  }


  void
  set_boundary_conditions(
    std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_density,
    std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_velocity,
    std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_pressure,
    std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, EnergyBoundaryVariable>         pair_variable;

    // these lines show exemplarily how the boundary descriptors are filled
    boundary_descriptor_density->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_energy->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    // set energy boundary variable
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(0, EnergyBoundaryVariable::Energy));
  }

  void
  set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim>> field_functions)
  {
    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(dim + 2));
    field_functions->right_hand_side_density.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->right_hand_side_energy.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    (void)degree;

    CompNS::PostProcessorData<dim> pp_data;

    // Here, fill postprocessor data

    std::shared_ptr<CompNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace Template
} // namespace CompNS

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEMPLATE_H_ */
