/*
 * template.h
 */

#ifndef APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

//  Example for a user defined function
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
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  set_input_parameters(InputParameters & param)
  {
    (void)param;

    // Here, set all parameters differing from their default values as initialized in
    // Poisson::InputParameters::InputParameters()
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


  void set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // these lines show exemplarily how the boundary descriptors are filled
    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }


  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    // these lines show exemplarily how the field functions are filled
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    (void)degree;

    PostProcessorData<dim> pp_data;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace Poisson

template<int dim, typename Number>
std::shared_ptr<Poisson::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<Poisson::Application<dim, Number>>(input_file);
}

} // namespace ExaDG

#endif /* APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_ */
