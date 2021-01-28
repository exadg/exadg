/*
 * template.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef STRUCTURE_TEMPLATE
#define STRUCTURE_TEMPLATE

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      // add parameters
    prm.leave_subsection();
    // clang-format on
  }

  void
  set_input_parameters(InputParameters & parameters)
  {
    (void)parameters;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)triangulation;
    (void)n_refine_space;
    (void)periodic_faces;
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
  {
    (void)boundary_descriptor;
  }

  void
  set_material(MaterialDescriptor & material_descriptor)
  {
    (void)material_descriptor;
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    (void)field_functions;
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    (void)degree;

    PostProcessorData<dim> pp_data;

    std::shared_ptr<PostProcessor<dim, Number>> post(
      new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return post;
  }
};

} // namespace Structure

template<int dim, typename Number>
std::shared_ptr<Structure::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<Structure::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif
