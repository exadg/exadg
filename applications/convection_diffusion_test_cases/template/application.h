/*
 * application.h
 *
 *  Created on: 22.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_APPLICATION_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_APPLICATION_H_

// template that shows how to setup a new application

namespace ConvDiff
{
namespace Template
{
//  Example of a user defined function
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

  std::string output_directory = "output/vtu/", output_name = "test";

  void
  set_input_parameters(InputParameters & param)
  {
    (void)param;
    // TODO fill parameters
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    GridGenerator::hyper_cube(*triangulation, -1.0, 1.0);
    triangulation->refine_global(n_refine_space);
  }

  std::shared_ptr<Function<dim>>
  set_mesh_movement_function()
  {
    std::shared_ptr<Function<dim>> mesh_motion;
    mesh_motion.reset(new Functions::ZeroFunction<dim>(dim));

    return mesh_motion;
  }

  void set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
  }


  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(InputParameters const & param, MPI_Comm const & mpi_comm)
  {
    (void)param;

    PostProcessorData<dim>                          pp_data;
    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace Template
} // namespace ConvDiff



#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_TEMPLATE_APPLICATION_H_ */
