#include <fstream>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

using namespace dealii;

template<int dim>
class VelocityField : public Function<dim>
{
public:
  VelocityField() : Function<dim>(dim)
  {
  }

  virtual double
  value(Point<dim> const & p, unsigned int const component) const
  {
    if(component == 0)
      return p[1];
    return 0;
  }
};

int
main()
{
  int const dim = 2;

  // backgroud mesh ...
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(3);

  MappingQ1<dim> mapping;

  // ... and velocity
  FESystem<dim>   fe{FE_Q<dim>{1}, dim};
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  Vector<double> velocity_vector(dof_handler.n_dofs());
  VectorTools::interpolate(mapping, dof_handler, VelocityField<dim>(), velocity_vector);

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, velocity_vector, "velocity");
  data_out.build_patches();
  std::ofstream output("background.vtu");
  data_out.write_vtu(output);

  // particle handler and initial position of particles
  Particles::ParticleHandler<dim> particle_handler(tria, mapping);
  std::vector<Point<dim>>         particle_positions(100);
  for(unsigned int i = 0; i < particle_positions.size(); ++i)
  {
    double const rad      = 2 * numbers::PI * i / particle_positions.size();
    particle_positions[i] = Point<dim>(0.3 + 0.25 * std::cos(rad), 0.5 + 0.25 * std::sin(rad));
  }

  particle_handler.insert_particles(particle_positions);

  // helper function to print partciles
  auto const post = [&]() {
    Particles::DataOut<dim> data_out;
    data_out.build_patches(particle_handler);
    static int    counter = 0;
    std::ofstream output("particles." + std::to_string(counter++) + ".vtu");
    data_out.write_vtu(output);
  };


  // run dummy time loop
  post(); // print intial configuration

  double const dt = 0.1;
  for(double t = 0.0; t < 0.7; t += dt)
  {
    Vector<double>              solution_values(fe.n_dofs_per_cell());
    FEPointEvaluation<dim, dim> evaluator(mapping, fe, update_values);

    // loop over all cells
    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(particle_handler.n_particles_in_cell(cell) == 0)
        continue; // this cell has not particles

      // collect current refernce position of particles
      std::vector<Point<dim>> particle_positions;
      for(auto const & particle : particle_handler.particles_in_cell(cell))
        particle_positions.push_back(particle.get_reference_location());

      // compute velocity at these positions
      cell->get_dof_values(velocity_vector, solution_values);
      evaluator.reinit(cell, particle_positions);
      evaluator.evaluate(make_array_view(solution_values), EvaluationFlags::values);

      // update position of particles in real space
      unsigned int p = 0;
      for(auto & particle : particle_handler.particles_in_cell(cell))
        particle.set_location(particle.get_location() + dt * evaluator.get_value(p++));
    }

    // sort particles into new cells
    particle_handler.sort_particles_into_subdomains_and_cells();

    post();
  }
}
