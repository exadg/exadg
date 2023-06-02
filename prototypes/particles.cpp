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

template<int dim>
class VelocityField : public dealii::Function<dim>
{
public:
  VelocityField() : dealii::Function<dim>(dim)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
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
  dealii::Triangulation<dim> tria;
  dealii::GridGenerator::hyper_cube(tria);
  tria.refine_global(3);

  dealii::MappingQ1<dim> mapping;

  // ... and velocity
  dealii::FESystem<dim>   fe{dealii::FE_Q<dim>{1}, dim};
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  dealii::Vector<double> velocity_vector(dof_handler.n_dofs());
  dealii::VectorTools::interpolate(mapping, dof_handler, VelocityField<dim>(), velocity_vector);

  dealii::DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, velocity_vector, "velocity");
  data_out.build_patches();
  std::ofstream output("background.vtu");
  data_out.write_vtu(output);

  // particle handler and initial position of particles
  dealii::Particles::ParticleHandler<dim> particle_handler(tria, mapping);
  std::vector<dealii::Point<dim>>         particle_positions(100);
  for(unsigned int i = 0; i < particle_positions.size(); ++i)
  {
    double const rad = 2 * dealii::numbers::PI * i / particle_positions.size();
    particle_positions[i] =
      dealii::Point<dim>(0.3 + 0.25 * std::cos(rad), 0.5 + 0.25 * std::sin(rad));
  }

  particle_handler.insert_particles(particle_positions);

  // helper function to print partciles
  auto const post = [&]() {
    dealii::Particles::DataOut<dim> data_out;
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
    dealii::Vector<double>              solution_values(fe.n_dofs_per_cell());
    dealii::FEPointEvaluation<dim, dim> evaluator(mapping, fe, dealii::update_values);

    // loop over all cells
    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(particle_handler.n_particles_in_cell(cell) == 0)
        continue; // this cell has not particles

      // collect current refernce position of particles
      std::vector<dealii::Point<dim>> particle_positions;
      for(auto const & particle : particle_handler.particles_in_cell(cell))
        particle_positions.push_back(particle.get_reference_location());

      // compute velocity at these positions
      cell->get_dof_values(velocity_vector, solution_values);
      evaluator.reinit(cell, particle_positions);
      evaluator.evaluate(make_array_view(solution_values), dealii::EvaluationFlags::values);

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
