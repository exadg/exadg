
#include "fluid_base_algorithm.h"


using namespace dealii;



template<int dim>
FluidBaseAlgorithm<dim>::FluidBaseAlgorithm(const unsigned int mapping_degree)
  : boundary(new helpers::BoundaryDescriptor<dim>()), mapping(mapping_degree), viscosity(1.)
{
}



template<int dim>
FluidBaseAlgorithm<dim>::~FluidBaseAlgorithm()
{
}



template<int dim>
void
FluidBaseAlgorithm<dim>::clear()
{
  boundary.reset(new helpers::BoundaryDescriptor<dim>());
  body_force.reset();
  constant_body_force = Tensor<1, dim>();
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_velocity_dirichlet_boundary(
  const types::boundary_id               boundary_id,
  const std::shared_ptr<Function<dim>> & velocity_function)
{
  if(velocity_function.get() == 0)
    return set_no_slip_boundary(boundary_id);
  AssertThrow(velocity_function->n_components == dim,
              ExcMessage("Velocity boundary function needs to have dim components."));
  boundary->dirichlet_conditions_u[boundary_id] = velocity_function;
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_open_boundary(const types::boundary_id               boundary_id,
                                           const std::shared_ptr<Function<dim>> & pressure_function)
{
  if(pressure_function.get() == 0)
    boundary->open_conditions_p[boundary_id] =
      std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(1));
  else
  {
    AssertThrow(pressure_function->n_components == 1,
                ExcMessage("Pressure boundary function needs to be scalar."));
    boundary->open_conditions_p[boundary_id] = pressure_function;
  }
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_open_boundary_with_normal_flux(
  const types::boundary_id               boundary_id,
  const std::shared_ptr<Function<dim>> & pressure_function)
{
  if(pressure_function.get() == 0)
    boundary->open_conditions_p[boundary_id] =
      std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(1));
  else
  {
    AssertThrow(pressure_function->n_components == 1,
                ExcMessage("Pressure boundary function needs to be scalar."));
    boundary->open_conditions_p[boundary_id] = pressure_function;
  }
  boundary->normal_flux.insert(boundary_id);
}



template<int dim>
void
FluidBaseAlgorithm<dim>::fix_pressure_constant(
  const types::boundary_id               boundary_id,
  const std::shared_ptr<Function<dim>> & pressure_function)
{
  AssertThrow(pressure_function.get() == 0 || pressure_function->n_components == 1,
              ExcMessage("Pressure boundary function needs to be scalar."));
  boundary->pressure_fix[boundary_id] = pressure_function;
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_symmetry_boundary(const types::boundary_id boundary_id)
{
  boundary->symmetry.insert(boundary_id);
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_no_slip_boundary(const types::boundary_id boundary_id)
{
  boundary->no_slip.insert(boundary_id);
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_periodic_boundaries(
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
    periodic_faces)
{
  boundary->periodic_face_pairs_level0 = periodic_faces;
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_body_force(const Tensor<1, dim> constant_body_force)
{
  this->constant_body_force = constant_body_force;
  this->body_force.reset();
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_body_force(const std::shared_ptr<TensorFunction<1, dim>> body_force)
{
  this->body_force          = body_force;
  this->constant_body_force = Tensor<1, dim>();
}



template<int dim>
void
FluidBaseAlgorithm<dim>::set_viscosity(const double viscosity)
{
  this->viscosity = viscosity;
}


template class FluidBaseAlgorithm<2>;
template class FluidBaseAlgorithm<3>;
