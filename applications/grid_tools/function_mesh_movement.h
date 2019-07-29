#ifndef INCLUDE_FUNCTION_MESH_MOVEMENT_H_
#define INCLUDE_FUNCTION_MESH_MOVEMENT_H_

namespace IncNS
{

template<int dim>
class FunctionMeshMovement : public Function<dim>
{

public:
  FunctionMeshMovement(InputParameters & param_in)
  :Function<dim>(dim, 0.0),
   param(param_in),
   left(param_in.triangulation_left),
   right(param_in.triangulation_right),
   amplitude(param_in.grid_movement_amplitude),
   delta_t(param_in.end_time - param_in.start_time),
   frequency(param_in.grid_movement_frequency),
   sin_t(0.0)
  {
  }


  double
  displacement(const Point<dim>    &point,
               const unsigned int  coordinate_direction = 0) const
  {
    double solution=0;

    if (param.analytical_mesh_movement == AnalyicMeshMovement::SinCosWithBoundaries)
    {
      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(point(1)-left)/width)*sin_t*amplitude;
      else if (coordinate_direction == 1)
        solution = std::sin(2* pi*(point(0)-left)/width)*sin_t*amplitude;
    }
    else if (param.analytical_mesh_movement == AnalyicMeshMovement::SinCosWithBoundariesOnlyX)
    {
      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(point(1)-left)/width)*sin_t*amplitude;
    }
    else if (param.analytical_mesh_movement == AnalyicMeshMovement::SinCosWithBoundariesOnlyY)
    {
      if (coordinate_direction == 1)
        solution = std::sin(2* pi*(point(0)-left)/width)*sin_t*amplitude;
    }
    else if (param.analytical_mesh_movement == AnalyicMeshMovement::InteriorSinCos ||
             param.analytical_mesh_movement == AnalyicMeshMovement::InteriorSinCosWithSinInTime )
    {
      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(point(1)-left)/width)*sin_t*amplitude*(1- std::pow(point(0)/right,2));
      else if (coordinate_direction == 1)
        solution = std::sin(2* pi*(point(0)-left)/width)*sin_t*amplitude*(1- std::pow(point(1)/right,2));
    }
    else if (param.analytical_mesh_movement == AnalyicMeshMovement::InteriorSinCosOnlyX)
    {
      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(point(1)-left)/width)*sin_t*amplitude*(1- std::pow(point(0)/right,2));
    }
    else if (param.analytical_mesh_movement == AnalyicMeshMovement::InteriorSinCosOnlyY)
    {
      if (coordinate_direction == 1)
        solution = std::sin(2* pi*(point(0)-left)/width)*sin_t*amplitude*(1- std::pow(point(1)/right,2));
    }
    else if (param.analytical_mesh_movement == AnalyicMeshMovement::XSquaredWithBoundaries)
    {
      if (coordinate_direction == 0)
        solution = std::pow(point(1),2)* std::pow((right-std::abs(point(1))),2)*sin_t*amplitude;
      else if (coordinate_direction == 1)
        solution = std::pow(point(0),2)* std::pow((right-std::abs(point(0))),2)*sin_t*amplitude;
    }
    else if (param.analytical_mesh_movement == AnalyicMeshMovement::DoubleInteriorSinCos)
    {
      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(point(1)-left)*2/width)*sin_t*amplitude*(1- std::pow(point(0)/right,2));
      else if (coordinate_direction == 1)
        solution = std::sin(2* pi*(point(0)-left)*2/width)*sin_t*amplitude*(1- std::pow(point(1)/right,2));
    }
    else if (param.analytical_mesh_movement == AnalyicMeshMovement::DoubleSinCosWithBoundaries)
    {
      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(point(1)-left)*2/width)*sin_t*amplitude;
      else if (coordinate_direction == 1)
        solution = std::sin(2* pi*(point(0)-left)*2/width)*sin_t*amplitude;
    }

    return solution;
  }

  void
  set_time_displacement(double const t)
  {
    if(param.analytical_mesh_movement == AnalyicMeshMovement::InteriorSinCosWithSinInTime)
      sin_t = std::sin(2*pi*t/T);
    else
      sin_t = std::pow(std::sin(2*pi*t/T),2);


  }

  void
  set_time_velocity(double t) const
  {
    if(param.analytical_mesh_movement == AnalyicMeshMovement::InteriorSinCosWithSinInTime)
      sin_t = std::cos(2*numbers::PI*t/T)*2*numbers::PI/T;
    else
      sin_t = (4*numbers::PI*std::sin(2*numbers::PI*t/T)*std::cos(2*numbers::PI*t/T)/T);
  }

  //Velocity doesnt require multigrid support, hence to be able to use interpolation by deal.II this function is called value()
  double
  value(const Point<dim>    &p,
        const unsigned int  component = 0) const{
    //Since displacements are of shape const*f(t), code duplication can be avoided using f(t)=\partial_t f(t)
    set_time_velocity(this->get_time());
    return displacement(p, component);

  }

private:
  InputParameters param;

  const double left;
  const double right;
  const double amplitude;
  const double delta_t;
  const double frequency;
  const double width = (right-left);
  const double T = delta_t/frequency; //duration of a period
  mutable double sin_t;
  double pi = numbers::PI;
};
}





#endif /*INCLUDE_FUNCTION_MESH_MOVEMENT_H_*/
