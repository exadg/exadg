
#ifndef __indexa_fluid_base_algorithm_h
#define __indexa_fluid_base_algorithm_h

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>

using namespace dealii;


namespace helpers
{
  /**
   * Internal structure that keeps all information about boundary
   * conditions. Necessary to enable different classes to share the boundary
   * conditions.
   */
  template <int dim>
  struct BoundaryDescriptor
  {
    std::map<types::boundary_id,std::shared_ptr<Function<dim> > > dirichlet_conditions_u;
    std::map<types::boundary_id,std::shared_ptr<Function<dim> > > open_conditions_p;
    std::map<types::boundary_id,std::shared_ptr<Function<dim> > > pressure_fix;

    std::set<types::boundary_id> normal_flux;
    std::set<types::boundary_id> symmetry;
    std::set<types::boundary_id> no_slip;

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;
  };
}


/**
 * Base class for incompressible Navier-Stokes problems that implements
 * boundary conditions and the main virtual functions in terms of solution
 * vectors, setup, and time advancement. Most methods are defined in derived
 * classes.
 *
 * Boundary conditions are set at the beginning of the setup() function of the
 * individual fluid solvers using the stored internal boundary information. In
 * order for the program to recognize the conditions, they must be set before
 * the setup() call. The right place to impose the conditions is when the
 * triangulation has been created and boundary indicators are set. There is no
 * need to re-impose boundary conditions when the mesh is refined as boundary
 * indicators do not change (but you need to reset them when the boundary
 * indicator changes, of course).
 *
 * All of the set functions below are only allowed to be called
 * once. Otherwise, the conditions need to be cleared with the clear()
 * function.
 *
 * This class includes the standard set of boundary conditions. It allows
 * derived classes to use the methods here as given or to override them when
 * additional information is necessary.
 */
template <int dim>
class FluidBaseAlgorithm
{
  public:
  /**
   * Constructor.
   */
  FluidBaseAlgorithm(const unsigned int degree_mapping);

  /**
   * Virtual destructor.
   */
  virtual ~FluidBaseAlgorithm();

  /**
   * Setup of problem. Initializes the degrees of freedom and solver-related
   * variables (vectors, matrices, etc.) and interpolates the initial velocity
   * field to the velocity variable.
   */
  virtual void setup_problem (const Function<dim> &initial_velocity_field) = 0;

  /**
   * Performs one complete time step of the problem. Returns the number of
   * accumulated linear iterations (on whatever solver is used) during the
   * time step.
   */
  virtual unsigned int advance_time_step () = 0;

  /**
   * Generic output interface. Allows to write the complete solution field to
   * a vtu file. Derived classes decide which variables need to be written and
   * how often this is about to happen.
   *
   * The optional argument @p n_subdivisions lets the user override the
   * default value (0, taking the velocity degree) the sub-refinement used for
   * representing higher order solutions.
   */
  virtual void output_solution (const std::string output_base_name,
                                const unsigned int n_subdivisions = 0) const = 0;

  /**
   * Deletes all stored boundary descriptions and the body force.
   */
  void clear();

  /*
   * Sets a Dirichlet condition for the fluid velocity on the boundary of the
   * domain with the given id. Note that you can specify time-dependent
   * functions. Remember to use the built-in "get_time()" within the function
   * to access the time and not define your own time variable, otherwise the
   * imposed conditions will not be correct.
   *
   * Prerequisite: The given function must consist of dim components.
   */
  void set_velocity_dirichlet_boundary (const types::boundary_id  boundary_id,
                                        const std::shared_ptr<Function<dim> > &velocity_function);

  /*
   * Sets a pressure condition on the boundary of the domain with the given
   * id. Note that you can specify time-dependent functions. It is only
   * important that you use the built-in "get_time()" within the function to
   * access the time and not define your own time variable.
   *
   * You can only set pressure boundary conditions on boundaries where there
   * is no velocity Dirichlet conditions.
   *
   * Prerequisite: The given function must be scalar.
   */
  void set_open_boundary (const types::boundary_id  boundary_id,
                          const std::shared_ptr<Function<dim> > &pressure_function
                          = std::shared_ptr<Function<dim> >());

  /*
   * Sets a pressure condition on the boundary of the domain with the given
   * id. It forces the flow field to be normal to the boundary, i.e., the
   * tangential component of the flow will be constrained to zero. Note that
   * you can specify time-dependent functions. It is only important that you
   * use the built-in "get_time()" within the function to access the time and
   * not define your own time variable.
   *
   * You can only set this boundary condition on boundaries where no velocity
   * boundary condition and no other "open" boundary condition is set.
   *
   * Prerequisite: The given function must be scalar.
   */
  void set_open_boundary_with_normal_flux (const types::boundary_id  boundary_id,
                                           const std::shared_ptr<Function<dim> > &pressure_function
                                           = std::shared_ptr<Function<dim> >());

  /*
   * Fix one boundary node to a value specified by the given function,
   * evaluating it on the smallest index of pressure on the given boundary
   * id. For specifying a zero function, you can skip the second argument.
   *
   * You can only set this conditions when all boundary conditions are of
   * Dirichlet or periodic type.
   *
   * Prerequisite: The given function must be scalar.
   */
  void fix_pressure_constant (const types::boundary_id  boundary_id,
                              const std::shared_ptr<Function<dim> > &pressure_function
                              = std::shared_ptr<Function<dim> >());

  /*
   * Sets symmetry boundary conditions on the given boundaries. A symmetry
   * condition sets the normal velocity on the boundary to zero but allows
   * tangential velocities. Symmetry boundary conditions can be set on both
   * straight boundaries and curved boundaries.
   */
  void set_symmetry_boundary (const types::boundary_id boundary_id);

  /*
   * Sets no-slip boundary conditions on the given side. This function sets
   * the velocity to zero along the boundary that corresponds to the given
   * boundary indicator.
   */
  void set_no_slip_boundary (const types::boundary_id boundary_id);

  /**
   * Sets face pairs that indicate periodic directions.
   *
   * For using this function on a parallel distributed triangulation, you need
   * to perform the following steps on the triangulation:
   *
   * @code
   * std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
   *     periodic_faces;
   * GridTools::collect_periodic_faces(triangulation, incoming_boundary_id,
   *                                   outgoing_boundary_id, direction, periodic_faces);
   * // possibly other directions you might want to be periodic
   * triangulation.add_periodicity(periodic_faces);
   * @endcode
   *
   * The variable 'periodic_faces' generated by this call is passed to this
   * function.
   */
  void set_periodic_boundaries (std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > &periodic_faces);

  /**
   * Sets a constant body force given as a tensor.
   */
  void set_body_force(const Tensor<1,dim> constant_body_force);

  /**
   * Sets a general function for the body force. This is slower than the other
   * function, so prefer the other one whenever the function is constant.
   */
  void set_body_force(const std::shared_ptr<TensorFunction<1,dim> > body_force);

  /**
   * Returns the body force on a given point.
   */
  bool body_force_is_constant() const
  {
    return body_force.get() == 0;
  }

  /**
   * Returns the body force on a given point.
   */
  Tensor<1,dim> get_body_force(const Point<dim> &p) const
  {
    if (body_force.get())
      return body_force->value(p);
    else
      return constant_body_force;
  }

  /**
   * Sets the fluid viscosity.
   */
  void set_viscosity (const double viscosity);

  /**
   * Returns the viscosity.
   */
  double get_viscosity () const
  {
    return viscosity;
  }

  /**
   * Sets the size of the time step.
   */
  void set_time_step (const double time_step)
  {
    time_step_size = time_step;
  }

  /**
   * Returns the time step size.
   */
  double get_time_step () const
  {
    return time_step_size;
  }

protected:
  /**
   * The data container holding all boundary conditions for use in derived
   * classes.
   */
  std::shared_ptr<helpers::BoundaryDescriptor<dim> > boundary;

  /**
   * Tensor holding constant body forces.
   */
  Tensor<1,dim> constant_body_force;

  /**
   * Function holding the body force.
   */
  std::shared_ptr<TensorFunction<1,dim> > body_force;

  /**
   * The mapping used for representing curved boundaries.
   */
  MappingQGeneric<dim> mapping;

  /**
   * The viscosity of the underlying fluid
   */
  double viscosity;

  /**
   * The chosen time step size
   */
  double time_step_size;
};


#endif // ifndef __indexa_fluid_base_algorithm_h
