
#ifndef __indexa_fe_navierstokes_evaluator_h
#define __indexa_fe_navierstokes_evaluator_h

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <fluid_base_algorithm.h>

namespace helpers
{
  // forward declaration
  template <int dim> class FENavierStokesEvaluator;
}

using namespace dealii;


// Implementation of evaluation class for Navier-Stokes projection solver
// based on continuous finite elements
template <int dim>
class FENavierStokesEvaluator
{
public:
  // Constructor. Takes the MatrixFree object, the pressure variables that are
  // not part of the input in the various evaluator functions, and the time
  // step size for passing them on to the evaluators
  FENavierStokesEvaluator(const MatrixFree<dim> &matrix_free,
                          const parallel::distributed::Vector<double> &pressure,
                          const parallel::distributed::Vector<double> &last_pressure_update,
                          const FluidBaseAlgorithm<dim> & fluid_algorithm);

  // Computes the right hand integrals for the advection step. The inverse
  // mass matrix and the boundary conditions are to be set outside of this
  // function
  void advection_integrals(const parallel::distributed::Vector<double> &src,
                           parallel::distributed::Vector<double> &dst) const;

  // Computes the right hand side for the pressure Poisson equation
  void divergence_integrals(const parallel::distributed::Vector<double> &src,
                            parallel::distributed::Vector<double> &dst) const;

  // Computes the integral part for the curl of the velocity (for output)
  void curl_integrals(const parallel::distributed::Vector<double> &src,
                      parallel::distributed::Vector<double> &dst) const;

private:
  std_cxx11::shared_ptr<const helpers::FENavierStokesEvaluator<dim> > evaluator;
};


#endif
