/*
 * application_base.h
 *
 *  Created on: 22.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_POISSON_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_description.h>

// functionalities
#include "../../functionalities/parse_input.h"

// user interface
#include "../../convection_diffusion/user_interface/boundary_descriptor.h"
#include "field_functions.h"
#include "input_parameters.h"

// postprocessor
#include "../../convection_diffusion/postprocessor/postprocessor.h"

using namespace dealii;

namespace Poisson
{
template<int dim, typename Number>
class ApplicationBase
{
public:
  virtual void
  add_parameters(ParameterHandler & prm)
  {
    (void)prm;

    // can be overwritten by derived classes and is for example necessary
    // in order to generate a default input file
  }

  ApplicationBase(std::string parameter_file)
    : parameter_file(parameter_file), n_subdivisions_1d_hypercube(1)
  {
  }

  virtual ~ApplicationBase()
  {
  }

  virtual void
  set_input_parameters(InputParameters & parameters) = 0;

  virtual void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces) = 0;

  virtual void set_boundary_conditions(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<0, dim>> boundary_descriptor) = 0;

  virtual void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions) = 0;

  virtual std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm) = 0;

  void
  set_subdivisions_hypercube(unsigned int const n_subdivisions_1d)
  {
    n_subdivisions_1d_hypercube = n_subdivisions_1d;
  }

protected:
  InputParameters param;
  std::string     parameter_file;

  unsigned int n_subdivisions_1d_hypercube;
};

} // namespace Poisson



#endif /* INCLUDE_POISSON_USER_INTERFACE_APPLICATION_BASE_H_ */
