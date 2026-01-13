/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef EXADG_POSTPROCESSOR_PARTICLE_CALCULATION_H_
#define EXADG_POSTPROCESSOR_PARTICLE_CALCULATION_H_

// deal.II
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/particles/particle_handler.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/postprocessor/time_control.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
struct ParticleData
{
  ParticleData() : directory("output/"), filename("particle")
  {
  }

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(time_control_data.is_active)
    {
      pcout << std::endl << "  Trace particles:" << std::endl;

      // only implemented for unsteady problem
      time_control_data.print(pcout, true /*unsteady*/);

      print_parameter(pcout, "Directory of output files", directory);
      print_parameter(pcout, "Filename", filename);
    }
  }

  TimeControlData time_control_data;

  // directory and filename
  std::string directory;
  std::string filename;

  // Starting points of particles
  std::vector<dealii::Point<2>> starting_points_2d;
  std::vector<dealii::Point<3>> starting_points_3d;
};

template<int dim, typename Number>
class ParticleCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  ParticleCalculator(MPI_Comm const & comm);

  void
  setup(dealii::DoFHandler<dim> const & dof_handler_in,
        dealii::Mapping<dim> const &    mapping_in,
        ParticleData const &            particle_data_in);

  void
  evaluate(VectorType const & velocity, double const time, bool const print_output);

  TimeControl time_control;

protected:
  void
  track_lost_particle(const typename dealii::Particles::ParticleIterator<dim> &         particle,
                      const typename dealii::Triangulation<dim>::active_cell_iterator & cell);


  MPI_Comm const mpi_comm;

  dealii::ObserverPointer<dealii::DoFHandler<dim> const> dof_handler;
  dealii::ObserverPointer<dealii::Mapping<dim> const>    mapping;

  ParticleData data;
  double       old_time;

  unsigned int                                       lost_paticles;
  std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_boxes;

  dealii::Vector<Number>                  solution_values;
  dealii::Particles::ParticleHandler<dim> particle_handler;
};

} // namespace ExaDG

#endif /* EXADG_POSTPROCESSOR_PARTICLE_CALCULATION_H_ */
