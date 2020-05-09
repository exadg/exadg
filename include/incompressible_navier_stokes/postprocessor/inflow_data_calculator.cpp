/*
 * inflow_data_calculator.cpp
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

#include "inflow_data_calculator.h"

#include "../../functions_and_boundary_conditions/linear_interpolation.h"
#include "../../postprocessor/evaluate_solution_in_given_point.h"

template<int dim, typename Number>
InflowDataCalculator<dim, Number>::InflowDataCalculator(InflowData<dim> const & inflow_data_in,
                                                        MPI_Comm const &        comm)
  : inflow_data(inflow_data_in), inflow_data_has_been_initialized(false), mpi_comm(comm)
{
}

template<int dim, typename Number>
void
InflowDataCalculator<dim, Number>::setup(DoFHandler<dim> const & dof_handler_velocity_in,
                                         Mapping<dim> const &    mapping_in)
{
  dof_handler_velocity = &dof_handler_velocity_in;
  mapping              = &mapping_in;

  array_dof_indices_and_shape_values.resize(inflow_data.n_points_y * inflow_data.n_points_z);
  array_counter.resize(inflow_data.n_points_y * inflow_data.n_points_z);
}

template<int dim, typename Number>
void
InflowDataCalculator<dim, Number>::calculate(
  LinearAlgebra::distributed::Vector<Number> const & velocity)
{
  if(inflow_data.write_inflow_data == true)
  {
    // initial data: do this expensive step only once at the beginning of the simulation
    if(inflow_data_has_been_initialized == false)
    {
      for(unsigned int iy = 0; iy < inflow_data.n_points_y; ++iy)
      {
        for(unsigned int iz = 0; iz < inflow_data.n_points_z; ++iz)
        {
          Point<dim> point;

          if(inflow_data.inflow_geometry == InflowGeometry::Cartesian)
          {
            AssertThrow(inflow_data.normal_direction == 0, ExcMessage("Not implemented."));

            point = Point<dim>(inflow_data.normal_coordinate,
                               (*inflow_data.y_values)[iy],
                               (*inflow_data.z_values)[iz]);
          }
          else if(inflow_data.inflow_geometry == InflowGeometry::Cylindrical)
          {
            AssertThrow(inflow_data.normal_direction == 2, ExcMessage("Not implemented."));

            Number const x = (*inflow_data.y_values)[iy] * std::cos((*inflow_data.z_values)[iz]);
            Number const y = (*inflow_data.y_values)[iy] * std::sin((*inflow_data.z_values)[iz]);
            point          = Point<dim>(x, y, inflow_data.normal_coordinate);
          }
          else
          {
            AssertThrow(false, ExcMessage("Not implemented."));
          }

          std::vector<std::pair<std::vector<types::global_dof_index>, std::vector<Number>>>
            dof_indices_and_shape_values;
          get_dof_indices_and_shape_values(
            *dof_handler_velocity, *mapping, velocity, point, dof_indices_and_shape_values);

          unsigned int array_index = iy * inflow_data.n_points_z + iz;

          array_dof_indices_and_shape_values[array_index] = dof_indices_and_shape_values;
        }
      }

      inflow_data_has_been_initialized = true;
    }

    // evaluate velocity in all points of the 2d grid
    for(unsigned int iy = 0; iy < inflow_data.n_points_y; ++iy)
    {
      for(unsigned int iz = 0; iz < inflow_data.n_points_z; ++iz)
      {
        // determine the array index, will be needed several times below
        unsigned int array_index = iy * inflow_data.n_points_z + iz;

        // initialize with zeros since we accumulate into these variables
        (*inflow_data.array)[array_index] = 0.0;
        array_counter[array_index]        = 0;

        auto & vector(array_dof_indices_and_shape_values[array_index]);

        // loop over all adjacent, locally owned cells for the current point
        for(auto iter = vector.begin(); iter != vector.end(); ++iter)
        {
          // increment counter (because this is a locally owned cell)
          array_counter[array_index] += 1;

          // interpolate solution using the precomputed shape values and the global dof index
          Tensor<1, dim, Number> velocity_value = Interpolator<1, dim, Number>::value(
            *dof_handler_velocity, velocity, iter->first, iter->second);

          // add result to array with velocity inflow data
          (*inflow_data.array)[array_index] += velocity_value;
        }
      }
    }

    // sum over all processors
    Utilities::MPI::sum(array_counter, mpi_comm, array_counter);
    Utilities::MPI::sum(
      ArrayView<const double>(&(*inflow_data.array)[0][0], dim * inflow_data.array->size()),
      mpi_comm,
      ArrayView<double>(&(*inflow_data.array)[0][0], dim * inflow_data.array->size()));

    // divide by counter in order to get the mean value (averaged over all
    // adjacent cells for a given point)
    for(unsigned int iy = 0; iy < inflow_data.n_points_y; ++iy)
    {
      for(unsigned int iz = 0; iz < inflow_data.n_points_z; ++iz)
      {
        unsigned int array_index = iy * inflow_data.n_points_z + iz;
        if(array_counter[array_index] >= 1)
          (*inflow_data.array)[array_index] /= Number(array_counter[array_index]);
      }
    }
  }
}

template class InflowDataCalculator<2, float>;
template class InflowDataCalculator<2, double>;

template class InflowDataCalculator<3, float>;
template class InflowDataCalculator<3, double>;
