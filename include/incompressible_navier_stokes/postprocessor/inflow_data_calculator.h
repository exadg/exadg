/*
 * inflow_data_calculator.h
 *
 *  Created on: Nov 17, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_

template<int dim>
double
linear_interpolation_2d_cartesian(Point<dim> const &                          p,
                                  std::vector<double> const &                 y_values,
                                  std::vector<double> const &                 z_values,
                                  std::vector<Tensor<1, dim, double>> const & solution_values,
                                  unsigned int const &                        component)
{
  AssertThrow(dim == 3, ExcMessage("not implemented"));

  double result = 0.0;

  double const tol = 1.e-2;

  unsigned int const n_points_y = y_values.size();
  unsigned int const n_points_z = z_values.size();

  AssertThrow((y_values[0] - tol < p[1]) && (p[1] < y_values[n_points_y - 1] + tol) &&
                (z_values[0] - tol < p[2]) && (p[2] < z_values[n_points_z - 1] + tol),
              ExcMessage("invalid point found."));

  // interpolate y and z-coordinates
  unsigned int iy = 0, iz = 0;

  iy = std::distance(y_values.begin(), std::lower_bound(y_values.begin(), y_values.end(), p[1]));
  iz = std::distance(z_values.begin(), std::lower_bound(z_values.begin(), z_values.end(), p[2]));
  // make sure that the index does not exceed the array bounds in case of round-off errors
  if(iy == y_values.size())
    iy--;
  if(iz == z_values.size())
    iz--;

  double const weight_yp = (p[1] - y_values[iy - 1]) / (y_values[iy] - y_values[iy - 1]);
  double const weight_ym = 1 - weight_yp;
  double const weight_zp = (p[2] - z_values[iz - 1]) / (z_values[iz] - z_values[iz - 1]);
  double const weight_zm = 1 - weight_zp;

  AssertThrow(-1.e-12 < weight_yp && weight_yp < 1. + 1e-12 && -1.e-12 < weight_ym &&
                weight_ym < 1. + 1e-12 && -1.e-12 < weight_zp && weight_zp < 1. + 1e-12 &&
                -1.e-12 < weight_zm && weight_zm < 1. + 1e-12,
              ExcMessage("invalid weights when interpolating solution in 2D."));

  result = weight_ym * weight_zm * solution_values[(iy - 1) * n_points_y + (iz - 1)][component] +
           weight_ym * weight_zp * solution_values[(iy - 1) * n_points_y + (iz)][component] +
           weight_yp * weight_zm * solution_values[(iy)*n_points_y + (iz - 1)][component] +
           weight_yp * weight_zp * solution_values[(iy)*n_points_y + (iz)][component];

  return result;
}

/*
 *  2D interpolation for cylindrical cross-sections
 */
template<int dim>
double
linear_interpolation_2d_cylindrical(double const                                r_in,
                                    double const                                phi,
                                    std::vector<double> const &                 r_values,
                                    std::vector<double> const &                 phi_values,
                                    std::vector<Tensor<1, dim, double>> const & solution_values,
                                    unsigned int const &                        component)
{
  AssertThrow(dim == 3, ExcMessage("not implemented"));

  double result = 0.0;

  double const tol = 1.e-10;

  unsigned int const n_points_r   = r_values.size();
  unsigned int const n_points_phi = phi_values.size();

  double r = r_in;

  if(r > r_values[n_points_r - 1])
    r = r_values[n_points_r - 1];

  AssertThrow(r > (r_values[0] - tol) && r < (r_values[n_points_r - 1] + tol) &&
                phi > (phi_values[0] - tol) && phi < (phi_values[n_points_phi - 1] + tol),
              ExcMessage("invalid point found."));

  // interpolate r and phi-coordinates
  unsigned int i_r = 0, i_phi = 0;

  i_r = std::distance(r_values.begin(), std::lower_bound(r_values.begin(), r_values.end(), r));
  i_phi =
    std::distance(phi_values.begin(), std::lower_bound(phi_values.begin(), phi_values.end(), phi));

  // make sure that the index does not exceed the array bounds in case of round-off errors
  if(i_r == r_values.size())
    i_r--;
  if(i_phi == phi_values.size())
    i_phi--;

  if(i_r == 0)
    i_r++;
  if(i_phi == 0)
    i_phi++;

  AssertThrow(i_r > 0 && i_r < n_points_r && i_phi > 0 && i_phi < n_points_phi,
              ExcMessage("Invalid point found"));

  double const weight_r_p = (r - r_values[i_r - 1]) / (r_values[i_r] - r_values[i_r - 1]);
  double const weight_r_m = 1 - weight_r_p;
  double const weight_phi_p =
    (phi - phi_values[i_phi - 1]) / (phi_values[i_phi] - phi_values[i_phi - 1]);
  double const weight_phi_m = 1 - weight_phi_p;

  AssertThrow(-1.e-12 < weight_r_p && weight_r_p < 1. + 1e-12 && -1.e-12 < weight_r_m &&
                weight_r_m < 1. + 1e-12 && -1.e-12 < weight_phi_p && weight_phi_p < 1. + 1e-12 &&
                -1.e-12 < weight_phi_m && weight_phi_m < 1. + 1e-12,
              ExcMessage("invalid weights when interpolating solution in 2D."));

  result =
    weight_r_m * weight_phi_m * solution_values[(i_r - 1) * n_points_r + (i_phi - 1)][component] +
    weight_r_m * weight_phi_p * solution_values[(i_r - 1) * n_points_r + (i_phi)][component] +
    weight_r_p * weight_phi_m * solution_values[(i_r)*n_points_r + (i_phi - 1)][component] +
    weight_r_p * weight_phi_p * solution_values[(i_r)*n_points_r + (i_phi)][component];

  return result;
}

template<int dim, typename Number>
class InflowDataCalculator
{
public:
  InflowDataCalculator(InflowData<dim> const & inflow_data_in)
    : inflow_data(inflow_data_in), inflow_data_has_been_initialized(false)
  {
  }

  void
  setup(DoFHandler<dim> const & dof_handler_velocity_in, Mapping<dim> const & mapping_in)
  {
    dof_handler_velocity = &dof_handler_velocity_in;
    mapping              = &mapping_in;

    array_dof_index_and_shape_values.resize(inflow_data.n_points_y * inflow_data.n_points_z);
    array_counter.resize(inflow_data.n_points_y * inflow_data.n_points_z);
  }

  void
  calculate(LinearAlgebra::distributed::Vector<Number> const & velocity)
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

              double const x = (*inflow_data.y_values)[iy] * std::cos((*inflow_data.z_values)[iz]);
              double const y = (*inflow_data.y_values)[iy] * std::sin((*inflow_data.z_values)[iz]);
              point          = Point<dim>(x, y, inflow_data.normal_coordinate);
            }
            else
            {
              AssertThrow(false, ExcMessage("Not implemented."));
            }

            std::vector<std::pair<unsigned int, std::vector<Number>>>
              global_dof_index_and_shape_values;
            get_global_dof_index_and_shape_values(
              *dof_handler_velocity, *mapping, velocity, point, global_dof_index_and_shape_values);

            unsigned int array_index                      = iy * inflow_data.n_points_y + iz;
            array_dof_index_and_shape_values[array_index] = global_dof_index_and_shape_values;
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
          unsigned int array_index = iy * inflow_data.n_points_y + iz;

          // initialize with zeros since we accumulate into these variables
          (*inflow_data.array)[array_index] = 0.0;
          array_counter[array_index]        = 0;

          std::vector<std::pair<unsigned int, std::vector<Number>>> & vector(
            array_dof_index_and_shape_values[array_index]);

          // loop over all adjacent, locally owned cells for the current point
          for(typename std::vector<std::pair<unsigned int, std::vector<Number>>>::iterator iter =
                vector.begin();
              iter != vector.end();
              ++iter)
          {
            // increment counter (because this is a locally owned cell)
            array_counter[array_index] += 1;

            // interpolate solution using the precomputed shape values and the global dof index
            Tensor<1, dim, double> velocity_value;
            interpolate_value_vectorial_quantity(
              *dof_handler_velocity, velocity, iter->first, iter->second, velocity_value);

            // add result to array with velocity inflow data
            (*inflow_data.array)[array_index] += velocity_value;
          }
        }
      }

      // sum over all processors
      Utilities::MPI::sum(array_counter, MPI_COMM_WORLD, array_counter);
      Utilities::MPI::sum(
        ArrayView<const double>(&(*inflow_data.array)[0][0], dim * inflow_data.array->size()),
        MPI_COMM_WORLD,
        ArrayView<double>(&(*inflow_data.array)[0][0], dim * inflow_data.array->size()));

      // divide by counter in order to get the mean value (averaged over all
      // adjacent cells for a given point)
      for(unsigned int iy = 0; iy < inflow_data.n_points_y; ++iy)
      {
        for(unsigned int iz = 0; iz < inflow_data.n_points_z; ++iz)
        {
          unsigned int array_index = iy * inflow_data.n_points_y + iz;
          if(array_counter[array_index] >= 1)
            (*inflow_data.array)[array_index] /= double(array_counter[array_index]);
        }
      }
    }
  }

private:
  SmartPointer<DoFHandler<dim> const> dof_handler_velocity;
  SmartPointer<Mapping<dim> const>    mapping;
  InflowData<dim>                     inflow_data;
  bool                                inflow_data_has_been_initialized;
  std::vector<std::vector<std::pair<unsigned int, std::vector<Number>>>>
                            array_dof_index_and_shape_values;
  std::vector<unsigned int> array_counter;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_INFLOW_DATA_CALCULATOR_H_ */
