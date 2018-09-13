/*
 * calculate_characteristic_element_length.h
 *
 *  Created on: May 31, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_CALCULATE_CHARACTERISTIC_ELEMENT_LENGTH_H_
#define INCLUDE_FUNCTIONALITIES_CALCULATE_CHARACTERISTIC_ELEMENT_LENGTH_H_


/*
 *  This function calculates the characteristic element length h
 *  defined as h = min_{e=1,...,N_el} h_e, where h_e is the
 *  minimum vertex distance of element e.
 */
template<int dim>
double
calculate_minimum_vertex_distance(Triangulation<dim> const & triangulation)
{
  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
                                                    endc = triangulation.end();

  double diameter = 0.0, min_cell_diameter = std::numeric_limits<double>::max();

  for(; cell != endc; ++cell)
  {
    if(cell->is_locally_owned())
    {
      diameter = cell->minimum_vertex_distance();
      if(diameter < min_cell_diameter)
        min_cell_diameter = diameter;
    }
  }

  const double global_min_cell_diameter = -Utilities::MPI::max(-min_cell_diameter, MPI_COMM_WORLD);

  return global_min_cell_diameter;
}

double
calculate_characteristic_element_length(double const element_length, unsigned int const fe_degree)
{
  return element_length / ((double)(fe_degree + 1));
}


#endif /* INCLUDE_FUNCTIONALITIES_CALCULATE_CHARACTERISTIC_ELEMENT_LENGTH_H_ */
