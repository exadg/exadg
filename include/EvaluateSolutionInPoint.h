/*
 * PressureDifferenceCalculation.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EVALUATESOLUTIONINPOINT_H_
#define INCLUDE_EVALUATESOLUTIONINPOINT_H_

template<int dim>
void my_point_value(const Mapping<dim>                                                           &mapping,
                    const DoFHandler<dim>                                                        &dof_handler,
                    const parallel::distributed::Vector<double>                                  &solution,
                    const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > &cell_point,
                    Vector<double>                                                               &value)
{
  const FiniteElement<dim> &fe = dof_handler.get_fe();
  Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,ExcInternalError());

  const Quadrature<dim> quadrature (GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

  FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
  fe_values.reinit(cell_point.first);

  // then use this to get the values of the given fe_function at this point
  std::vector<Vector<double> > u_value(1, Vector<double> (fe.n_components()));
  fe_values.get_function_values(solution, u_value);
  value = u_value[0];
}

template<int dim>
void evaluate_solution_in_point(DoFHandler<dim> const                       &dof_handler,
                                Mapping<dim> const                          &mapping,
                                parallel::distributed::Vector<double> const &numerical_solution,
                                Point<dim> const                            &point,
                                double                                      &solution_value)
{
  // parallel computation
  const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
  cell = GridTools::find_active_cell_around_point (mapping,dof_handler,point);

  unsigned int counter = 0;

  if(cell.first->is_locally_owned())
  {
    counter = 1;

    Vector<double> value(1);
    my_point_value(mapping,
                   dof_handler,
                   numerical_solution,
                   cell,
                   value);

    solution_value = value(0);
  }
  counter = Utilities::MPI::sum(counter,MPI_COMM_WORLD);
  solution_value = Utilities::MPI::sum(solution_value,MPI_COMM_WORLD);
  solution_value /= counter;
}

template<int dim, int fe_degree_u, int fe_degree_p>
class PressureDifferenceCalculator
{
public:
  PressureDifferenceCalculator()
    :
    clear_files_pressure_difference(true)
  {}

  void setup(DoFHandler<dim> const             &dof_handler_pressure_in,
             Mapping<dim> const                &mapping_in,
             PressureDifferenceData<dim> const &pressure_difference_data_in)
  {
    dof_handler_pressure = &dof_handler_pressure_in;
    mapping = &mapping_in;
    pressure_difference_data = pressure_difference_data_in;
  }

  void evaluate(parallel::distributed::Vector<double> const &pressure,
                double const                                &time) const
  {
    if(pressure_difference_data.calculate_pressure_difference == true)
    {
      double pressure_1 = 0.0, pressure_2 = 0.0;

      Point<dim> point_1, point_2;
      point_1 = pressure_difference_data.point_1;
      point_2 = pressure_difference_data.point_2;

      evaluate_solution_in_point<dim>(*dof_handler_pressure,*mapping,pressure,point_1,pressure_1);
      evaluate_solution_in_point<dim>(*dof_handler_pressure,*mapping,pressure,point_2,pressure_2);

      double const pressure_difference = pressure_1 - pressure_2;

      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::string filename = "output/FPC/"
            + pressure_difference_data.filename_prefix_pressure_difference
            + "_refine_" + Utilities::int_to_string(dof_handler_pressure->get_triangulation().n_levels()-1)
            + "_fe_degree_" + Utilities::int_to_string(fe_degree_u) + Utilities::int_to_string(fe_degree_p)
            + "_pressure_difference.txt";

        std::ofstream f;
        if(clear_files_pressure_difference)
        {
          f.open(filename.c_str(),std::ios::trunc);
          clear_files_pressure_difference = false;
        }
        else
        {
          f.open(filename.c_str(),std::ios::app);
        }
        f << std::scientific << std::setprecision(6) << time << "\t" << pressure_difference << std::endl;
        f.close();
      }
    }
  }

private:
  mutable bool clear_files_pressure_difference;

  SmartPointer< DoFHandler<dim> const > dof_handler_pressure;
  SmartPointer< Mapping<dim> const > mapping;

  PressureDifferenceData<dim> pressure_difference_data;

};



#endif /* INCLUDE_EVALUATESOLUTIONINPOINT_H_ */
