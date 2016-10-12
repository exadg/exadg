/*
 * PostProcessor.h
 *
 *  Created on: Aug 8, 2016
 *      Author: krank
 */

#ifndef INCLUDE_POSTPROCESSOR_H_
#define INCLUDE_POSTPROCESSOR_H_

#include "LiftAndDragData.h"
#include "PressureDifferenceData.h"

#include "../include/AnalyticalSolutionNavierStokes.h"

template<int dim>
struct PostProcessorData
{
  PostProcessorData()
    :
    dof_index_velocity(0),
    dof_index_pressure(1),
    quad_index_velocity(0)
  {}

  OutputDataNavierStokes output_data;
  ErrorCalculationData error_data;

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;
  unsigned int quad_index_velocity;

  LiftAndDragData lift_and_drag_data;
  PressureDifferenceData<dim> pressure_difference_data;
  MassConservationData mass_data;
};

template<int dim, int fe_degree, int fe_degree_p>
class PostProcessor
{
public:
  PostProcessor()
    :
    matrix_free_data(nullptr),
    time(0.0),
    time_step_number(1),
    output_counter(0),
    error_counter(0),
    clear_files_lift_and_drag(true),
    clear_files_pressure_difference(true),
    number_of_samples(0),
    divergence_sample(0.0),
    mass_sample(0.0)
  {}

  virtual ~PostProcessor()
  {

  }

  virtual void setup(){}

  virtual void setup(PostProcessorData<dim> const                                 &postprocessor_data_in,
                     DoFHandler<dim> const                                        &dof_handler_velocity_in,
                     DoFHandler<dim> const                                        &dof_handler_pressure_in,
                     Mapping<dim> const                                           &mapping_in,
                     MatrixFree<dim,double> const                                 &matrix_free_data_in,
                     std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution_in)
  {
    pp_data = postprocessor_data_in;
    dof_handler_velocity = &dof_handler_velocity_in;
    dof_handler_pressure = &dof_handler_pressure_in;
    mapping = &mapping_in;
    matrix_free_data = &matrix_free_data_in;
    analytical_solution = analytical_solution_in;
  }

  void init_from_restart(unsigned int o_counter)
  {
    output_counter = o_counter;
  }

  unsigned int get_output_counter() const {return output_counter;}

  virtual void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                                 parallel::distributed::Vector<double> const &pressure,
                                 parallel::distributed::Vector<double> const &vorticity,
                                 parallel::distributed::Vector<double> const &divergence,
                                 double const                                time_in,
                                 unsigned int const                          time_step_number_in)
  {
    time = time_in;
    time_step_number = time_step_number_in;

    /*
     *  write output
     */
    const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
    if( pp_data.output_data.write_output == true &&
        time > (pp_data.output_data.output_start_time + output_counter*pp_data.output_data.output_interval_time - EPSILON))
    {
      ConditionalOStream pcout(std::cout,
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << std::endl << "OUTPUT << Write data at time t = "
            << std::scientific << std::setprecision(4) << time << std::endl;

      write_output(velocity,pressure,vorticity,divergence);
      ++output_counter;
    }

    /*
     *  calculate error
     */
    if( (pp_data.error_data.analytical_solution_available == true) &&
        (time > (pp_data.error_data.error_calc_start_time + error_counter*pp_data.error_data.error_calc_interval_time - EPSILON)) )
    {
      ConditionalOStream pcout(std::cout,
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << std::endl << "Calculate error at time t = "
            << std::scientific << std::setprecision(4) << time << ":" << std::endl;

      calculate_error(velocity,pressure);
      ++error_counter;
    }

    /*
     *  calculation of lift and drag coefficients
     */
    if(pp_data.lift_and_drag_data.calculate_lift_and_drag == true)
      compute_lift_and_drag(velocity,pressure);
    /*
     *  calculation of pressure difference
     */
    if(pp_data.pressure_difference_data.calculate_pressure_difference == true)
      compute_pressure_difference(pressure);

  };

  // postprocessing for steady-state problems
  void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                         parallel::distributed::Vector<double> const &pressure,
                         parallel::distributed::Vector<double> const &vorticity,
                         parallel::distributed::Vector<double> const &divergence)
  {
    /*
     *  write output
     */
    if(pp_data.output_data.write_output == true)
    {
      ConditionalOStream pcout(std::cout,
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << std::endl << "OUTPUT << Write "
            << (output_counter == 0 ? "initial" : "solution") << " data"
            << std::endl;

      write_output(velocity,pressure,vorticity,divergence);
      ++output_counter;
    }

    /*
     *  calculate error
     */
    if(pp_data.error_data.analytical_solution_available == true)
    {
      ConditionalOStream pcout(std::cout,
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << std::endl << "Calculate error for "
            << (error_counter == 0 ? "initial" : "solution") << " data"
            << std::endl;

      calculate_error(velocity,pressure);
      ++error_counter;
    }

    /*
     *  calculation of lift and drag coefficients
     */
    if(pp_data.lift_and_drag_data.calculate_lift_and_drag == true)
      compute_lift_and_drag(velocity,pressure);
    /*
     *  calculation of pressure difference
     */
    if(pp_data.pressure_difference_data.calculate_pressure_difference == true)
      compute_pressure_difference(pressure);
  };

  virtual void analyze_divergence_error(parallel::distributed::Vector<double> const &velocity_temp,
                                        double const                                time_in,
                                        unsigned int const                          time_step_number_in)
  {
    time = time_in;
    time_step_number = time_step_number_in;

    write_divu_timeseries(velocity_temp);

    const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
    if(time > this->pp_data.mass_data.start_time-EPSILON &&
       time_step_number % this->pp_data.mass_data.sample_every_time_steps == 0)
    {
      write_divu_statistics(velocity_temp);
    }
  }

protected:
  SmartPointer< DoFHandler<dim> const > dof_handler_velocity;
  SmartPointer< DoFHandler<dim> const > dof_handler_pressure;
  SmartPointer< Mapping<dim> const > mapping;
  MatrixFree<dim,double> const * matrix_free_data;

  std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;

  PostProcessorData<dim> pp_data;

  double time;
  unsigned int time_step_number;
  unsigned int output_counter;
  unsigned int error_counter;

  mutable bool clear_files_lift_and_drag;
  mutable bool clear_files_pressure_difference;

  int number_of_samples;
  double divergence_sample;
  double mass_sample;

  void calculate_error(parallel::distributed::Vector<double> const &velocity,
                       parallel::distributed::Vector<double> const &pressure);

  virtual void write_output(parallel::distributed::Vector<double> const &velocity,
                            parallel::distributed::Vector<double> const &pressure,
                            parallel::distributed::Vector<double> const &vorticity,
                            parallel::distributed::Vector<double> const &divergence);

  void compute_lift_and_drag(parallel::distributed::Vector<double> const &velocity,
                             parallel::distributed::Vector<double> const &pressure) const;

  void compute_pressure_difference(parallel::distributed::Vector<double> const &pressure) const;

  void my_point_value(const Mapping<dim>                                                            &mapping,
                      const DoFHandler<dim>                                                         &dof_handler,
                      const parallel::distributed::Vector<double>                                   &solution,
                      const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >  &cell_point,
                      Vector<double>                                                                &value) const;

  void write_divu_timeseries(parallel::distributed::Vector<double> const &velocity_temp);

  void evaluate_mass_error(parallel::distributed::Vector<double> const &velocity_temp,
                           double                                      &divergence,
                           double                                      &volume,
                           double                                      &diff_mass,
                           double                                      &mean_mass);

  void local_compute_divu(const MatrixFree<dim,double>                &data,
                          std::vector<double >                        &test,
                          const parallel::distributed::Vector<double> &source,
                          const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_compute_divu_face (const MatrixFree<dim,double>                &data,
                                std::vector<double >                        &test,
                                const parallel::distributed::Vector<double> &source,
                                const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_compute_divu_boundary_face (const MatrixFree<dim,double>                &data,
                                         std::vector<double >                        &test,
                                         const parallel::distributed::Vector<double> &source,
                                         const std::pair<unsigned int,unsigned int>  &face_range) const;

  virtual void write_divu_statistics(parallel::distributed::Vector<double> const &velocity_temp);

};

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
calculate_error(parallel::distributed::Vector<double> const  &velocity,
                parallel::distributed::Vector<double> const  &pressure)
{
  ConditionalOStream pcout(std::cout,
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  Vector<double> error_norm_per_cell_u (dof_handler_velocity->get_triangulation().n_active_cells());
  Vector<double> solution_norm_per_cell_u (dof_handler_velocity->get_triangulation().n_active_cells());
  analytical_solution->velocity->set_time(time);
  VectorTools::integrate_difference (*mapping,
                                     *dof_handler_velocity,
                                     velocity,
                                     *(analytical_solution->velocity),
                                     error_norm_per_cell_u,
                                     QGauss<dim>(dof_handler_velocity->get_fe().degree+4),//(fe().degree+2),
                                     VectorTools::L2_norm);
  parallel::distributed::Vector<double> dummy_u;
  dummy_u.reinit(velocity);
  VectorTools::integrate_difference (*mapping,
                                     *dof_handler_velocity,
                                     dummy_u,
                                     *(analytical_solution->velocity),
                                     solution_norm_per_cell_u,
                                     QGauss<dim>(dof_handler_velocity->get_fe().degree+4), //(fe().degree+2),
                                     VectorTools::L2_norm);
  double error_norm_u = std::sqrt(Utilities::MPI::sum (error_norm_per_cell_u.norm_sqr(), MPI_COMM_WORLD));
  double solution_norm_u = std::sqrt(Utilities::MPI::sum (solution_norm_per_cell_u.norm_sqr(), MPI_COMM_WORLD));
  if(solution_norm_u > 1.e-12)
    pcout << "  Relative error (L2-norm) velocity u: "
          << std::scientific << std::setprecision(5) << error_norm_u/solution_norm_u << std::endl;
  else
    pcout << "  ABSOLUTE error (L2-norm) velocity u: "
          << std::scientific << std::setprecision(5) << error_norm_u << std::endl;

  Vector<double> error_norm_per_cell_p (dof_handler_pressure->get_triangulation().n_active_cells());
  Vector<double> solution_norm_per_cell_p (dof_handler_pressure->get_triangulation().n_active_cells());
  analytical_solution->pressure->set_time(time);
  VectorTools::integrate_difference (*mapping,
                                     *dof_handler_pressure,
                                     pressure,
                                     *(analytical_solution->pressure),
                                     error_norm_per_cell_p,
                                     QGauss<dim>(dof_handler_pressure->get_fe().degree+4), //(fe_p.degree+2),
                                     VectorTools::L2_norm);

  parallel::distributed::Vector<double> dummy_p;
  dummy_p.reinit(pressure);
  VectorTools::integrate_difference (*mapping,
                                     *dof_handler_pressure,
                                     dummy_p,
                                     *(analytical_solution->pressure),
                                     solution_norm_per_cell_p,
                                     QGauss<dim>(dof_handler_pressure->get_fe().degree+4), //(fe_p.degree+2),
                                     VectorTools::L2_norm);

  double error_norm_p = std::sqrt(Utilities::MPI::sum (error_norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));
  double solution_norm_p = std::sqrt(Utilities::MPI::sum (solution_norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));
  if(solution_norm_p > 1.e-12)
    pcout << "  Relative error (L2-norm) pressure p: "
          << std::scientific << std::setprecision(5) << error_norm_p/solution_norm_p << std::endl;
  else
    pcout << "  ABSOLUTE error (L2-norm) pressure p: "
          << std::scientific << std::setprecision(5) << error_norm_p << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
write_output(parallel::distributed::Vector<double> const &velocity,
             parallel::distributed::Vector<double> const &pressure,
             parallel::distributed::Vector<double> const &vorticity,
             parallel::distributed::Vector<double> const &divergence)
{
  DataOut<dim> data_out;
  std::vector<std::string> velocity_names (dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    velocity_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (*dof_handler_velocity, velocity, velocity_names, velocity_component_interpretation);

  std::vector<std::string> vorticity_names (dim, "vorticity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    vorticity_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (*dof_handler_velocity, vorticity, vorticity_names, vorticity_component_interpretation);

  pressure.update_ghost_values();
  data_out.add_data_vector (*dof_handler_pressure,pressure, "p");

  if(pp_data.output_data.compute_divergence == true)
  {
    std::vector<std::string> divergence_names (dim, "divergence");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      divergence_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (*dof_handler_pressure, divergence, divergence_names, divergence_component_interpretation);
  }

  std::ostringstream filename;
  filename << "output/"
           << pp_data.output_data.output_prefix
           << "_Proc"
           << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
           << "_"
           << output_counter
           << ".vtu";

  data_out.build_patches (*mapping, pp_data.output_data.number_of_patches, DataOut<dim>::curved_inner_cells);

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);

  if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
    {
      std::ostringstream filename;
      filename << pp_data.output_data.output_prefix
               << "_Proc"
               << i
               << "_"
               << output_counter
               << ".vtu";

      filenames.push_back(filename.str().c_str());
    }
    std::string master_name = "output/" + pp_data.output_data.output_prefix + "_" + Utilities::int_to_string(output_counter) + ".pvtu";
    std::ofstream master_output (master_name.c_str());
    data_out.write_pvtu_record (master_output, filenames);
  }
}

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
compute_lift_and_drag(parallel::distributed::Vector<double> const &velocity,
                      parallel::distributed::Vector<double> const &pressure) const
{
  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,double> fe_eval_velocity
      (*matrix_free_data,true,pp_data.dof_index_velocity,pp_data.quad_index_velocity);
  FEFaceEvaluation<dim,fe_degree_p,fe_degree+1,1,double> fe_eval_pressure
      (*matrix_free_data,true,pp_data.dof_index_pressure,pp_data.quad_index_velocity);

  Tensor<1,dim,double> Force;
  for(unsigned int d=0;d<dim;++d)
    Force[d] = 0.0;

  for(unsigned int face=matrix_free_data->n_macro_inner_faces();
      face<(matrix_free_data->n_macro_inner_faces()+matrix_free_data->n_macro_boundary_faces());
      face++)
  {
    fe_eval_velocity.reinit (face);
    fe_eval_velocity.read_dof_values(velocity);
    fe_eval_velocity.evaluate(false,true);

    fe_eval_pressure.reinit (face);
    fe_eval_pressure.read_dof_values(pressure);
    fe_eval_pressure.evaluate(true,false);

    typename std::set<types::boundary_id>::iterator it;
    types::boundary_id boundary_id = (*matrix_free_data).get_boundary_indicator(face);

    it = pp_data.lift_and_drag_data.boundary_IDs.find(boundary_id);
    if (it != pp_data.lift_and_drag_data.boundary_IDs.end())
    {
      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<double> pressure = fe_eval_pressure.get_value(q);
        Tensor<1,dim,VectorizedArray<double> > normal = fe_eval_velocity.get_normal_vector(q);
        Tensor<2,dim,VectorizedArray<double> > velocity_gradient = fe_eval_velocity.get_gradient(q);
        fe_eval_velocity.submit_value(pressure*normal -  make_vectorized_array<double>(pp_data.lift_and_drag_data.viscosity)*
                                        (velocity_gradient+transpose(velocity_gradient))*normal,q);
      }
      Tensor<1,dim,VectorizedArray<double> > Force_local = fe_eval_velocity.integrate_value();

      // sum over all entries of VectorizedArray
      for (unsigned int d=0; d<dim; ++d)
        for (unsigned int n=0; n<VectorizedArray<double>::n_array_elements; ++n)
          Force[d] += Force_local[d][n];
    }
  }
  Force = Utilities::MPI::sum(Force,MPI_COMM_WORLD);

  // compute lift and drag coefficients (c = (F/rho)/(1/2 U² A)
  const double reference_value = pp_data.lift_and_drag_data.reference_value;
  Force /= reference_value;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {
    std::string filename_drag, filename_lift;
    filename_drag = "output/FPC/"
        + pp_data.lift_and_drag_data.filename_prefix_drag
        + "_l" + Utilities::int_to_string(dof_handler_velocity->get_triangulation().n_levels()-1)
        + "_p" + Utilities::int_to_string(fe_degree)
        + "_drag.txt";
    filename_lift = "output/FPC/"
        + pp_data.lift_and_drag_data.filename_prefix_lift
        + "_l" + Utilities::int_to_string(dof_handler_velocity->get_triangulation().n_levels()-1)
        + "_p" + Utilities::int_to_string(fe_degree)
        + "_lift.txt";

    std::ofstream f_drag, f_lift;
    if(clear_files_lift_and_drag)
    {
      f_drag.open(filename_drag.c_str(),std::ios::trunc);
      f_lift.open(filename_lift.c_str(),std::ios::trunc);
      clear_files_lift_and_drag = false;
    }
    else
    {
      f_drag.open(filename_drag.c_str(),std::ios::app);
      f_lift.open(filename_lift.c_str(),std::ios::app);
    }

    f_drag << std::scientific << std::setprecision(6)
           << time << "\t" << Force[0] << std::endl;
    f_drag.close();

    f_lift << std::scientific << std::setprecision(6)
           << time << "\t" << Force[1] << std::endl;
    f_lift.close();
  }
}

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
compute_pressure_difference(parallel::distributed::Vector<double> const &pressure) const
{
  double pressure_1 = 0.0, pressure_2 = 0.0;
  unsigned int counter_1 = 0, counter_2 = 0;

  Point<dim> point_1, point_2;
  point_1 = pp_data.pressure_difference_data.point_1;
  point_2 = pp_data.pressure_difference_data.point_2;

  // parallel computation
  const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
  cell_point_1 = GridTools::find_active_cell_around_point (*mapping,
                                                           *dof_handler_pressure,
                                                           point_1);
  if(cell_point_1.first->is_locally_owned())
  {
    counter_1 = 1;

    Vector<double> value(1);
    my_point_value(*mapping,
                   *dof_handler_pressure,
                   pressure,
                   cell_point_1,
                   value);

    pressure_1 = value(0);
  }
  counter_1 = Utilities::MPI::sum(counter_1,MPI_COMM_WORLD);
  pressure_1 = Utilities::MPI::sum(pressure_1,MPI_COMM_WORLD);
  pressure_1 = pressure_1/counter_1;

  const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
  cell_point_2 = GridTools::find_active_cell_around_point (*mapping,
                                                           *dof_handler_pressure,
                                                           point_2);
  if(cell_point_2.first->is_locally_owned())
  {
    counter_2 = 1;

    Vector<double> value(1);
    my_point_value(*mapping,
                   *dof_handler_pressure,
                   pressure,
                   cell_point_2,
                   value);

    pressure_2 = value(0);
  }
  counter_2 = Utilities::MPI::sum(counter_2,MPI_COMM_WORLD);
  pressure_2 = Utilities::MPI::sum(pressure_2,MPI_COMM_WORLD);
  pressure_2 = pressure_2/counter_2;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {
    std::string filename = "output/FPC/"
        + pp_data.pressure_difference_data.filename_prefix_pressure_difference
        + "_l" + Utilities::int_to_string(dof_handler_velocity->get_triangulation().n_levels()-1)
        + "_p" + Utilities::int_to_string(fe_degree)
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
    f << std::scientific << std::setprecision(6) << time << "\t" << pressure_1-pressure_2 << std::endl;
    f.close();
  }
}


template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
my_point_value(const Mapping<dim>                                                           &mapping,
               const DoFHandler<dim>                                                        &dof_handler,
               const parallel::distributed::Vector<double>                                  &solution,
               const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > &cell_point,
               Vector<double>                                                               &value) const
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

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
evaluate_mass_error(parallel::distributed::Vector<double> const &velocity_temp,
                    double                                      &divergence,
                    double                                      &volume,
                    double                                      &diff_mass,
                    double                                      &mean_mass)
{
  std::vector<double> dst(4,0.0);
  matrix_free_data->loop (&PostProcessor<dim,fe_degree,fe_degree_p>::local_compute_divu,
                          &PostProcessor<dim,fe_degree,fe_degree_p>::local_compute_divu_face,
                          &PostProcessor<dim,fe_degree,fe_degree_p>::local_compute_divu_boundary_face,
                          this, dst, velocity_temp);

  divergence = Utilities::MPI::sum (dst.at(0), MPI_COMM_WORLD);
  volume = Utilities::MPI::sum (dst.at(1), MPI_COMM_WORLD);
  diff_mass = Utilities::MPI::sum (dst.at(2), MPI_COMM_WORLD);
  mean_mass = Utilities::MPI::sum (dst.at(3), MPI_COMM_WORLD);
}

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
write_divu_timeseries(parallel::distributed::Vector<double> const &velocity_temp)
{
  double divergence = 0.;
  double volume = 0.;
  double diff_mass = 0.;
  double mean_mass = 0.;
  evaluate_mass_error(velocity_temp, divergence, volume, diff_mass, mean_mass);
  double div_normalized = divergence/volume;
  double diff_mass_normalized = diff_mass/mean_mass;
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {
    std::ostringstream filename;
    filename << pp_data.output_data.output_prefix << ".divu_timeseries";

    std::ofstream f;
    if(time_step_number==1)
    {
      f.open(filename.str().c_str(),std::ios::trunc);
      f << "Error incompressibility constraint:\n\n\t(1,|divu|)_Omega/(1,1)_Omega\n" << std::endl
        << "Error mass flux over interior element faces:\n\n\t(1,|(um - up)*n|)_dOmegaI / (1,|0.5(um + up)*n|)_dOmegaI\n" << std::endl
        << "       n       |       t      |    divergence    |      mass       " << std::endl;
    }
    else
    {
      f.open(filename.str().c_str(),std::ios::app);
    }
    f << std::setw(15) <<time_step_number;
    f << std::scientific<<std::setprecision(7) << std::setw(15) << time;
    f << std::scientific<<std::setprecision(7) << std::setw(15) << div_normalized;
    f << std::scientific<<std::setprecision(7) << std::setw(15) << diff_mass_normalized << std::endl;
  }
}

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
local_compute_divu(const MatrixFree<dim,double>                &data,
                   std::vector<double>                         &dst,
                   const parallel::distributed::Vector<double> &source,
                   const std::pair<unsigned int,unsigned int>  &cell_range) const
{
  FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> phi(data,0,0);

  AlignedVector<VectorizedArray<double> > JxW_values(phi.n_q_points);
  VectorizedArray<double> div_vec = make_vectorized_array(0.);
  VectorizedArray<double> vol_vec = make_vectorized_array(0.);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit(cell);
    phi.read_dof_values(source);
    phi.evaluate(false,true);
    phi.fill_JxW_values(JxW_values);

    for (unsigned int q=0; q<phi.n_q_points; ++q)
    {
      vol_vec += JxW_values[q];
      div_vec += JxW_values[q]*std::abs(phi.get_divergence(q));
    }
  }
  double div = 0.;
  double vol = 0.;
  for (unsigned int v=0;v<VectorizedArray<double>::n_array_elements;v++)
  {
    div += div_vec[v];
    vol += vol_vec[v];
  }
  dst.at(0) += div;
  dst.at(1) += vol;
}

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
local_compute_divu_face (const MatrixFree<dim,double>                &data,
                         std::vector<double >                        &dst,
                         const parallel::distributed::Vector<double> &source,
                         const std::pair<unsigned int,unsigned int>  &face_range) const
{

  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,double> fe_eval_xwall(data,true,0,0);
  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,double> fe_eval_xwall_neighbor(data,false,0,0);

  AlignedVector<VectorizedArray<double> > JxW_values(fe_eval_xwall.n_q_points);
  VectorizedArray<double> diff_mass_flux_vec = make_vectorized_array(0.);
  VectorizedArray<double> mean_mass_flux_vec = make_vectorized_array(0.);
  for (unsigned int face=face_range.first; face<face_range.second; ++face)
  {
    fe_eval_xwall.reinit(face);
    fe_eval_xwall.read_dof_values(source);
    fe_eval_xwall.evaluate(true,false);
    fe_eval_xwall_neighbor.reinit(face);
    fe_eval_xwall_neighbor.read_dof_values(source);
    fe_eval_xwall_neighbor.evaluate(true,false);
    fe_eval_xwall.fill_JxW_values(JxW_values);

    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      mean_mass_flux_vec += JxW_values[q]*std::abs(0.5*(fe_eval_xwall.get_value(q)+fe_eval_xwall_neighbor.get_value(q))*fe_eval_xwall.get_normal_vector(q));

      diff_mass_flux_vec += JxW_values[q]*std::abs((fe_eval_xwall.get_value(q)-fe_eval_xwall_neighbor.get_value(q))*fe_eval_xwall.get_normal_vector(q));
    }
  }
  double diff_mass_flux = 0.;
  double mean_mass_flux = 0.;
  for (unsigned int v=0;v<VectorizedArray<double>::n_array_elements;v++)
  {
    diff_mass_flux += diff_mass_flux_vec[v];
    mean_mass_flux += mean_mass_flux_vec[v];
  }
  dst.at(2) += diff_mass_flux;
  dst.at(3) += mean_mass_flux;
}

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
local_compute_divu_boundary_face (const MatrixFree<dim,double>                 &,
                                  std::vector<double >                         &,
                                  const parallel::distributed::Vector<double>  &,
                                  const std::pair<unsigned int,unsigned int>   &) const
{

}

template<int dim, int fe_degree, int fe_degree_p>
void PostProcessor<dim,fe_degree,fe_degree_p>::
write_divu_statistics(parallel::distributed::Vector<double> const &velocity_temp)
{
  ++number_of_samples;

  double divergence = 0.;
  double volume = 0.;
  double diff_mass = 0.;
  double mean_mass = 0.;
  this->evaluate_mass_error(velocity_temp, divergence, volume, diff_mass, mean_mass);

  divergence_sample += divergence/volume;
  mass_sample += diff_mass/mean_mass;
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {
    std::ostringstream filename;
    filename << this->pp_data.output_data.output_prefix << ".divu_statistics";

    std::ofstream f;

    f.open(filename.str().c_str(),std::ios::trunc);
    f << "average divergence over space and time" << std::endl;
    f << "number of samples:   " << number_of_samples << std::endl;
    f << "Mean error incompressibility constraint:   " << divergence_sample/number_of_samples << std::endl;
    f << "Mean error mass flux over interior element faces:  " << mass_sample/number_of_samples << std::endl;
    f.close();
  }
}



#endif /* INCLUDE_POSTPROCESSOR_H_ */
