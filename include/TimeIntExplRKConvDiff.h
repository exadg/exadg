/*
 * TimeIntExplRKConvDiff.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTEXPLRKCONVDIFF_H_
#define INCLUDE_TIMEINTEXPLRKCONVDIFF_H_

template<int dim> class PostProcessor;

template<int dim, int fe_degree, typename value_type>
class TimeIntExplRKConvDiff
{
public:
  TimeIntExplRKConvDiff(std_cxx11::shared_ptr<DGConvDiffOperation<dim, fe_degree, value_type> > conv_diff_operation_in,
                        std_cxx11::shared_ptr<PostProcessor<dim> >                              postprocessor_in,
                        InputParametersConvDiff const                                           &param_in,
                        std_cxx11::shared_ptr<Function<dim> >                                   velocity_in,
                        unsigned int const                                                      n_refine_time_in)
    :
    conv_diff_operation(conv_diff_operation_in),
    postprocessor(postprocessor_in),
    param(param_in),
    velocity(velocity_in),
    total_time(0.0),
    time(param.start_time),
    time_step(1.0),
    order(param.order_time_integrator),
    cfl_number(param.cfl_number/std::pow(2.0,n_refine_time_in)),
    diffusion_number(param.diffusion_number/std::pow(2.0,n_refine_time_in))
  {}

  void timeloop();

  void setup();

private:
  void initialize_vectors();
  void initialize_solution();
  void postprocessing() const;
  void solve_timestep();
  void prepare_vectors_for_next_timestep();
  void calculate_timestep();
  void analyze_computing_times() const;

  std_cxx11::shared_ptr<DGConvDiffOperation<dim, fe_degree, value_type> > conv_diff_operation;
  std_cxx11::shared_ptr<PostProcessor<dim> > postprocessor;
  InputParametersConvDiff const & param;
  std_cxx11::shared_ptr<Function<dim> > velocity;

  Timer global_timer;
  value_type total_time;

  parallel::distributed::Vector<value_type> solution_n, solution_np;

  parallel::distributed::Vector<value_type> vec_rhs, vec_temp;

  value_type time, time_step;
  unsigned int const order;
  double const cfl_number;
  double const diffusion_number;
};

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // calculate time step size
  calculate_timestep();

  // initializes the solution by interpolation of analytical solution
  initialize_solution();
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::initialize_vectors()
{
  conv_diff_operation->initialize_solution_vector(solution_n);
  conv_diff_operation->initialize_solution_vector(solution_np);

  //TODO: only initialize these vectors if necessary
  conv_diff_operation->initialize_solution_vector(vec_rhs);
  conv_diff_operation->initialize_solution_vector(vec_temp);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::initialize_solution()
{
  conv_diff_operation->prescribe_initial_conditions(solution_n,time);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::calculate_timestep()
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  typename Triangulation<dim>::active_cell_iterator
    cell = conv_diff_operation->get_data().get_dof_handler().get_triangulation().begin_active(),
    endc =  conv_diff_operation->get_data().get_dof_handler().get_triangulation().end();

  double diameter = 0.0, min_cell_diameter = std::numeric_limits<double>::max();
  Tensor<1,dim,value_type> vel;
  velocity->set_time(time);
  double a = 0.0, max_cell_a = std::numeric_limits<double>::min();
  for (; cell!=endc; ++cell)
  {
    // calculate minimum diameter
    diameter = cell->diameter()/std::sqrt(dim); // diameter is the largest diagonal -> divide by sqrt(dim)
    //diameter = cell->minimum_vertex_distance();
    if (diameter < min_cell_diameter)
      min_cell_diameter = diameter;

    // calculate maximum velocity a
    Point<dim> point = cell->center();

    for(unsigned int d=0;d<dim;++d)
      vel[d] = velocity->value(point,d);

    a = vel.norm();
    if (a > max_cell_a)
      max_cell_a = a;
  }
  const double global_min_cell_diameter = -Utilities::MPI::max(-min_cell_diameter, MPI_COMM_WORLD);
  const double global_max_cell_a = Utilities::MPI::max(max_cell_a, MPI_COMM_WORLD);
  pcout << std::endl << "min cell diameter:\t" << std::setw(10) << global_min_cell_diameter;
  pcout << std::endl << "maximum velocity:\t" << std::setw(10) << global_max_cell_a << std::endl;

  // diffusion_number = diffusivity * time_step / d_min²
  double time_step_diff = std::numeric_limits<double>::max();
  if(param.diffusivity > 1.0e-12)
  {
    time_step_diff = diffusion_number/pow(fe_degree,3.0) * pow(global_min_cell_diameter,2.0) / param.diffusivity;
    pcout << std::endl << "time step size (diffusion):\t" << std::setw(10) << time_step_diff;
  }
  else
    pcout << std::endl << "time step size (diffusion):\t" << std::setw(10) << "infinity";

  // cfl = a * time_step / d_min
  double time_step_conv = std::numeric_limits<double>::max();
  if(a > 1.0e-12)
  {
    time_step_conv = cfl_number/pow(fe_degree,2.0)* global_min_cell_diameter / global_max_cell_a;
    pcout << std::endl << "time step size (convection):\t" << std::setw(10) << time_step_conv;
  }
  else
    pcout << std::endl << "time step size (convection):\t" << std::setw(10) << "infinity";

  //adopt minimum time step size
  time_step = time_step_diff < time_step_conv ? time_step_diff : time_step_conv;

  // decrease time_step in order to exactly hit END_TIME
  time_step = (param.end_time-param.start_time)/(1+int((param.end_time-param.start_time)/time_step));

  pcout << std::endl << "time step size (combined):\t" << std::setw(10) << time_step << std::endl;
}


template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::timeloop()
{
  global_timer.restart();

  postprocessing();

  const double EPSILON = 1.0e-10;
  while(time<(param.end_time-EPSILON))
  {
    solve_timestep();

    prepare_vectors_for_next_timestep();

    time += time_step;

    postprocessing();
  }

  total_time += global_timer.wall_time();

  analyze_computing_times();
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::postprocessing() const
{
  postprocessor->do_postprocessing(solution_n,time);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::
prepare_vectors_for_next_timestep()
{
  // solution at t_n+1 -> solution at t_n
  solution_n.swap(solution_np);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::
solve_timestep()
{
  // RK4
  solution_np = solution_n;
  // stage 1
  conv_diff_operation->evaluate(vec_temp,solution_n,time);
  solution_np.add(time_step/6., vec_temp);

  // stage 2
//  vec_rhs.equ(1., solution_n, time_step/2., vec_temp);
  vec_rhs.equ(1.,solution_n);
  vec_rhs.add(time_step/2., vec_temp);
  conv_diff_operation->evaluate(vec_temp,vec_rhs,time+time_step/2.);
  solution_np.add(time_step/3., vec_temp);

  // stage 3
//  vec_rhs.equ(1., solution_n, time_step/2., vec_temp);
  vec_rhs.equ(1., solution_n);
  vec_rhs.add(time_step/2., vec_temp);
  conv_diff_operation->evaluate(vec_temp,vec_rhs,time+time_step/2.);
  solution_np.add(time_step/3., vec_temp);

  // stage 4
//  vec_rhs.equ(1., solution_n, time_step, vec_temp);
  vec_rhs.equ(1., solution_n);
  vec_rhs.add(time_step, vec_temp);
  conv_diff_operation->evaluate(vec_temp,vec_rhs,time+time_step);
  solution_np.add(time_step/6., vec_temp);
}

template<int dim, int fe_degree, typename value_type>
void TimeIntExplRKConvDiff<dim,fe_degree,value_type>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "_________________________________________________________________________________" << std::endl
        << std::endl << "Computing times:          min        avg        max        rel      p_min  p_max" << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (this->total_time, MPI_COMM_WORLD);
  pcout  << "  Global time:         " << std::scientific
         << std::setprecision(4) << std::setw(10) << data.min << " "
         << std::setprecision(4) << std::setw(10) << data.avg << " "
         << std::setprecision(4) << std::setw(10) << data.max << " "
         << "          " << "  "
         << std::setw(6) << std::left << data.min_index << " "
         << std::setw(6) << std::left << data.max_index << std::endl
         << "_________________________________________________________________________________"
         << std::endl << std::endl;
}

#endif /* INCLUDE_TIMEINTEXPLRKCONVDIFF_H_ */
