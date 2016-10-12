/*
 * Restart.h
 *
 *  Created on: Jul 6, 2016
 *      Author: krank
 */

#ifndef INCLUDE_RESTART_H_
#define INCLUDE_RESTART_H_

#include <deal.II/lac/vector_view.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

template<int dim, int fe_degree, int fe_degree_p> class PostProcessor;

template<int dim>
const std::string restart_filename(InputParametersNavierStokes<dim> const & param)
{
  const std::string filename = param.output_data.output_prefix + "." +
    Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
    ".restart";
  return filename;
}

void check_file(std::ifstream const & in, const std::string filename)
{
  if (!in)
    AssertThrow (false,
                 ExcMessage (std::string("You are trying to restart a previous computation, "
                                         "but the restart file <") +
                             filename +
                             "> does not appear to exist on proc " +
                             Utilities::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
                             "!"));
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall,typename value_type>
void resume_restart(boost::archive::binary_iarchive & ia, InputParametersNavierStokes<dim> const & param, double & time,
    std_cxx11::shared_ptr<PostProcessor<dim, fe_degree, fe_degree_p> > & postprocessor,
    std::vector<value_type> & time_steps, unsigned int const order)
{

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << std::endl << std::endl << "______________________________________________________________________" << std::endl << std::endl;
    std::cout << " Resuming from a restart file " << std::endl;
  }

  // the operations done here must be in sync with the output
  std::vector <value_type> tmp_time_steps;
  unsigned int n_ranks;
  unsigned int output_counter;
  unsigned int tmp_order;
  ia & n_ranks;
  ia & time;
  ia & tmp_order;
  tmp_time_steps.resize(tmp_order);
  for (unsigned int i = 0; i < tmp_order; i++)
    ia & tmp_time_steps[i];
  ia & output_counter;

  postprocessor->init_from_restart(output_counter);

  if(param.start_with_low_order == false)
  {
    AssertThrow(tmp_order == order, ExcMessage("temporal orders have to be identical if start_with_low_order == false"));
  }

  for (unsigned int i=0; i<std::min(time_steps.size(),tmp_time_steps.size()); i++)
  {
    time_steps[i] = tmp_time_steps[i];
  }

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << " Resuming to time t = " << time ;
    if(param.start_with_low_order)
      std::cout << " and compute new time steps according to input parameters. Old time step was " << time_steps[0] << std::endl;
    else
      std::cout << " and continue with old time step " << time_steps[0] << std::endl;
  }

  AssertThrow(n_ranks == Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
              ExcMessage("Tried to resume on "
                  +
                         Utilities::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
                  +
                         " processes, but restart was written on "
                  +
                         Utilities::to_string(n_ranks)
                  +
                         " processes. Processor numbers must match!"));

}

template<int dim, typename value_type>
void write_restart_preamble(boost::archive::binary_oarchive & oa, InputParametersNavierStokes<dim> const & param, std::vector<value_type> const & time_steps,
                   double const time, unsigned int const output_counter, unsigned int const order)
{

  unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  oa & n_ranks;
  oa & time;
  oa & order;
  for (unsigned int i = 0; i< order;i++)
    oa & time_steps[i];
  oa & output_counter;


  // move current restart to old one in case something fails while writing
  static bool first_time = true;
  if (first_time == false)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      unsigned int n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)-1;
      for (unsigned int n = 0; n < n_ranks; n++)
      {
        const std::string rank_string = Utilities::int_to_string(n);
        const int error = system (("mv -f " + param.output_data.output_prefix  +"." + rank_string + ".restart" + " " + param.output_data.output_prefix +"." + rank_string + ".restart" + ".old").c_str());
        AssertThrow (error == 0, ExcMessage(std::string ("Can't move files: ")
                        +
                        param.output_data.output_prefix  +"." + rank_string + ".restart" + " -> " + param.output_data.output_prefix  +"." + rank_string + ".restart" +".old"));
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
  first_time = false;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << std::endl << std::endl << "______________________________________________________________________" << std::endl << std::endl;
    std::cout << " Writing a restart file " << std::endl;
  }
}

template<int dim>
void write_restart_file(std::ostringstream & oss, InputParametersNavierStokes<dim> const & param)
{
  const std::string filename = restart_filename(param);
  std::ofstream stream(filename.c_str());
  stream << oss.str() << std::endl;
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << " Done writing file " << std::endl;
    std::cout << std::endl << "______________________________________________________________________" << std::endl << std::endl;
  }
}

void finished_reading_restart_output()
{
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << " Done reading vectors " << std::endl;
    std::cout << std::endl << "______________________________________________________________________" << std::endl << std::endl;
  }
}

#endif /* INCLUDE_RESTART_H_ */
