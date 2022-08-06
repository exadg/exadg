/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */


// deal.II
#include <deal.II/lac/full_matrix.h>

// ExaDG
#include <exadg/postprocessor/pointwise_output_generator_base.h>
#include <exadg/utilities/create_directories.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
template<int dim>
PointwiseOutputDataBase<dim>::PointwiseOutputDataBase()
  : write_output(false),
    directory("output/"),
    filename("name"),
    update_points_before_evaluation(false),
    start_time(std::numeric_limits<double>::max()),
    end_time(std::numeric_limits<double>::max()),
    interval_time(std::numeric_limits<double>::max())
{
}

template<int dim>
void
PointwiseOutputDataBase<dim>::print(dealii::ConditionalOStream & pcout) const
{
  ExaDG::print_parameter(pcout, "PointwiseOutput", write_output && evaluation_points.size() > 0);

  if(write_output && evaluation_points.size() > 0)
  {
    ExaDG::print_parameter(pcout, "Output directory", directory);
    ExaDG::print_parameter(pcout, "Name of output file", filename);
    ExaDG::print_parameter(pcout, "Output start time", start_time);
    ExaDG::print_parameter(pcout, "Output end time", end_time);
    ExaDG::print_parameter(pcout, "Output interval time", interval_time);
    ExaDG::print_parameter(pcout,
                           "Update Points before evaluation",
                           update_points_before_evaluation);
  }
}

template class PointwiseOutputDataBase<2>;
template class PointwiseOutputDataBase<3>;

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::evaluate(VectorType const & solution,
                                                    double const &     time,
                                                    int const &        time_step_number)
{
  if(pointwise_output_data.write_output && pointwise_output_data.evaluation_points.size() > 0)
  {
    if(time_step_number >= 0) // unsteady problem
    {
      // small number which is much smaller than the time step size
      double const EPSILON = 1.0e-10;

      // In the first time step, the current time might be larger than start_time. In that
      // case, we first have to reset the counter in order to avoid that output is written every
      // time step.
      if(reset_counter)
      {
        counter += int((time - pointwise_output_data.start_time + EPSILON) /
                       pointwise_output_data.interval_time);
        reset_counter = false;
      }


      if((time > (pointwise_output_data.start_time + counter * pointwise_output_data.interval_time -
                  EPSILON)) &&
         time < pointwise_output_data.end_time + EPSILON)
      {
        if(pointwise_output_data.update_points_before_evaluation)
          reinit_remote_evaluator();

        write_time(time);
        do_evaluate(solution);

        ++counter;
      }
    }
    else // steady problem (time_step_number = -1)
    {
      write_time(time);
      do_evaluate(solution);
      ++counter;
    }
  }
}

template<int dim, typename Number>
PointwiseOutputGeneratorBase<dim, Number>::PointwiseOutputGeneratorBase(MPI_Comm const & comm)
  : mpi_comm(comm), counter(0), reset_counter(true)
{
}

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::setup_base(
  dealii::DoFHandler<dim> const &      dof_handler_in,
  dealii::Mapping<dim> const &         mapping_in,
  PointwiseOutputDataBase<dim> const & pointwise_output_data_in)
{
#ifdef DEAL_II_WITH_HDF5
  pointwise_output_data = pointwise_output_data_in;

  if(pointwise_output_data.write_output && pointwise_output_data.evaluation_points.size() > 0)
  {
    dof_handler = &dof_handler_in;
    mapping     = &mapping_in;

    // allocate memory for hyperslab
    componentwise_result.reinit(pointwise_output_data.evaluation_points.size());

    // number of samples to write into file
    n_out_samples = 1 + static_cast<unsigned int>(std::ceil(
                          (pointwise_output_data.end_time - pointwise_output_data.start_time) /
                          pointwise_output_data.interval_time));

    setup_remote_evaluator();

    create_hdf5_file();

    {
      auto group = hdf5_file->create_group("GeneralInformation");
      add_evaluation_points_dataset(group, "EvaluationPoints");
      write_evaluation_points("GeneralInformation/EvaluationPoints");
    }

    {
      auto group = hdf5_file->create_group("PhysicalInformation");
      add_time_dataset(group, "Time");
    }
  }
#else
  (void)dof_handler_in;
  (void)mapping_in;
  (void)pointwise_output_data_in;
  AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with HDF5!"));
#endif
}

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::add_quantity(std::string const & name,
                                                        unsigned int const  n_components)
{
  AssertThrow(n_components > 0, dealii::ExcMessage("n_components has to be > 0."));

  auto const & [it, success] = name_to_components.try_emplace(name, n_components);
  AssertThrow(success, dealii::ExcMessage("Name already given to quantity dataset."));

#ifdef DEAL_II_WITH_HDF5
  auto group = hdf5_file->open_group("PhysicalInformation");
  for(unsigned int comp = 0; comp < n_components; ++comp)
  {
    group.template create_dataset<Number>(
      (n_components == 1) ? name : name + std::to_string(comp),
      std::vector<hsize_t>{pointwise_output_data.evaluation_points.size(), 1 * n_out_samples});
  }
#else
  AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with HDF5!"));
#endif
}

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::setup_remote_evaluator()
{
  remote_evaluator =
    std::make_shared<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(1e-6, false, 0);
  reinit_remote_evaluator();
}

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::reinit_remote_evaluator()
{
  remote_evaluator->reinit(pointwise_output_data.evaluation_points,
                           dof_handler->get_triangulation(),
                           *mapping);
  AssertThrow(remote_evaluator->all_points_found(),
              dealii::ExcMessage("Not all remote points found."));
}

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::create_hdf5_file()
{
#ifdef DEAL_II_WITH_HDF5
  ExaDG::create_directories(pointwise_output_data.directory, mpi_comm);
  hdf5_file = std::make_unique<dealii::HDF5::File>(pointwise_output_data.directory +
                                                     pointwise_output_data.filename + ".h5",
                                                   dealii::HDF5::File::FileAccessMode::create,
                                                   mpi_comm);
#else
  AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with HDF5!"));
#endif
}

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::write_evaluation_points(std::string const & name)
{
#ifdef DEAL_II_WITH_HDF5
  auto dataset = hdf5_file->open_dataset(name);

  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    dealii::FullMatrix<point_value_type> evaluation_points(
      dataset.get_dimensions()[0],
      dataset.get_dimensions()[1],
      &pointwise_output_data.evaluation_points[0][0]);

    std::vector<hsize_t> hyperslab_offset = {0, 0};

    dataset.write_hyperslab(evaluation_points, hyperslab_offset, dataset.get_dimensions());
  }
  else
  {
    dataset.template write_none<point_value_type>();
  }
#else
  (void)name;
  AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with HDF5!"));
#endif
}

#ifdef DEAL_II_WITH_HDF5
template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::add_evaluation_points_dataset(
  dealii::HDF5::Group & group,
  std::string const &   name)
{
  std::vector<hsize_t> hdf5_point_dims{pointwise_output_data.evaluation_points.size(), dim};

  group.template create_dataset<point_value_type>(name, hdf5_point_dims);
}

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::add_time_dataset(dealii::HDF5::Group & group,
                                                            std::string const &   name)
{
  group.template create_dataset<Number>(name, std::vector<hsize_t>{1, 1 * n_out_samples});
}
#endif

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::write_time(double time)
{
#ifdef DEAL_II_WITH_HDF5
  auto dataset = hdf5_file->open_dataset("PhysicalInformation/Time");

  std::vector<hsize_t> hyperslab_offset = {0, counter};

  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    dataset.write_hyperslab(std::vector<Number>{static_cast<Number>(time)},
                            hyperslab_offset,
                            std::vector<hsize_t>{1, 1});
  }
  else
  {
    dataset.template write_none<Number>();
  }
#else
  (void)time;
  AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with HDF5!"));
#endif
}

template<int dim, typename Number>
void
PointwiseOutputGeneratorBase<dim, Number>::write_component(
  std::string const &            name,
  dealii::Vector<Number> const & componentwise_result)
{
#ifdef DEAL_II_WITH_HDF5
  auto dataset = hdf5_file->open_dataset("PhysicalInformation/" + name);

  std::vector<hsize_t> hyperslab_offset = {0, counter};
  std::vector<hsize_t> hyperslab_dim    = {pointwise_output_data.evaluation_points.size(), 1};

  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    dataset.write_hyperslab(componentwise_result, hyperslab_offset, hyperslab_dim);
  else
    dataset.template write_none<Number>();
#else
  (void)name;
  (void)componentwise_result;
  AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with HDF5!"));
#endif
}

template class PointwiseOutputGeneratorBase<2, float>;
template class PointwiseOutputGeneratorBase<2, double>;

template class PointwiseOutputGeneratorBase<3, float>;
template class PointwiseOutputGeneratorBase<3, double>;

} // namespace ExaDG
