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

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_BASE_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_BASE_H_


// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>

#ifdef DEAL_II_WITH_HDF5
#  include <deal.II/base/hdf5.h>
#endif

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/postprocessor/time_control.h>
#include <exadg/utilities/tensor_utilities.h>


namespace ExaDG
{
template<int dim>
struct PointwiseOutputDataBase
{
  using point_value_type = typename dealii::Point<dim>::value_type;

  PointwiseOutputDataBase();

  TimeControlData time_control_data;

  std::string directory;
  std::string filename;

  bool update_points_before_evaluation;

  std::vector<dealii::Point<dim>> evaluation_points;

  void
  print(dealii::ConditionalOStream & pcout) const;
};

template<int dim, typename Number>
class PointwiseOutputGeneratorBase
{
public:
  using point_value_type = typename PointwiseOutputDataBase<dim>::point_value_type;

  TimeControl time_control;

protected:
  void
  do_evaluate(std::function<void()> const & write_solution, double const time, bool const unsteady);

  PointwiseOutputGeneratorBase(MPI_Comm const & comm);

  virtual ~PointwiseOutputGeneratorBase() = default;

  void
  setup_base(dealii::Triangulation<dim> const &   triangulation_in,
             dealii::Mapping<dim> const &         mapping_in,
             PointwiseOutputDataBase<dim> const & pointwise_output_data_in);

  void
  add_quantity(std::string const & name, unsigned int const n_components);

  template<int n_components>
  void
  write_quantity(std::string const &                                          name,
                 std::vector<dealii::Tensor<1, n_components, Number>> const & values,
                 unsigned int const                                           first_component)
  {
    unsigned int const components = name_to_components.at(name);
    for(unsigned int comp = 0; comp < components; ++comp)
    {
      extract_component_from_tensors(componentwise_result, values, first_component + comp);
      write_component((components == 1) ? name : name + std::to_string(comp), componentwise_result);
    }
  }

  void
  write_quantity(std::string const & name, std::vector<Number> const & values)
  {
    write_component(name, values);
  }

  template<int n_components>
  [[nodiscard]] std::vector<
    typename dealii::FEPointEvaluation<n_components, dim, dim, Number>::value_type>
  compute_point_values(dealii::LinearAlgebra::distributed::Vector<Number> const & solution,
                       dealii::DoFHandler<dim> const &                            dof_handler) const
  {
    return dealii::VectorTools::point_values<n_components>(*remote_evaluator,
                                                           dof_handler,
                                                           solution);
  }

private:
  void
  setup_remote_evaluator();

  void
  reinit_remote_evaluator();

  void
  create_hdf5_file();

  void
  write_evaluation_points(std::string const & name);

#ifdef DEAL_II_WITH_HDF5
  void
  add_evaluation_points_dataset(dealii::HDF5::Group & group, std::string const & name);

  void
  add_time_dataset(dealii::HDF5::Group & group, std::string const & name);
#endif

  void
  write_time(double time);

  template<typename ComponentwiseContainerType>
  void
  write_component(std::string const & name, ComponentwiseContainerType const & componentwise_result)
  {
#ifdef DEAL_II_WITH_HDF5
    auto dataset = hdf5_file->open_dataset("PhysicalInformation/" + name);

    std::vector<hsize_t> hyperslab_offset = {0, time_control.get_counter()};
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

  MPI_Comm const mpi_comm;

  dealii::SmartPointer<dealii::Triangulation<dim> const>              triangulation;
  dealii::SmartPointer<dealii::Mapping<dim> const>                    mapping;
  PointwiseOutputDataBase<dim>                                        pointwise_output_data;
  dealii::Vector<Number>                                              componentwise_result;
  unsigned int                                                        n_out_samples;
  std::shared_ptr<dealii::Utilities::MPI::RemotePointEvaluation<dim>> remote_evaluator;
  bool                                                                first_evaluation;

  std::map<std::string, unsigned int> name_to_components;

#ifdef DEAL_II_WITH_HDF5
  std::unique_ptr<dealii::HDF5::File> hdf5_file;
#endif
};

} // namespace ExaDG

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POINTWISE_OUTPUT_GENERATOR_BASE_H_*/
