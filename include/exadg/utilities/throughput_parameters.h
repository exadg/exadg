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

#ifndef INCLUDE_EXADG_UTILITIES_THROUGHPUT_PARAMETERS_H_
#define INCLUDE_EXADG_UTILITIES_THROUGHPUT_PARAMETERS_H_

// likwid
#ifdef EXADG_WITH_LIKWID
#  include <likwid.h>
#endif

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG
#include <exadg/utilities/enum_patterns.h>
#include <exadg/utilities/print_solver_results.h>


namespace ExaDG
{
inline double
measure_operator_evaluation_time(std::function<void(void)> const & evaluate_operator,
                                 unsigned int const                degree,
                                 unsigned int const                n_repetitions_inner,
                                 unsigned int const                n_repetitions_outer,
                                 MPI_Comm const &                  mpi_comm)
{
  (void)degree;

  dealii::Timer global_timer;
  global_timer.restart();
  dealii::Utilities::MPI::MinMaxAvg global_time;

  double wall_time = std::numeric_limits<double>::max();

  do
  {
    for(unsigned int i_outer = 0; i_outer < n_repetitions_outer; ++i_outer)
    {
      dealii::Timer timer;
      timer.restart();

#ifdef EXADG_WITH_LIKWID
      LIKWID_MARKER_START(("degree_" + std::to_string(degree)).c_str());
#endif

      // apply matrix-vector product several times
      for(unsigned int i = 0; i < n_repetitions_inner; ++i)
      {
        evaluate_operator();
      }

#ifdef EXADG_WITH_LIKWID
      LIKWID_MARKER_STOP(("degree_" + std::to_string(degree)).c_str());
#endif

      MPI_Barrier(mpi_comm);
      dealii::Utilities::MPI::MinMaxAvg wall_time_inner =
        dealii::Utilities::MPI::min_max_avg(timer.wall_time(), mpi_comm);

      wall_time = std::min(wall_time, wall_time_inner.avg / (double)n_repetitions_inner);
    }

    global_time = dealii::Utilities::MPI::min_max_avg(global_timer.wall_time(), mpi_comm);
  } while(global_time.avg < 1.0 /*wall time in seconds*/);

  return wall_time;
}

template<typename EnumOperatorType>
struct ThroughputParameters
{
  ThroughputParameters()
  {
  }

  ThroughputParameters(std::string const & input_file)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    prm.enter_subsection("Throughput");
    {
      prm.add_parameter("EnumOperatorType",
                        operator_type,
                        "Operator type.",
                        Patterns::Enum<EnumOperatorType>(),
                        true);
      prm.add_parameter("RepetitionsInner",
                        n_repetitions_inner,
                        "Number of operator evaluations.",
                        dealii::Patterns::Integer(1),
                        true);
      prm.add_parameter("RepetitionsOuter",
                        n_repetitions_outer,
                        "Number of runs (taking minimum wall time).",
                        dealii::Patterns::Integer(1, 10),
                        true);
    }
    prm.leave_subsection();
  }

  void
  print_results(MPI_Comm const & mpi_comm)
  {
    print_throughput(wall_times, ExaDG::Utilities::enum_to_string(operator_type), mpi_comm);
  }

  // type of PDE operator
  EnumOperatorType operator_type = Utilities::default_constructor<EnumOperatorType>();

  // number of repetitions used to determine the average/minimum wall time required
  // to compute the matrix-vector product
  unsigned int n_repetitions_inner = 100; // take the average of inner repetitions
  unsigned int n_repetitions_outer = 1;   // take the minimum of outer repetitions

  // variable used to store the wall times for different polynomial degrees and problem sizes
  mutable std::vector<std::tuple<unsigned int, dealii::types::global_dof_index, double>> wall_times;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_UTILITIES_THROUGHPUT_PARAMETERS_H_ */
