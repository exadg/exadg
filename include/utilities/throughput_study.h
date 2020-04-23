/*
 * throughput_study.h
 *
 *  Created on: 24.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_UTILITIES_THROUGHPUT_STUDY_H_
#define INCLUDE_UTILITIES_THROUGHPUT_STUDY_H_

#include "../utilities/print_throughput.h"

struct ThroughputStudy
{
  ThroughputStudy()
  {
  }

  ThroughputStudy(const std::string & input_file)
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm);

    parse_input(input_file, prm, true, true);
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Throughput");
      prm.add_parameter("OperatorType",     operator_type,        "Type of operator.");
      prm.add_parameter("RepetitionsInner", n_repetitions_inner,  "Number of operator evaluations.",               Patterns::Integer(1));
      prm.add_parameter("RepetitionsOuter", n_repetitions_outer,  "Number of runs (taking minimum wall time).",    Patterns::Integer(1,10));
    prm.leave_subsection();
    // clang-format on
  }

  void
  print_results(MPI_Comm const & mpi_comm)
  {
    print_throughput(wall_times, operator_type, mpi_comm);
  }

  std::string operator_type = "Undefined";

  // number of repetitions used to determine the average/minimum wall time required
  // to compute the matrix-vector product
  unsigned int n_repetitions_inner = 100; // take the average of inner repetitions
  unsigned int n_repetitions_outer = 1;   // take the minimum of outer repetitions

  // global variable used to store the wall times for different polynomial degrees and problem sizes
  mutable std::vector<std::tuple<unsigned int, types::global_dof_index, double>> wall_times;
};



#endif /* INCLUDE_UTILITIES_THROUGHPUT_STUDY_H_ */
