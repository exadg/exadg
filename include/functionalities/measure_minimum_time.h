#ifndef MEASURE_MINIMUM_TIME
#define MEASURE_MINIMUM_TIME

class MeasureMinimumTime
{
public:
  template<typename Function>
  static void
  basic(int best_of, ConvergenceTable & convergence_table, std::string label, Function f)
  {
    MeasureMinimumTime::repeat(best_of, convergence_table, label, "", f);
  }

private:
  template<typename Function>
  static void
  repeat(int                best_of,
         ConvergenceTable & convergence_table,
         std::string        label,
         std::string        likwid_suffix,
         Function           f,
         MPI_Comm const &   mpi_comm)
  {
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);
    Timer  time;
    double min_time = std::numeric_limits<double>::max();
    double max_time = 0.0;
    double sum_time = 0.0;
    for(int i = 0; i < best_of; i++)
    {
      MPI_Barrier(mpi_comm);
#ifdef LIKWID_PERFMON
      std::string likwid_label = label + likwid_suffix;
      LIKWID_MARKER_START(likwid_label.c_str());
#else
      (void)likwid_suffix;
#endif
      time.restart();
      f();
      MPI_Barrier(mpi_comm);
      double temp = time.wall_time();
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(likwid_label.c_str());
#endif
      double temp_global;
      MPI_Reduce(&temp, &temp_global, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
      min_time = std::min(min_time, temp_global);
      max_time = std::max(max_time, temp_global);
      sum_time += temp;
    }
    convergence_table.add_value(label + "_min", min_time);
    convergence_table.set_scientific(label + "_min", true);
    convergence_table.add_value(label + "_max", max_time);
    convergence_table.set_scientific(label + "_max", true);
    convergence_table.add_value(label + "_ave", sum_time / best_of);
    convergence_table.set_scientific(label + "_ave", true);
  }
};


#endif
