#ifndef INCLUDE_EXADG_SAVE_TO_CSV_H 
#define INCLUDE_EXADG_SAVE_TO_CSV_H 

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/point.h>
#include <fstream>
#include <mutex>
#include <filesystem>

namespace ExaDG
{
template<int dim, typename Number>
class CSVWriter
{
  typedef dealii::VectorizedArray<Number>   scalar;
public:
  CSVWriter() : rank(0), time(0.0) {}

  void initialize(unsigned int rank_in)
  {
    rank = rank_in;
  }

  void set_time(double time_in)
  {
    time = time_in;
  }

  void
  write_to_file(std::string const & folder_name,
                std::string const & variable_name,
                scalar const & value,
                dealii::Point<dim, dealii::VectorizedArray<Number>> const & pnt)
  {
    std::ostringstream file_string;
    file_string << "output/"
             << folder_name
             << "/"
             << variable_name
             << "_rank_" << rank
             << "_time_" << time
             << ".csv";

    std::string filename = file_string.str();

    std::lock_guard<std::mutex> lock(mutex);

    std::filesystem::path path(filename);

    if(path.has_parent_path())
    {
      std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream file(filename, std::ios::app);
    AssertThrow(file.is_open(), dealii::ExcMessage("Error opening csv file"));

    for(unsigned int i=0; i<scalar::size(); ++i)
    {
      file << pnt[0][i] << "," << pnt[1][i] << "," << value[i] << "\n";
    }
    file.close();
  }
private:
  unsigned int rank;
  double time;
  mutable std::mutex mutex;
};
} // namepspace ExaDG

#endif
