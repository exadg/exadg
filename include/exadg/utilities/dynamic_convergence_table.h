#ifndef DYNAMIC_CONVERGENCE_TABLE
#define DYNAMIC_CONVERGENCE_TABLE

namespace ExaDG
{
class DynamicConvergenceTable
{
public:
  DynamicConvergenceTable(MPI_Comm const & comm) : mpi_comm(comm)
  {
    this->add_new_row();
  }

  void
  add_new_row()
  {
    vec.push_back(std::map<std::string, double>());
  }

  void
  put(std::string label, double value) const
  {
    auto & map = vec.back();
    auto   it  = map.find(label);
    if(it != map.end())
      it->second += value;
    else
      map[label] = value;
  }

  void
  set(std::string label, double value) const
  {
    auto & map = vec.back();
    auto   it  = map.find(label);
    if(it != map.end())
      it->second = value;
    else
      map[label] = value;
  }

  void
  print(FILE * f) const
  {
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    if(rank)
      return;

    std::vector<std::string> header;

    for(auto & map : vec)
      for(auto & it : map)
        if(std::find(header.begin(), header.end(), it.first) == header.end())
          header.push_back(it.first);

    std::sort(header.begin(), header.end());

    for(auto it : header)
      fprintf(f, "%12s", it.c_str());
    fprintf(f, "\n");

    for(auto & map : vec)
    {
      if(map.size() == 0)
        continue;
      for(auto h : header)
      {
        auto it = map.find(h);
        if(it == map.end())
          fprintf(f, "%12.4e", 0.0);
        else
          fprintf(f, "%12.4e", it->second);
      }
      fprintf(f, "\n");
    }
  }

  void
  print() const
  {
    this->print(stdout);
  }


  void
  print(std::string filename) const
  {
    FILE * f = fopen(filename.c_str(), "w");
    this->print(f);
    fclose(f);
  }

private:
  MPI_Comm const & mpi_comm;

  mutable std::vector<std::map<std::string, double>> vec;
};
} // namespace ExaDG

#endif
