#ifndef LUNG_METIS
#define LUNG_METIS

#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
class PartMetis
{
public:
  std::vector<int> local;
  std::vector<int> ghost;
};

class Graph
{
public:
  void
  to_file(std::string filename)
  {
    FILE * pFile;
    pFile = fopen(filename.c_str(), "w");
    for(auto i : vtxdist)
      fprintf(pFile, "%2d ", i);
    fprintf(pFile, "\n");

    for(auto i : xadj)
      fprintf(pFile, "%2d ", i);
    fprintf(pFile, "\n");

    for(auto i : adjncy)
      fprintf(pFile, "%2d ", i);
    fprintf(pFile, "\n");

    for(auto i : parts)
      fprintf(pFile, "%2d ", i);
    fprintf(pFile, "\n");

    for(unsigned int i = 0; i < parts_rank.size(); i++)
    {
      auto & temp = parts_rank[i];

      fprintf(pFile, "  l: ");
      for(auto i : temp.local)
        fprintf(pFile, "%2d ", i);

      fprintf(pFile, "  g: ");
      for(auto i : temp.ghost)
        fprintf(pFile, "%2d ", i);

      fprintf(pFile, "\n");
    }
    fprintf(pFile, "\n");

    fprintf(pFile, "\n\n");
    fclose(pFile);
  }

  std::vector<int> xadj;
  std::vector<int> adjncy;
  std::vector<int> vtxdist;
  std::vector<int> parts;

  std::vector<PartMetis> parts_rank;
};

template<int dim>
void
create_hierchy(unsigned int         levels,
               int                  partitions,
               Graph &              graph_in,
               std::vector<Graph> & graphs_out)
{
  for(int p = 0; p < partitions; p++)
  {
    PartMetis p_temp;

    // collect local cells
    for(unsigned int j = 0; j < graph_in.parts.size(); j++)
      if(p == graph_in.parts[j])
        p_temp.local.push_back(j);

    // U(neighbors) for all local cells
    std::map<unsigned int, int> map;
    for(auto j : p_temp.local)
      for(int k = graph_in.xadj[j]; k < graph_in.xadj[j + 1]; k++)
        map[graph_in.adjncy[k]] = 0;

    // extract ghost cells
    for(auto i : map)
      if(p != graph_in.parts[i.first])
        p_temp.ghost.push_back(i.first);

    graph_in.parts_rank.push_back(p_temp);
  }

  graphs_out.push_back(graph_in);

  const unsigned int stride = std::pow(2, dim);

  for(unsigned int i = 0; i < levels; i++)
  {
    Graph   temp;
    Graph & in = graphs_out[i];

    // dummy operation: since we only consider METIS here
    temp.vtxdist.push_back(0);
    temp.vtxdist.push_back(in.vtxdist.back() / stride);

    // initialization of starting point
    temp.xadj.push_back(0);

    // loop over all new nodes and insert its info into the coarse CSR
    for(unsigned int i = 0; i < in.parts.size(); i += stride)
    {
      // collect all neighbors of children in a map such that they are sorted and unique
      std::map<int, int> map;
      for(int j = in.xadj[i]; j < in.xadj[i + stride]; j++)
        if(i / stride != in.adjncy[j] / stride) // ignore self-loops
          map[in.adjncy[j] / stride] = 0;

      // save all sorted and unique neighbors
      for(auto j : map)
        temp.adjncy.push_back(j.first);

      // save index of end
      temp.xadj.push_back(temp.adjncy.size());

      // save rank of first child
      temp.parts.push_back(in.parts[i]);
    }

    // loop over all partitions on this level and collect local and ghost cells
    for(int p = 0; p < partitions; p++)
    {
      // create new struct
      PartMetis p_temp;

      // collect local cells
      for(unsigned int j = 0; j < temp.parts.size(); j++)
        if(p == temp.parts[j]) // it is local
          p_temp.local.push_back(j);

      // collect neighbors of all local cells
      std::map<unsigned int, int> map;
      for(auto j : p_temp.local)
        for(int k = temp.xadj[j]; k < temp.xadj[j + 1]; k++)
          map[temp.adjncy[k]] = 0;

      // make sure that also the parents of the local cells of finer level ...
      if(i > 0)
        for(auto k : graphs_out[i].parts_rank[p].ghost)
          map[k / stride] = 0;
      // ... and also the ghost cells are included
      if(i > 0)
        for(auto k : graphs_out[i].parts_rank[p].local)
          map[k / stride] = 0;

      // ... save only non-local cells
      for(auto j : map)
        if(temp.parts[j.first] != p)
          p_temp.ghost.push_back(j.first);

      // save struct
      temp.parts_rank.push_back(p_temp);
    }

    graphs_out.push_back(temp);
  }
}

template<int dim>
void partition(Triangulation<3> & tria)
{
  int      actual_size = 20;
  int      size        = 1;
  MPI_Comm comm        = MPI_COMM_SELF;
  int      rank        = 0;

  FE_DGQ<dim> fe(0);

  // create dof_handler
  DoFHandler<dim> dofhanlder(tria);
  dofhanlder.distribute_dofs(fe);

  Graph graph_in;

  int                active_cells = 0;
  std::vector<int> & xadj         = graph_in.xadj;
  xadj.push_back(0);

  std::vector<int> & adjncy  = graph_in.adjncy;
  int                counter = 0;

  for(auto cell : dofhanlder.active_cell_iterators())
  {
    if(!cell->is_locally_owned())
      continue;
    active_cells++;
    for(int i = 0; i < dim * 2; i++)
    {
      if(cell->neighbor_index(i) != -1)
      {
        std::vector<types::global_dof_index> dof_indices(1);
        cell->neighbor(i)->get_dof_indices(dof_indices);
        adjncy.push_back(dof_indices[0]);
        counter++;
      }
    }
    xadj.push_back(counter);
  }

  std::vector<int> & vtxdist = graph_in.vtxdist;
  vtxdist.resize(size + 1);
  std::vector<int> vtxdist_temp_(size);

  std::vector<int> & parts = graph_in.parts;
  parts.resize(active_cells);

  MPI_Allgather(&active_cells, 1, MPI_INT, &vtxdist_temp_[0], 1, MPI_INT, comm);
  vtxdist[0] = 0;
  for(int i = 0; i < size; i++)
    vtxdist[i + 1] = vtxdist_temp_[i] + vtxdist[i];


  idx_t ncon, nparts;

  ncon   = 1;
  nparts = actual_size;

  idx_t edgecut;


  METIS_PartGraphKway(&vtxdist[size],
                      &ncon,
                      &xadj[0],
                      &adjncy[0],
                      NULL,
                      NULL,
                      NULL,
                      &nparts,
                      NULL,
                      NULL,
                      NULL,
                      &edgecut,
                      &parts[0]);

  IndexSet is(vtxdist[rank + 1]);
  is.add_range(vtxdist[rank], vtxdist[rank + 1]);
  LinearAlgebra::distributed::Vector<double> solution(is, comm);


  for(int i = 0; i < vtxdist[rank + 1] - vtxdist[rank]; i++)
    solution.local_element(i) = parts[i];

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dofhanlder);

    data_out.add_data_vector(solution, "solution", DataOut<dim>::DataVectorType::type_cell_data);

    //        auto ranks = solution;
    //        ranks = rank;
    //        data_out.add_data_vector(ranks, "ranks");

    data_out.build_patches(1);

    data_out.write_vtu_in_parallel("temp.vtu", comm);
  }
}

} // namespace ExaDG


#endif
