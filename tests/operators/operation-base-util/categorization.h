#ifndef OPERATOR_BASE_CATEGORIZATION_H
#define OPERATOR_BASE_CATEGORIZATION_H

class Categorization
{
public:
  template<int dim, typename AdditionalData>
  static void
  do_cell_based_loops(const parallel::distributed::Triangulation<dim> & tria,
                      AdditionalData &  data,
                      const unsigned int level  = numbers::invalid_unsigned_int)
  {
      
    bool is_mg = level != numbers::invalid_unsigned_int;

    int cell_first = -1;
    int cell_local = +0;
      
    // ... create list for the category of each cell
    if (is_mg) 
    {
      for (auto cell = tria.begin(level); cell != tria.end(level); ++cell) 
        if (cell->is_locally_owned_on_level()) 
        {
          if(cell_first == -1)
            cell_first = cell->index(); // index of first local cell
          cell_local++; // count number of local cells
        }
      data.cell_vectorization_category.resize(cell_local);
    }
    else 
      data.cell_vectorization_category.resize(tria.n_active_cells());

    // ... setup scaling factor
    std::vector<unsigned int> factors(dim * 2);


    std::map<unsigned int, unsigned int> bid_map;
    for(unsigned int i = 0; i < tria.get_boundary_ids().size(); i++)
      bid_map[tria.get_boundary_ids()[i]] = i + 1;

    {
      unsigned int bids   = tria.get_boundary_ids().size() + 1;
      int          offset = 1;
      for(unsigned int i = 0; i < dim * 2; i++, offset = offset * bids)
        factors[i] = offset;
    }

    if(!is_mg)
      for(auto cell = tria.begin_active(); cell != tria.end(); ++cell)
      {
        // accumulator for category of this cell: start with 0
        unsigned int c_num = 0;
        if(cell->is_locally_owned())
          // loop over all faces
          for(unsigned int i = 0; i < dim * 2; i++)
          {
            auto & face = *cell->face(i);
            if(face.at_boundary())
              // and update accumulator if on boundary
              c_num += factors[i] * bid_map[face.boundary_id()];
          }
        // save the category of this cell
        data.cell_vectorization_category[cell->active_cell_index()] = c_num;
      }
    else
      for(auto cell = tria.begin(level); cell != tria.end(level); ++cell)
      {
        // accumulator for category of this cell: start with 0
        if(cell->is_locally_owned_on_level())
        {
        unsigned int c_num = 0;
            
          // loop over all faces
          for(unsigned int i = 0; i < dim * 2; i++)
          {
            auto & face = *cell->face(i);
            if(face.at_boundary())
              // and update accumulator if on boundary
              c_num += factors[i] * bid_map[face.boundary_id()];
          }
        // save the category of this cell
        data.cell_vectorization_category[cell->index() - cell_first] = c_num;
        }
      }

    // ... finalize setup of matrix_free
    data.hold_all_faces_to_owned_cells = true;
    data.cell_vectorization_categories_strict = true;
    data.mapping_update_flags_faces_by_cells =
      data.mapping_update_flags_inner_faces | data.mapping_update_flags_boundary_faces;
  }
};

#endif