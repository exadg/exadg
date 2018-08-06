#ifndef OPERATOR_BASE_CATEGORIZATION_H
#define OPERATOR_BASE_CATEGORIZATION_H

class Categorization
{
public:
  template<int dim>
  static void
  do_cell_based_loops(parallel::distributed::Triangulation<dim> & tria,
                      typename MatrixFree<dim>::AdditionalData &  data)
  {
    // ... create list for the category of each cell
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

    // ... finalize setup of matrix_free
    data.cell_vectorization_categories_strict = true;
    data.mapping_update_flags_faces_by_cells =
      (update_JxW_values | update_normal_vectors | update_quadrature_points | update_values);
  }
};

#endif