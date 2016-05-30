/*
 * InverseMassMatrix.h
 *
 *  Created on: May 12, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INVERSEMASSMATRIX_H_
#define INCLUDE_INVERSEMASSMATRIX_H_

// Collect all data for the inverse mass matrix operation in a struct in order to avoid allocating the memory repeatedly.
template <int dim, int fe_degree, typename Number>
struct InverseMassMatrixData
{
  InverseMassMatrixData(const MatrixFree<dim,Number> &data,
                        const unsigned int fe_index = 0,
                        const unsigned int quad_index = 0)
    :
    fe_eval(1, FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number>(data,fe_index,quad_index)),
    coefficients(FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number>::n_q_points),
    inverse(fe_eval[0])
  {}

  // Manually implement the copy operator because CellwiseInverseMassMatrix must point to the object 'fe_eval'
  InverseMassMatrixData(const InverseMassMatrixData &other)
    :
    fe_eval(other.fe_eval),
    coefficients(other.coefficients),
    inverse(fe_eval[0])
  {}

  // For memory alignment reasons, need to place the FEEvaluation object into an aligned vector
  AlignedVector<FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number> > fe_eval;
  AlignedVector<VectorizedArray<Number> > coefficients;
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim,Number> inverse;
};

template <int dim, int fe_degree, typename value_type>
class InverseMassMatrixOperator
{
public:
  InverseMassMatrixOperator()
    :
    matrix_free_data(nullptr)
  {}

  void initialize(MatrixFree<dim,value_type> const &mf_data,
                  const unsigned int dof_index,
                  const unsigned int quad_index)
  {
    this->matrix_free_data = &mf_data;

    // generate initial mass matrix data to avoid allocating it over and over again
    mass_matrix_data.reset(new Threads::ThreadLocalStorage<InverseMassMatrixData<dim,fe_degree,value_type> >
            (InverseMassMatrixData<dim,fe_degree,value_type>(*matrix_free_data, dof_index, quad_index)));
  }

  void apply_inverse_mass_matrix (const parallel::distributed::BlockVector<value_type>  &src,
                                  parallel::distributed::BlockVector<value_type>        &dst) const
  {
    /*
     * the function "local_apply_inverse_mass_matrix" uses
     *    fe_eval.set_dof_values(dst,0);
     * which overwrites dof values of dst, but not the ghosts elements that are subsequently added to the vector
     *
     * -> ensure that ghost elements are set to zero before calling the cell_loop
     */

    /* Solution 1:
     * use "dst = 0;" to set all elements (including the ghost elements) to zero
     * However: apply_inverse_mass_matirx may be called with two identical parameters:
     *    apply_inverse_mass_matrix(vector,vector);
     * In that case, "dst = 0;" will also overwrite src, which has to be avoided
     */
//    if(&dst!=&src)
//      dst = 0;

    /* Solution 2:
     * set ghost elements of dst to zero
     */
    dst.zero_out_ghosts();

    matrix_free_data->cell_loop(&InverseMassMatrixOperator<dim,fe_degree,value_type>::local_apply_inverse_mass_matrix, this, dst, src);
  }

  void apply_inverse_mass_matrix (const std::vector<parallel::distributed::Vector<value_type> >  &src,
                                  std::vector<parallel::distributed::Vector<value_type> >        &dst) const
  {
    matrix_free_data->cell_loop(&InverseMassMatrixOperator<dim,fe_degree,value_type>::local_apply_inverse_mass_matrix, this, dst, src);
  }

private:
  MatrixFree<dim,value_type> const * matrix_free_data;
  mutable std_cxx11::shared_ptr<Threads::ThreadLocalStorage<InverseMassMatrixData<dim,fe_degree,value_type> > > mass_matrix_data;

  void local_apply_inverse_mass_matrix (const MatrixFree<dim,value_type>                      &,
                                        parallel::distributed::BlockVector<value_type>        &dst,
                                        const parallel::distributed::BlockVector<value_type>  &src,
                                        const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    InverseMassMatrixData<dim,fe_degree,value_type>& mass_data = mass_matrix_data->get();

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      mass_data.fe_eval[0].reinit(cell);
      mass_data.fe_eval[0].read_dof_values(src, 0);

      mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
      mass_data.inverse.apply(mass_data.coefficients, dim,
                              mass_data.fe_eval[0].begin_dof_values(),
                              mass_data.fe_eval[0].begin_dof_values());

      mass_data.fe_eval[0].set_dof_values(dst,0);
    }
  }

  void local_apply_inverse_mass_matrix (const MatrixFree<dim,value_type>                              &,
                                        std::vector<parallel::distributed::Vector<value_type> >       &dst,
                                        const std::vector<parallel::distributed::Vector<value_type> > &src,
                                        const std::pair<unsigned int,unsigned int>                    &cell_range) const
  {
    InverseMassMatrixData<dim,fe_degree,value_type>& mass_data = mass_matrix_data->get();

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      mass_data.fe_eval[0].reinit(cell);
      mass_data.fe_eval[0].read_dof_values(src, 0);

      mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
      mass_data.inverse.apply(mass_data.coefficients, dim,
                              mass_data.fe_eval[0].begin_dof_values(),
                              mass_data.fe_eval[0].begin_dof_values());

      mass_data.fe_eval[0].set_dof_values(dst,0);
    }
  }
};

#endif /* INCLUDE_INVERSEMASSMATRIX_H_ */
