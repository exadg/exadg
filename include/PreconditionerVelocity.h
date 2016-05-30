/*
 * PreconditionerVelocity.h
 *
 *  Created on: May 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRECONDITIONERVELOCITY_H_
#define INCLUDE_PRECONDITIONERVELOCITY_H_


class PreconditionerVelocityBase
{
public:
  virtual ~PreconditionerVelocityBase(){}

  virtual void vmult(parallel::distributed::BlockVector<double>        &dst,
                     const parallel::distributed::BlockVector<double>  &src) const = 0;
};

template<int dim, int fe_degree, typename value_type>
class InverseMassMatrixPreconditionerVelocity : public PreconditionerVelocityBase
{
public:
  InverseMassMatrixPreconditionerVelocity(MatrixFree<dim,value_type> const &mf_data,
                                          const unsigned int dof_index,
                                          const unsigned int quad_index)
  {
    inverse_mass_matrix_operator.initialize(mf_data,dof_index,quad_index);
  }

  virtual void vmult (parallel::distributed::BlockVector<double>        &dst,
                      const parallel::distributed::BlockVector<double>  &src) const
  {
    inverse_mass_matrix_operator.apply_inverse_mass_matrix(src,dst);
  }

  InverseMassMatrixOperator<dim,fe_degree,value_type> inverse_mass_matrix_operator;
};


template<int dim, typename value_type, typename Operator>
class JacobiPreconditionerVelocity : public PreconditionerVelocityBase
{
public:
  JacobiPreconditionerVelocity(MatrixFree<dim,value_type> const &mf_data,
                               const unsigned int dof_index,
                               Operator const &underlying_operator)
    :
    inverse_diagonal(dim)
  {
    mf_data.initialize_dof_vector(inverse_diagonal.block(0),dof_index);
    for(unsigned int d=1;d<dim;++d)
      inverse_diagonal.block(d) = inverse_diagonal.block(0);
    inverse_diagonal.collect_sizes();

    underlying_operator.calculate_inverse_diagonal(inverse_diagonal);
  }

  void vmult (parallel::distributed::BlockVector<value_type>        &dst,
              const parallel::distributed::BlockVector<value_type>  &src) const
  {
    for(unsigned int d=0;d<dim;++d)
    {
      for (unsigned int i=0;i<src.block(d).local_size();++i)
      {
        dst.block(d).local_element(i) = inverse_diagonal.block(d).local_element(i)*src.block(d).local_element(i);
      }
    }
  }

  void recalculate_diagonal(Operator const &underlying_operator)
  {
    underlying_operator.calculate_inverse_diagonal(inverse_diagonal);
  }
private:
  parallel::distributed::BlockVector<double> inverse_diagonal;
};


#endif /* INCLUDE_PRECONDITIONERVELOCITY_H_ */
