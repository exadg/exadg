#ifndef INCLUDE_FUNCTION_WITH_NORMAL_H_
#define INCLUDE_FUNCTION_WITH_NORMAL_H_

/*
 * Class that extends the Function class of deal.II by the possibility of using normal vectors.
 */

using namespace dealii;

template<int dim>
class FunctionWithNormal : public Function<dim>
{
public:
  FunctionWithNormal(unsigned int const n_components, const double time)
    : Function<dim>(n_components, time)
  {
  }

  virtual ~FunctionWithNormal()
  {
  }

  void set_normal_vector(Tensor<1, dim> normal_vector_in)
  {
    normal_vector = normal_vector_in;
  }

  Tensor<1, dim>
  get_normal_vector() const
  {
    return normal_vector;
  }

private:
  Tensor<1, dim> normal_vector;
};

#endif /* INCLUDE_FUNCTION_WITH_NORMAL_H_ */
