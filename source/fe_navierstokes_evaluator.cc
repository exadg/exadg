
#include "fe_navierstokes_evaluator.h"

#include <deal.II/base/tensor.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>
#include <sstream>


using namespace dealii;

namespace helpers
{

  // Translation layer between the evaluation class of the Navier-Stokes
  // projection solver based on continuous finite elements that does not have
  // a template on the element degree and the actual evaluators that do have a
  // template parameter.
  template <int dim>
  class FENavierStokesEvaluator
  {
  public:
    virtual ~FENavierStokesEvaluator() {}

    virtual void do_advect(const parallel::distributed::Vector<double> &src,
                           parallel::distributed::Vector<double> &dst) const = 0;

    virtual void do_divergence(const parallel::distributed::Vector<double> &src,
                               parallel::distributed::Vector<double> &dst) const = 0;

    virtual void do_curl(const parallel::distributed::Vector<double> &src,
                         parallel::distributed::Vector<double> &dst) const = 0;
  };


  // implementation of actual code
  template <int dim, int u_degree>
  class FENavierStokesEvaluatorImpl : public FENavierStokesEvaluator<dim>
  {
  public:
    FENavierStokesEvaluatorImpl (const MatrixFree<dim> &matrix_free,
                                 const parallel::distributed::Vector<double> &pressure,
                                 const parallel::distributed::Vector<double> &last_pressure_update,
                                 const FluidBaseAlgorithm<dim> &fluid_algo)
      :
      matrix_free (matrix_free),
      pressure (pressure),
      last_pressure_update (last_pressure_update),
      fluid_algorithm (fluid_algo)
    {}

    virtual void do_advect(const parallel::distributed::Vector<double> &src,
                           parallel::distributed::Vector<double> &dst) const
    {
      matrix_free.cell_loop(&FENavierStokesEvaluatorImpl::local_advect,
                            this, dst, src);
    }

    virtual void do_divergence(const parallel::distributed::Vector<double> &src,
                               parallel::distributed::Vector<double> &dst) const
    {
      matrix_free.cell_loop(&FENavierStokesEvaluatorImpl::local_divergence,
                            this, dst, src);
    }

    virtual void do_curl(const parallel::distributed::Vector<double> &src,
                         parallel::distributed::Vector<double> &dst) const
    {
      matrix_free.cell_loop(&FENavierStokesEvaluatorImpl::local_vorticity,
                            this, dst, src);
    }

  private:
    const MatrixFree<dim>                       &matrix_free;
    const parallel::distributed::Vector<double> &pressure;
    const parallel::distributed::Vector<double> &last_pressure_update;
    const FluidBaseAlgorithm<dim>               &fluid_algorithm;

    void local_advect (const MatrixFree<dim>              &data,
                       parallel::distributed::Vector<double> &dst,
                       const parallel::distributed::Vector<double> &src,
                       const std::pair<unsigned int,unsigned int> &cell_range) const
    {
      const VectorizedArray<double> inv_time_step = make_vectorized_array(1./fluid_algorithm.get_time_step());
      FEEvaluation<dim,u_degree,u_degree+1,dim> phi_u(matrix_free, 0);
      FEEvaluation<dim,u_degree-1,u_degree+1> phi_p(matrix_free, 1);
      FEEvaluation<dim,u_degree-1,u_degree+1> phi_p_old(matrix_free, 1);

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi_u.reinit(cell);
          phi_p.reinit(cell);
          phi_p_old.reinit(cell);
          phi_u.read_dof_values(src);
          phi_p.read_dof_values(pressure);
          phi_p_old.read_dof_values(last_pressure_update);
          phi_u.evaluate(true, true);
          phi_p.evaluate(true, false);
          phi_p_old.evaluate(true, false);
          for (unsigned int q=0; q<phi_u.n_q_points; ++q)
            {
              Tensor<1,dim,VectorizedArray<double> > vel = phi_u.get_value(q);
              Tensor<2,dim,VectorizedArray<double> > grad = phi_u.get_gradient(q);
              VectorizedArray<double> pres = phi_p.get_value(q)+phi_p_old.get_value(q)*inv_time_step;
              VectorizedArray<double> div = grad[0][0];
              for (unsigned int d=1; d<dim; ++d)
                div += grad[d][d];

              Tensor<1,dim,VectorizedArray<double> >
                mom_val = grad * vel + 0.5 * div * vel;

              if (const TensorFunction<1,dim> * funct =
                  fluid_algorithm.get_body_force().get())
                {
                  Point<dim,VectorizedArray<double> > q_points = phi_u.quadrature_point(q);
                  for (unsigned int n=0; n<VectorizedArray<double>::n_array_elements; ++n)
                    {
                      Point<dim> q_point;
                      for (unsigned int d=0; d<dim; ++d)
                        q_point[d] = q_points[d][n];
                      Tensor<1,dim> value = funct->value(q_point);
                      for (unsigned int d=0; d<dim; ++d)
                        mom_val[d][n] -= value[d];
                    }
                }
              phi_u.submit_value(mom_val, q);

              // weak form (eps v, nu eps u) - (div v, p)
              for (unsigned int d=0; d<dim; ++d)
                for (unsigned int e=d+1; e<dim; ++e)
                  {
                    grad[d][e] = fluid_algorithm.get_viscosity() * (grad[d][e] + grad[e][d]);
                    grad[e][d] = grad[d][e];
                  }
              for (unsigned int d=0; d<dim; ++d)
                {
                  grad[d][d] *= make_vectorized_array(2.*fluid_algorithm.get_viscosity());
                  grad[d][d] -= pres;
                }

              // alternative form:
              // compute weak form (div v, nu div u) + (curl v, nu curl u) - (div v, p)
              // for (unsigned int d=0; d<dim; ++d)
              //   mom_grad[d][d] = viscosity * div - pres;
              // VectorizedArray<double> curl = grad[1][0] - grad[0][1];
              // mom_grad[1][0] = viscosity * curl;
              // mom_grad[0][1] = -viscosity * curl;
              // if (dim == 3)
              //   {
              //     curl = grad[2][1] - grad[1][2];
              //     mom_grad[2][1] = viscosity * curl;
              //     mom_grad[1][2] = -viscosity * curl;
              //     curl = grad[0][2] - grad[2][0];
              //     mom_grad[0][2] = viscosity * curl;
              //     mom_grad[2][0] = -viscosity * curl;
              //   }
              phi_u.submit_gradient(grad, q);
            }

          phi_u.integrate(true, true);
          phi_u.distribute_local_to_global(dst);
        }
    }

    void local_divergence(const MatrixFree<dim>                       &,
                          parallel::distributed::Vector<double>       &dst,
                          const parallel::distributed::Vector<double> &src,
                          const std::pair<unsigned int,unsigned int>  &cell_range) const
    {
      FEEvaluation<dim,u_degree,u_degree+1,dim> phi_u(matrix_free, 0);
      FEEvaluation<dim,u_degree-1,u_degree+1> phi_p(matrix_free, 1);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi_u.reinit(cell);
          phi_p.reinit(cell);
          phi_u.read_dof_values(src);
          phi_u.evaluate(false, true);
          for (unsigned int q=0; q<phi_u.n_q_points; ++q)
            phi_p.submit_value(-phi_u.get_divergence(q), q);
          phi_p.integrate(true, false);
          phi_p.distribute_local_to_global(dst);
        }
    }

    void local_vorticity(const MatrixFree<dim>                       &,
                         parallel::distributed::Vector<double>       &dst,
                         const parallel::distributed::Vector<double> &src,
                         const std::pair<unsigned int,unsigned int>  &cell_range) const
    {
      FEEvaluation<dim,u_degree,u_degree+1,dim> phi_u(matrix_free, 0);
      Tensor<1,dim==2?1:dim,VectorizedArray<double> > curl;
      Tensor<1,dim,VectorizedArray<double> > curl_submit;
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          phi_u.reinit(cell);
          phi_u.read_dof_values(src);
          phi_u.evaluate(false, true);
          for (unsigned int q=0; q<phi_u.n_q_points; ++q)
            {
              curl = phi_u.get_curl(q);
              for (unsigned int d=0; d<(dim==2?1:dim); ++d)
                curl_submit[d] = curl[d];
              phi_u.submit_value(curl_submit, q);
            }
          phi_u.integrate(true, false);
          phi_u.distribute_local_to_global(dst);
        }
    }
  };
}



template <int dim>
FENavierStokesEvaluator<dim>
::FENavierStokesEvaluator(const MatrixFree<dim> &matrix_free,
                          const parallel::distributed::Vector<double> &pressure,
                          const parallel::distributed::Vector<double> &last_pressure_update,
                          const FluidBaseAlgorithm<dim> &fluid_algorithm)
{
  const unsigned int degree_u = matrix_free.get_dof_handler(0).get_fe().degree;
  if (degree_u == 2)
    evaluator.reset(new helpers::FENavierStokesEvaluatorImpl<dim,2>
                    (matrix_free, pressure, last_pressure_update, fluid_algorithm));
  else if (degree_u == 3)
    evaluator.reset(new helpers::FENavierStokesEvaluatorImpl<dim,3>
                    (matrix_free, pressure, last_pressure_update, fluid_algorithm));
  else if (degree_u == 4)
    evaluator.reset(new helpers::FENavierStokesEvaluatorImpl<dim,4>
                    (matrix_free, pressure, last_pressure_update, fluid_algorithm));
  else if (degree_u == 5)
    evaluator.reset(new helpers::FENavierStokesEvaluatorImpl<dim,5>
                    (matrix_free, pressure, last_pressure_update, fluid_algorithm));
  else if (degree_u == 6)
    evaluator.reset(new helpers::FENavierStokesEvaluatorImpl<dim,6>
                    (matrix_free, pressure, last_pressure_update, fluid_algorithm));
  else if (degree_u == 7)
    evaluator.reset(new helpers::FENavierStokesEvaluatorImpl<dim,7>
                    (matrix_free, pressure, last_pressure_update, fluid_algorithm));
  else if (degree_u == 8)
    evaluator.reset(new helpers::FENavierStokesEvaluatorImpl<dim,8>
                    (matrix_free, pressure, last_pressure_update, fluid_algorithm));
  else
    AssertThrow(false, ExcMessage("Only velocity degrees 2 up to 8 are implemented!"));
}



template <int dim>
void FENavierStokesEvaluator<dim>
::advection_integrals(const parallel::distributed::Vector<double> &src,
                      parallel::distributed::Vector<double> &dst) const
{
  evaluator->do_advect(src, dst);
}



template <int dim>
void FENavierStokesEvaluator<dim>
::divergence_integrals(const parallel::distributed::Vector<double> &src,
                       parallel::distributed::Vector<double> &dst) const
{
  evaluator->do_divergence(src, dst);
}



template <int dim>
void FENavierStokesEvaluator<dim>
::curl_integrals(const parallel::distributed::Vector<double> &src,
                 parallel::distributed::Vector<double> &dst) const
{
  evaluator->do_curl(src, dst);
}


// explicit instantiations
template class FENavierStokesEvaluator<2>;
template class FENavierStokesEvaluator<3>;
