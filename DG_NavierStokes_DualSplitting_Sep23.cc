//---------------------------------------------------------------------------
//    $Id: program.cc 56 2015-02-06 13:05:10Z kronbichler $
//    Version: $Name$
//
//    Copyright (C) 2013 - 2015 by Katharina Kormann and Martin Kronbichler
//
//---------------------------------------------------------------------------

// program based on step-37 but implementing interior penalty DG (currently
// without multigrid)

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/integrators/laplace.h>

#include <fstream>
#include <sstream>

namespace DG_NavierStokes
{
  using namespace dealii;

  const unsigned int fe_degree = 2;
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int dimension = 2; // dimension >= 2
  const unsigned int refine_steps_min = 3;
  const unsigned int refine_steps_max = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0; // Poisseuille 5.0;  Kovasznay 1.0
  const double OUTPUT_INTERVAL_TIME = 1.0;
  const double CFL = 0.05;

  // flow past cylinder
  const double Um = 0.3;//(dimension == 2 ? 1.5 : 2.25);
  const double D = 0.1;

  const double VISCOSITY = 0.025; // Taylor vortex: 0.01; vortex problem (Hesthaven): 0.025; Poisseuille 0.005; Kovasznay 0.025; Stokes 1.0; flow past cylinder 0.001
  const double MAX_VELOCITY = 1.4;//Um; // Taylor vortex: 1; vortex problem (Hesthaven): 1.4; Poisseuille 1.0; Kovasznay 3.6; flow past cylinder Um
  const double stab_factor = 6.0; //flow past cylinder 4

  bool pure_dirichlet_bc = false;

  const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);

  template<int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    AnalyticalSolution (const unsigned int  component,
                        const double    time = 0.) : Function<dim>(1, time),component(component) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double t = this->get_time();
    double result = 0.0;

    /*********************** cavitiy flow *******************************/
//    const double T = 0.1;
//    if(component == 0 && (std::abs(p[1]-1.0)<1.0e-15))
//      result = t<T? (t/T) : 1.0;
    /********************************************************************/

    /*********************** Cuette flow problem ************************/
    // stationary
    /*  if(component == 0)
            result = ((p[1]+1.0)*0.5); */

    // instationary
    /* const double T = 1.0;
     if(component == 0)
      result = ((p[1]+1.0)*0.5)*(t<T? (t/T) : 1.0); */
    /********************************************************************/

    /****************** Poisseuille flow problem ************************/
    // constant velocity profile at inflow
    /* const double pressure_gradient = -0.01;
     double T = 0.5;
     if(component == 0 && (std::abs(p[0]+1.0)<1.0e-12))
    result = (t<T? (t/T) : 1.0); */

    // parabolic velocity profile at inflow - stationary
    /*const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
    if(component == 0)
    result = 1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0;
    if(component == dim)
    result = (p[0]-1.0)*pressure_gradient;*/

    // parabolic velocity profile at inflow - instationary
//    const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
//    double T = 0.5;
//    if(component == 0)
//      result = 1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0*(t<T? (t/T) : 1.0);
//    if(component == dim)
//    result = (p[0]-1.0)*pressure_gradient*(t<T? (t/T) : 1.0);
    /********************************************************************/

    /************************* vortex problem ***************************/
    //Taylor vortex problem (Shahbazi et al.,2007)
//    const double pi = numbers::PI;
//    if(component == 0)
//      result = (-std::cos(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 1)
//      result = (+std::sin(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 2)
//      result = -0.25*(std::cos(2*pi*p[0])+std::cos(2*pi*p[1]))*std::exp(-4.0*pi*pi*t*VISCOSITY);

    // vortex problem (Hesthaven)
    const double pi = numbers::PI;
    if (component == 0)
      result = -std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if (component == 1)
      result = std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if (component == dim)
      result = -std::cos(2*pi*p[0])*std::cos(2*pi*p[1])*std::exp(-8.0*pi*pi*VISCOSITY*t);
    /********************************************************************/

    /************************* Kovasznay flow ***************************/
//    const double pi = numbers::PI;
//    if (component == 0)
//      result = 1.0 - std::exp(lambda*p[0])*std::cos(2*pi*p[1]);
//    else if (component == 1)
//      result = lambda/2.0/pi*std::exp(lambda*p[0])*std::sin(2*pi*p[1]);
//    else if (component == dim)
//      result = 0.5*(1.0-std::exp(2.0*lambda*p[0]));
    /********************************************************************/

    /************************* Beltrami flow ****************************/
//    const double pi = numbers::PI;
//    const double a = 0.25*pi;
//    const double d = 2*a;
//    if (component == 0)
//      result = -a*(std::exp(a*p[0])*std::sin(a*p[1]+d*p[2]) + std::exp(a*p[2])*std::cos(a*p[0]+d*p[1]))*std::exp(-VISCOSITY*d*d*t);
//    else if (component == 1)
//      result = -a*(std::exp(a*p[1])*std::sin(a*p[2]+d*p[0]) + std::exp(a*p[0])*std::cos(a*p[1]+d*p[2]))*std::exp(-VISCOSITY*d*d*t);
//    else if (component == 2)
//      result = -a*(std::exp(a*p[2])*std::sin(a*p[0]+d*p[1]) + std::exp(a*p[1])*std::cos(a*p[2]+d*p[0]))*std::exp(-VISCOSITY*d*d*t);
//    else if (component == dim)
//        result = -a*a*0.5*(std::exp(2*a*p[0]) + std::exp(2*a*p[1]) + std::exp(2*a*p[2]) +
//                           2*std::sin(a*p[0]+d*p[1])*std::cos(a*p[2]+d*p[0])*std::exp(a*(p[1]+p[2])) +
//                           2*std::sin(a*p[1]+d*p[2])*std::cos(a*p[0]+d*p[1])*std::exp(a*(p[2]+p[0])) +
//                           2*std::sin(a*p[2]+d*p[0])*std::cos(a*p[1]+d*p[2])*std::exp(a*(p[0]+p[1]))) * std::exp(-2*VISCOSITY*d*d*t);
    /********************************************************************/

    /************* Stokes problem (Guermond,2003 & 2006) ****************/
//    const double pi = numbers::PI;
//    double sint = std::sin(t);
//    double sinx = std::sin(pi*p[0]);
//    double siny = std::sin(pi*p[1]);
//    double cosx = std::cos(pi*p[0]);
//    double sin2x = std::sin(2.*pi*p[0]);
//    double sin2y = std::sin(2.*pi*p[1]);
//    if (component == 0)
//      result = pi*sint*sin2y*pow(sinx,2.);
//    else if (component == 1)
//      result = -pi*sint*sin2x*pow(siny,2.);
//    else if (component == dim)
//      result = cosx*siny*sint;
    /********************************************************************/

    /********************** flow past cylinder **************************/
//    if(component == 0 && std::abs(p[0]-(dim==2 ? 0.3 : 0.0))<1.e-12)
//    {
//    const double pi = numbers::PI;
//    const double T = 0.2;
//      const double H = 0.41;
//      double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
//      result = coefficient * p[1] * (H-p[1]) *  ( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0); //( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0); //( (t/T)<1.0 ? t/T : 1.0); //std::sin(pi*t/END_TIME);
//      if (dim == 3)
//        result *= p[2] * (H-p[2]);
//    }
    /********************************************************************/

    return result;
  }

  template<int dim>
  class NeumannBoundaryVelocity : public Function<dim>
  {
  public:
    NeumannBoundaryVelocity (const unsigned int   component,
                             const double    time = 0.) : Function<dim>(1, time),component(component) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double NeumannBoundaryVelocity<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double t = this->get_time();
    double result = 0.0;

    // Kovasznay flow
//    const double pi = numbers::PI;
//    if (component == 0)
//      result = -lambda*std::exp(lambda)*std::cos(2*pi*p[1]);
//    else if (component == 1)
//      result = std::pow(lambda,2.0)/2/pi*std::exp(lambda)*std::sin(2*pi*p[1]);

    //Taylor vortex (Shahbazi et al.,2007)
//    const double pi = numbers::PI;
//    if(component == 0)
//      result = (pi*std::sin(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 1)
//      result = (+pi*std::cos(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);

    // vortex problem (Hesthaven)
    const double pi = numbers::PI;
    if (component==0)
      {
        if ( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
          result = 2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if ( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
          result = -2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
    else if (component==1)
      {
        if ( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
          result = -2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if ((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
          result = 2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
    return result;
  }

  template<int dim>
  class NeumannBoundaryPressure : public Function<dim>
  {
  public:
    NeumannBoundaryPressure (const unsigned int   n_components = 1,
                             const double       time = 0.) : Function<dim>(n_components, time) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double NeumannBoundaryPressure<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double result = 0.0;
    // Kovasznay flow
//    if(std::abs(p[0]+1.0)<1.0e-12)
//      result = lambda*std::exp(2.0*lambda*p[0]);

    // Poiseuille
//    const double pressure_gradient = -0.01;
//    if(std::abs(p[0]+1.0)<1.0e-12)
//      result = -pressure_gradient;

    return result;
  }

  template <int dim, typename Number>
  Tensor<1,dim,Number> get_rhs(Point<dim,Number> &point, double time)
  {
    Tensor<1,dim,Number> rhs;
    for (unsigned int d=0; d<dim; ++d)
      rhs[d] = 0.0;

    return rhs;
  }

  template<int dim>
  class RHS : public Function<dim>
  {
  public:
    RHS (const unsigned int   component,
         const double    time = 0.) : Function<dim>(1, time),component(component) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double RHS<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double t = this->get_time();
    double result = 0.0;

    // Stokes problem (Guermond,2003 & 2006)
//  const double pi = numbers::PI;
//  double sint = std::sin(t);
//  double cost = std::cos(t);
//  double sinx = std::sin(pi*p[0]);
//  double siny = std::sin(pi*p[1]);
//  double cosx = std::cos(pi*p[0]);
//  double cosy = std::cos(pi*p[1]);
//  double sin2x = std::sin(2.*pi*p[0]);
//  double sin2y = std::sin(2.*pi*p[1]);
//  if (component == 0)
//    result = pi*cost*sin2y*pow(sinx,2.)
//        - 2.*pow(pi,3.)*sint*sin2y*(1.-4.*pow(sinx,2.))
//        - pi*sint*sinx*siny;
//  else if (component == 1)
//    result = -pi*cost*sin2x*pow(siny,2.)
//        + 2.*pow(pi,3.)*sint*sin2x*(1.-4.*pow(siny,2.))
//        + pi*sint*cosx*cosy;

    return result;
  }

  template<int dim>
  class PressureBC_dudt : public Function<dim>
  {
  public:
    PressureBC_dudt (const unsigned int   component,
                     const double    time = 0.) : Function<dim>(1, time),component(component) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double PressureBC_dudt<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double t = this->get_time();
    double result = 0.0;

    //Taylor vortex (Shahbazi et al.,2007)
//  const double pi = numbers::PI;
//  if(component == 0)
//    result = (2.0*pi*pi*VISCOSITY*std::cos(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//  else if(component == 1)
//    result = (-2.0*pi*pi*VISCOSITY*std::sin(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);

    // vortex problem (Hesthaven)
    const double pi = numbers::PI;
    if (component == 0)
      result = 4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if (component == 1)
      result = -4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);

    // Beltrami flow
//  const double pi = numbers::PI;
//  const double a = 0.25*pi;
//  const double d = 2*a;
//  if (component == 0)
//    result = a*VISCOSITY*d*d*(std::exp(a*p[0])*std::sin(a*p[1]+d*p[2]) + std::exp(a*p[2])*std::cos(a*p[0]+d*p[1]))*std::exp(-VISCOSITY*d*d*t);
//  else if (component == 1)
//    result = a*VISCOSITY*d*d*(std::exp(a*p[1])*std::sin(a*p[2]+d*p[0]) + std::exp(a*p[0])*std::cos(a*p[1]+d*p[2]))*std::exp(-VISCOSITY*d*d*t);
//  else if (component == 2)
//    result = a*VISCOSITY*d*d*(std::exp(a*p[2])*std::sin(a*p[0]+d*p[1]) + std::exp(a*p[1])*std::cos(a*p[2]+d*p[0]))*std::exp(-VISCOSITY*d*d*t);

    // Stokes problem (Guermond,2003 & 2006)
//  const double pi = numbers::PI;
//  double cost = std::cos(t);
//  double sinx = std::sin(pi*p[0]);
//  double siny = std::sin(pi*p[1]);
//  double cosx = std::cos(pi*p[0]);
//  double cosy = std::cos(pi*p[1]);
//  double sin2x = std::sin(2.*pi*p[0]);
//  double sin2y = std::sin(2.*pi*p[1]);
//  if (component == 0)
//    result = pi*cost*sin2y*pow(sinx,2.);
//  else if (component == 1)
//    result = -pi*cost*sin2x*pow(siny,2.);

    // flow past cylinder
//  if(component == 0 && std::abs(p[0]-0.3)<1.e-12)
//  {
//    const double pi = numbers::PI;
//    const double H = 0.41;
//    double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
//    result = coefficient * p[1] * (H-p[1]) * std::cos(pi*t/END_TIME)*pi/END_TIME;
//  }

    return result;
  }

  template<int dim, int fe_degree, int fe_degree_p> struct NavierStokesPressureMatrix;
  template<int dim, int fe_degree, int fe_degree_p> struct NavierStokesViscousMatrix;


  template <typename MATRIX>
  class MGTransferMF : public MGTransferPrebuilt<parallel::distributed::Vector<double> >
  {
  public:
    MGTransferMF(const MGLevelObject<MATRIX> &matrix)
      :
      matrix_operator (matrix)
    {};

    /**
     * Overload copy_to_mg from MGTransferPrebuilt
     */
    template <int dim, class InVector, int spacedim>
    void
    copy_to_mg (const DoFHandler<dim,spacedim> &mg_dof,
                MGLevelObject<parallel::distributed::Vector<double> > &dst,
                const InVector &src) const
    {
      for (unsigned int level=dst.min_level();
           level<=dst.max_level(); ++level)
        matrix_operator[level].initialize_dof_vector(dst[level]);
      MGTransferPrebuilt<parallel::distributed::Vector<double> >::copy_to_mg(mg_dof, dst, src);
    }

  private:
    const MGLevelObject<MATRIX> &matrix_operator;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  class MGCoarsePressure : public MGCoarseGridBase<parallel::distributed::Vector<double> >
  {
  public:
    MGCoarsePressure() {}

    void initialize(const NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p> &pressure)
    {
      ns_pressure_coarse = &pressure;
    }

    virtual void operator() (const unsigned int   level,
                             parallel::distributed::Vector<double> &dst,
                             const parallel::distributed::Vector<double> &src) const
    {
      SolverControl solver_control (1e4, 1e-12); //1e-10
      SolverCG<parallel::distributed::Vector<double> > solver_coarse (solver_control);
      solver_coarse.solve (*ns_pressure_coarse, dst, src, PreconditionIdentity());
    }

    const  NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p> *ns_pressure_coarse;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  class MGCoarseViscous : public MGCoarseGridBase<parallel::distributed::Vector<double> >
  {
  public:
    MGCoarseViscous() {}

    void initialize(const NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p> &viscous)
    {
      ns_viscous_coarse = &viscous;
    }

    virtual void operator() (const unsigned int   level,
                             parallel::distributed::Vector<double> &dst,
                             const parallel::distributed::Vector<double> &src) const
    {
      SolverControl solver_control (1e4, 1e-12); //1e-10
      SolverCG<parallel::distributed::Vector<double> > solver_coarse (solver_control);
      solver_coarse.solve (*ns_viscous_coarse, dst, src, PreconditionIdentity());
    }

    const  NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p> *ns_viscous_coarse;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  class NavierStokesOperation
  {
  public:
    typedef double value_type;
    static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;

    NavierStokesOperation(const Mapping<dim> &mapping,
                          const DoFHandler<dim> &dof_handler,
                          const DoFHandler<dim> &dof_handler_p,
                          const double time_step_size,
                          std::set<types::boundary_id> dirichlet_bc_indicator,
                          std::set<types::boundary_id> neumann_bc_indicator);

    void do_timestep (const double  &cur_time,const double  &delta_t, const unsigned int &time_step_number);

    void  rhs_convection (const std::vector<parallel::distributed::Vector<value_type> > &src,
                          std::vector<parallel::distributed::Vector<value_type> >    &dst);

    void  compute_rhs (std::vector<parallel::distributed::Vector<value_type> >  &dst);

    void  apply_viscous (const parallel::distributed::Vector<value_type>    &src,
                         parallel::distributed::Vector<value_type>      &dst) const;

    void  apply_viscous (const parallel::distributed::Vector<value_type> &src,
                         parallel::distributed::Vector<value_type>     &dst,
                         const unsigned int                &level) const;

    void  rhs_viscous (const std::vector<parallel::distributed::Vector<value_type> >  &src,
                       std::vector<parallel::distributed::Vector<value_type> >  &dst);

    void  apply_pressure (const parallel::distributed::Vector<value_type>     &src,
                          parallel::distributed::Vector<value_type>      &dst) const;

    void  apply_pressure (const parallel::distributed::Vector<value_type>   &src,
                          parallel::distributed::Vector<value_type>    &dst,
                          const unsigned int               &level) const;

    void  apply_P (parallel::distributed::Vector<value_type> &dst) const;

    void  shift_pressure (parallel::distributed::Vector<value_type> &pressure);

    void apply_inverse_mass_matrix(const parallel::distributed::Vector<value_type>  &src,
                                   parallel::distributed::Vector<value_type>   &dst) const;

    void  rhs_pressure (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                        std::vector<parallel::distributed::Vector<value_type> >      &dst);

    void  apply_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                            std::vector<parallel::distributed::Vector<value_type> >      &dst);

    void compute_vorticity (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                            std::vector<parallel::distributed::Vector<value_type> >      &dst);

    void analyse_computing_times();

    std::vector<parallel::distributed::Vector<value_type> > solution_nm, solution_n, velocity_temp, velocity_temp2, solution_np;
    std::vector<parallel::distributed::Vector<value_type> > vorticity_nm, vorticity_n;
    std::vector<parallel::distributed::Vector<value_type> > rhs_convection_nm, rhs_convection_n;
    std::vector<parallel::distributed::Vector<value_type> > f;

    const MatrixFree<dim,value_type> &get_data() const
    {
      return data.back();
    }

    const MatrixFree<dim,value_type> &get_data(const unsigned int level) const
    {
      return data[level];
    }

    void calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal) const;

    void calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal, unsigned int level) const;

    void calculate_diagonal_viscous(parallel::distributed::Vector<value_type> &diagonal, unsigned int level) const;

  private:
    //MatrixFree<dim,value_type> data;
    std::vector<MatrixFree<dim,value_type> > data;

    MappingQ<dim> mapping;

    ConditionalOStream pcout;

    double time, time_step;
    const double viscosity;
    double gamma0;
    double alpha[2], beta[2];
    std::vector<double> computing_times;
    std::vector<double> times_cg_velo;
    std::vector<unsigned int> iterations_cg_velo;
    std::vector<double> times_cg_pressure;
    std::vector<unsigned int> iterations_cg_pressure;

    //NavierStokesPressureMatrix<dim,fe_degree, fe_degree_p> ns_pressure_matrix;
    MGLevelObject<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p> > mg_matrices_pressure;
    //MGTransferPrebuilt<parallel::distributed::Vector<double> > mg_transfer_pressure;
    MGTransferMF<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p> > mg_transfer_pressure;

    typedef PreconditionChebyshev<NavierStokesPressureMatrix<dim,fe_degree, fe_degree_p>,
            parallel::distributed::Vector<double> > SMOOTHER_PRESSURE;
    // typename SMOOTHER_PRESSURE::AdditionalData smoother_data_pressure;
    MGLevelObject<typename SMOOTHER_PRESSURE::AdditionalData> smoother_data_pressure;

    MGSmootherPrecondition<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p>,
                           SMOOTHER_PRESSURE, parallel::distributed::Vector<double> > mg_smoother_pressure;
    MGCoarsePressure<dim,fe_degree,fe_degree_p> mg_coarse_pressure;

    MGLevelObject<NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p> > mg_matrices_viscous;
    //MGTransferPrebuilt<parallel::distributed::Vector<double> > mg_transfer_viscous;
    MGTransferMF<NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p> > mg_transfer_viscous;

    typedef PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p>,
            parallel::distributed::Vector<double> > SMOOTHER_VISCOUS;
    // typename SMOOTHER_VISCOUS::AdditionalData smoother_data_viscous;
    MGLevelObject<typename SMOOTHER_VISCOUS::AdditionalData> smoother_data_viscous;
    MGSmootherPrecondition<NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p>,
                           SMOOTHER_VISCOUS, parallel::distributed::Vector<double> > mg_smoother_viscous;
    MGCoarseViscous<dim,fe_degree,fe_degree_p> mg_coarse_viscous;

    Point<dim> first_point;
    types::global_dof_index dof_index_first_point;

    std::vector< AlignedVector<VectorizedArray<value_type> > > array_penalty_parameter;

    std::set<types::boundary_id> dirichlet_boundary;
    std::set<types::boundary_id> neumann_boundary;

    bool clear_files;

    void update_time_integrator();
    void check_time_integrator();

    // impulse equation
    void local_rhs_convection (const MatrixFree<dim,value_type>               &data,
                               std::vector<parallel::distributed::Vector<double> >     &dst,
                               const std::vector<parallel::distributed::Vector<double> >   &src,
                               const std::pair<unsigned int,unsigned int>          &cell_range) const;

    void local_rhs_convection_face (const MatrixFree<dim,value_type>              &data,
                                    std::vector<parallel::distributed::Vector<double> >     &dst,
                                    const std::vector<parallel::distributed::Vector<double> > &src,
                                    const std::pair<unsigned int,unsigned int>          &face_range) const;

    void local_rhs_convection_boundary_face(const MatrixFree<dim,value_type>              &data,
                                            std::vector<parallel::distributed::Vector<double> >     &dst,
                                            const std::vector<parallel::distributed::Vector<double> > &src,
                                            const std::pair<unsigned int,unsigned int>          &face_range) const;

    void local_compute_rhs (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >     &dst,
                            const std::vector<parallel::distributed::Vector<double> > &,
                            const std::pair<unsigned int,unsigned int>          &cell_range) const;

    void local_apply_viscous (const MatrixFree<dim,value_type>        &data,
                              parallel::distributed::Vector<double>     &dst,
                              const parallel::distributed::Vector<double> &src,
                              const std::pair<unsigned int,unsigned int>  &cell_range) const;

    void local_apply_viscous_face (const MatrixFree<dim,value_type>     &data,
                                   parallel::distributed::Vector<double>   &dst,
                                   const parallel::distributed::Vector<double> &src,
                                   const std::pair<unsigned int,unsigned int>  &face_range) const;

    void local_apply_viscous_boundary_face(const MatrixFree<dim,value_type>     &data,
                                           parallel::distributed::Vector<double>   &dst,
                                           const parallel::distributed::Vector<double> &src,
                                           const std::pair<unsigned int,unsigned int>  &face_range) const;

    void local_rhs_viscous (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >     &dst,
                            const std::vector<parallel::distributed::Vector<double> >   &src,
                            const std::pair<unsigned int,unsigned int>          &cell_range) const;

    void local_rhs_viscous_face (const MatrixFree<dim,value_type>             &data,
                                 std::vector<parallel::distributed::Vector<double> >     &dst,
                                 const std::vector<parallel::distributed::Vector<double> > &src,
                                 const std::pair<unsigned int,unsigned int>          &face_range) const;

    void local_rhs_viscous_boundary_face(const MatrixFree<dim,value_type>             &data,
                                         std::vector<parallel::distributed::Vector<double> >     &dst,
                                         const std::vector<parallel::distributed::Vector<double> > &src,
                                         const std::pair<unsigned int,unsigned int>          &face_range) const;

    // poisson equation
    void local_apply_pressure (const MatrixFree<dim,value_type>         &data,
                               parallel::distributed::Vector<double>     &dst,
                               const parallel::distributed::Vector<double> &src,
                               const std::pair<unsigned int,unsigned int>  &cell_range) const;

    void local_apply_pressure_face (const MatrixFree<dim,value_type>      &data,
                                    parallel::distributed::Vector<double>   &dst,
                                    const parallel::distributed::Vector<double> &src,
                                    const std::pair<unsigned int,unsigned int>  &face_range) const;

    void local_apply_pressure_boundary_face(const MatrixFree<dim,value_type>      &data,
                                            parallel::distributed::Vector<double>   &dst,
                                            const parallel::distributed::Vector<double> &src,
                                            const std::pair<unsigned int,unsigned int>  &face_range) const;

    void local_laplace_diagonal(const MatrixFree<dim,value_type>        &data,
                                parallel::distributed::Vector<double>     &dst,
                                const parallel::distributed::Vector<double> &src,
                                const std::pair<unsigned int,unsigned int>  &cell_range) const;

    void local_laplace_diagonal_face (const MatrixFree<dim,value_type>      &data,
                                      parallel::distributed::Vector<double>   &dst,
                                      const parallel::distributed::Vector<double> &src,
                                      const std::pair<unsigned int,unsigned int>  &face_range) const;

    void local_laplace_diagonal_boundary_face(const MatrixFree<dim,value_type>      &data,
                                              parallel::distributed::Vector<double>   &dst,
                                              const parallel::distributed::Vector<double> &src,
                                              const std::pair<unsigned int,unsigned int>  &face_range) const;

    void local_diagonal_viscous(const MatrixFree<dim,value_type>        &data,
                                parallel::distributed::Vector<double>     &dst,
                                const parallel::distributed::Vector<double> &src,
                                const std::pair<unsigned int,unsigned int>  &cell_range) const;

    void local_diagonal_viscous_face (const MatrixFree<dim,value_type>      &data,
                                      parallel::distributed::Vector<double>   &dst,
                                      const parallel::distributed::Vector<double> &src,
                                      const std::pair<unsigned int,unsigned int>  &face_range) const;

    void local_diagonal_viscous_boundary_face(const MatrixFree<dim,value_type>      &data,
                                              parallel::distributed::Vector<double>   &dst,
                                              const parallel::distributed::Vector<double> &src,
                                              const std::pair<unsigned int,unsigned int>  &face_range) const;

    void local_rhs_pressure (const MatrixFree<dim,value_type>               &data,
                             std::vector<parallel::distributed::Vector<double> >     &dst,
                             const std::vector<parallel::distributed::Vector<double> >   &src,
                             const std::pair<unsigned int,unsigned int>          &cell_range) const;

    void local_rhs_pressure_face (const MatrixFree<dim,value_type>              &data,
                                  std::vector<parallel::distributed::Vector<double> >     &dst,
                                  const std::vector<parallel::distributed::Vector<double> > &src,
                                  const std::pair<unsigned int,unsigned int>          &face_range) const;

    void local_rhs_pressure_boundary_face(const MatrixFree<dim,value_type>              &data,
                                          std::vector<parallel::distributed::Vector<double> >     &dst,
                                          const std::vector<parallel::distributed::Vector<double> > &src,
                                          const std::pair<unsigned int,unsigned int>          &face_range) const;

    // inverse mass matrix velocity
    void local_apply_mass_matrix(const MatrixFree<dim,value_type>               &data,
                                 std::vector<parallel::distributed::Vector<value_type> >   &dst,
                                 const std::vector<parallel::distributed::Vector<value_type> > &src,
                                 const std::pair<unsigned int,unsigned int>          &cell_range) const;

    void local_apply_mass_matrix(const MatrixFree<dim,value_type>         &data,
                                 parallel::distributed::Vector<value_type>     &dst,
                                 const parallel::distributed::Vector<value_type>   &src,
                                 const std::pair<unsigned int,unsigned int>    &cell_range) const;

    // projection step
    void local_projection (const MatrixFree<dim,value_type>               &data,
                           std::vector<parallel::distributed::Vector<double> >     &dst,
                           const std::vector<parallel::distributed::Vector<double> >   &src,
                           const std::pair<unsigned int,unsigned int>          &cell_range) const;

    void local_compute_vorticity (const MatrixFree<dim,value_type>                &data,
                                  std::vector<parallel::distributed::Vector<double> >     &dst,
                                  const std::vector<parallel::distributed::Vector<double> >   &src,
                                  const std::pair<unsigned int,unsigned int>          &cell_range) const;

    //penalty parameter
    void calculate_penalty_parameter(double &factor) const;

    void calculate_penalty_parameter_pressure(double &factor) const;

    void compute_lift_and_drag();
    void compute_pressure_difference();

    void my_point_value(const Mapping<dim> &mapping,
                        const DoFHandler<dim> &dof_handler,
                        const parallel::distributed::Vector<double> &solution,
                        const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > &cell_point,
                        Vector<double> &value);
  };

  template<int dim, int fe_degree, int fe_degree_p>
  NavierStokesOperation<dim,fe_degree, fe_degree_p>::NavierStokesOperation(const Mapping<dim> &mapping,
      const DoFHandler<dim> &dof_handler,
      const DoFHandler<dim> &dof_handler_p,
      const double time_step_size,
      std::set<types::boundary_id> dirichlet_bc_indicator,
      std::set<types::boundary_id> neumann_bc_indicator):
    mapping(fe_degree),
    pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
    time(0.0),
    time_step(time_step_size),
    viscosity(VISCOSITY),
    gamma0(1.0),
    alpha({1.0,0.0}),
        beta({1.0,0.0}),
        computing_times(4),
        times_cg_velo(3),
        iterations_cg_velo(3),
        times_cg_pressure(2),
        iterations_cg_pressure(2),
        mg_transfer_pressure(mg_matrices_pressure),
        mg_transfer_viscous(mg_matrices_viscous),
        dirichlet_boundary(dirichlet_bc_indicator),
        neumann_boundary(neumann_bc_indicator),
        clear_files(true)
  {
    data.resize(dof_handler_p.get_tria().n_levels());
    //mg_matrices_pressure.resize(dof_handler_p.get_tria().n_levels()-2, dof_handler_p.get_tria().n_levels()-1);
    mg_matrices_pressure.resize(0, dof_handler_p.get_tria().n_levels()-1);
    mg_matrices_viscous.resize(0, dof_handler.get_tria().n_levels()-1);
    gamma0 = 3.0/2.0;
    array_penalty_parameter.resize(dof_handler_p.get_tria().n_levels());

    for (unsigned int level=mg_matrices_pressure.min_level(); level<=mg_matrices_pressure.max_level(); ++level)
      {
        // initialize matrix_free_data
        typename MatrixFree<dim,value_type>::AdditionalData additional_data;
        additional_data.mpi_communicator = MPI_COMM_WORLD;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim,value_type>::AdditionalData::partition_partition;
        additional_data.build_face_info = true;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                update_quadrature_points | update_normal_vectors |
                                                update_values);
        additional_data.level_mg_handler = level;

        std::vector<const DoFHandler<dim> * >  dof_handler_vec;
        dof_handler_vec.push_back(&dof_handler);
        dof_handler_vec.push_back(&dof_handler_p);

        ConstraintMatrix constraint, constraint_p;
        constraint.close();
        constraint_p.close();
        std::vector<const ConstraintMatrix *> constraint_matrix_vec;
        constraint_matrix_vec.push_back(&constraint);
        constraint_matrix_vec.push_back(&constraint_p);

        std::vector<Quadrature<1> > quadratures;
        quadratures.push_back(QGauss<1>(fe_degree+1));
        quadratures.push_back(QGauss<1>(fe_degree_p+1));
        // quadrature formula 2: exact integration of convective term
        quadratures.push_back(QGauss<1>(fe_degree + (fe_degree+2)/2));

        data[level].reinit (mapping, dof_handler_vec, constraint_matrix_vec,
                            quadratures, additional_data);

        // penalty parameter: calculate surface/volume ratio for each cell
        QGauss<dim> quadrature(fe_degree+1);
        FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature, update_JxW_values);
        QGauss<dim-1> face_quadrature(fe_degree+1);
        FEFaceValues<dim> fe_face_values(mapping, dof_handler.get_fe(), face_quadrature,update_JxW_values);
        //pcout << "Level " << level << std::endl;
        array_penalty_parameter[level].resize(data[level].n_macro_cells()+data[level].n_macro_ghost_cells());
        for (unsigned int i=0; i<data[level].n_macro_cells()+data[level].n_macro_ghost_cells(); ++i)
          for (unsigned int v=0; v<data[level].n_components_filled(i); ++v)
            {
              typename DoFHandler<dim>::cell_iterator cell = data[level].get_cell_iterator(i,v);
              fe_values.reinit(cell);
              double volume = 0;
              for (unsigned int q=0; q<quadrature.size(); ++q)
                volume += fe_values.JxW(q);
              double surface_area = 0;
              for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                {
                  fe_face_values.reinit(cell, f);
                  const double factor = cell->at_boundary(f) ? 1. : 0.5;
                  for (unsigned int q=0; q<face_quadrature.size(); ++q)
                    surface_area += fe_face_values.JxW(q) * factor;
                }
              array_penalty_parameter[level][i][v] = surface_area / volume;
              //pcout << "surface to volume ratio: " << array_penalty_parameter[level][i][v] << std::endl;
            }
        mg_matrices_pressure[level].initialize(*this, level);
        mg_matrices_viscous[level].initialize(*this, level);
      }
//  Vector<double> tau_stab(triangulation.n_active_cells());
//  for (unsigned int c=0; c<data.back().n_macro_cells(); ++c)
//    for (unsigned int v=0; v<data.back().n_components_filled(c); ++v)
//      tau_stab(data.back().get_cell_iterator(c,v)->active_cell_index()) =
//      array_penalty_parameter.back()[c][v];
//  DataOut<dim> data_out;
//  data_out.attach_dof_handler (data.back().get_dof_handler(0));
//  data_out.add_data_vector(tau_stab,"StabilisationParameter");
//  data_out.build_patches ();
//  std::ostringstream filename;
//  filename << "StabilisationParameter"
//       << ".vtk";
//  std::ofstream output (filename.str().c_str());
//  data_out.write_vtk(output);

    mg_transfer_pressure.build_matrices(dof_handler_p);
    mg_coarse_pressure.initialize(mg_matrices_pressure[mg_matrices_pressure.min_level()]);

    mg_transfer_viscous.build_matrices(dof_handler);
    mg_coarse_viscous.initialize(mg_matrices_viscous[mg_matrices_viscous.min_level()]);

    smoother_data_pressure.resize(0, dof_handler_p.get_tria().n_levels()-1);
    for (unsigned int level=0; level<dof_handler_p.get_tria().n_levels(); ++level)
      {
        smoother_data_pressure[level].smoothing_range = 30;
        smoother_data_pressure[level].degree = 5; //empirically: use degree = 3 - 6
        smoother_data_pressure[level].eig_cg_n_iterations = 20; //20

        smoother_data_pressure[level].matrix_diagonal_inverse = mg_matrices_pressure[level].get_inverse_diagonal();
      }
    mg_smoother_pressure.initialize(mg_matrices_pressure, smoother_data_pressure);

    smoother_data_viscous.resize(0,dof_handler.get_tria().n_levels()-1);
    for (unsigned int level=0; level<dof_handler.get_tria().n_levels(); ++level)
      {
        smoother_data_viscous[level].smoothing_range = 30;
        smoother_data_viscous[level].degree = 5; //empirically: use degree = 3 - 6
        smoother_data_viscous[level].eig_cg_n_iterations = 30;

        smoother_data_viscous[level].matrix_diagonal_inverse = mg_matrices_viscous[level].get_inverse_diagonal();
      }
    mg_smoother_viscous.initialize(mg_matrices_viscous, smoother_data_viscous);
    gamma0 = 1.0;

    // initialize solution vectors
    solution_n.resize(dim+1);
    data.back().initialize_dof_vector(solution_n[0]);
    for (unsigned int d=1; d<dim; ++d)
      {
        solution_n[d] = solution_n[0];
      }
    for (unsigned int d=0; d<dim; ++d)
      {
        solution_n[d] = 0.0;
      }
    data.back().initialize_dof_vector(solution_n[dim], 1);
    solution_n[dim] = 0.0;
    solution_nm = solution_n;
    solution_np = solution_n;

    velocity_temp.resize(dim);
    data.back().initialize_dof_vector(velocity_temp[0]);
    for (unsigned int d=1; d<dim; ++d)
      {
        velocity_temp[d] = velocity_temp[0];
      }
    velocity_temp2 = velocity_temp;

    vorticity_n.resize(number_vorticity_components);
    data.back().initialize_dof_vector(vorticity_n[0]);
    for (unsigned int d=1; d<number_vorticity_components; ++d)
      {
        vorticity_n[d] = vorticity_n[0];
      }
    vorticity_nm = vorticity_n;

    rhs_convection_n.resize(dim);
    data.back().initialize_dof_vector(rhs_convection_n[0]);
    for (unsigned int d=1; d<dim; ++d)
      {
        rhs_convection_n[d] = rhs_convection_n[0];
      }
    rhs_convection_nm = rhs_convection_n;
    f = rhs_convection_n;

    dof_index_first_point = 0;
    for (unsigned int d=0; d<dim; ++d)
      first_point[d] = 0.0;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        typename DoFHandler<dim>::active_cell_iterator first_cell;
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_p.begin_active(), endc = dof_handler_p.end();
        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              first_cell = cell;
              break;
            }
        FEValues<dim> fe_values(dof_handler_p.get_fe(),
                                Quadrature<dim>(dof_handler_p.get_fe().get_unit_support_points()),
                                update_quadrature_points);
        fe_values.reinit(first_cell);
        first_point = fe_values.quadrature_point(0);
        std::vector<types::global_dof_index> dof_indices(dof_handler_p.get_fe().dofs_per_cell);
        first_cell->get_dof_indices(dof_indices);
        dof_index_first_point = dof_indices[0];
      }
    dof_index_first_point = Utilities::MPI::sum(dof_index_first_point,MPI_COMM_WORLD);
    for (unsigned int d=0; d<dim; ++d)
      first_point[d] = Utilities::MPI::sum(first_point[d],MPI_COMM_WORLD);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  struct NavierStokesPressureMatrix : public Subscriptor
  {
    void initialize(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op, unsigned int lvl)
    {
      ns_operation = &ns_op;
      level = lvl;
      ns_operation->get_data(level).initialize_dof_vector(diagonal,1);
      ns_operation->calculate_laplace_diagonal(diagonal,level);

      // initialize inverse diagonal
      inverse_diagonal = diagonal;
      for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
        if ( std::abs(inverse_diagonal.local_element(i)) > 1.0e-10 )
          inverse_diagonal.local_element(i) = 1.0/diagonal.local_element(i);
    }

    unsigned int m() const
    {
      return ns_operation->get_data(level).get_vector_partitioner(1)->size();
    }

    double el(const unsigned int row,const unsigned int /*col*/) const
    {
      return diagonal(row);
    }

    void vmult (parallel::distributed::Vector<double> &dst,
                const parallel::distributed::Vector<double> &src) const
    {
      dst = 0;
      vmult_add(dst,src);
    }

    void Tvmult (parallel::distributed::Vector<double> &dst,
                 const parallel::distributed::Vector<double> &src) const
    {
      dst = 0;
      vmult_add(dst,src);
    }

    void Tvmult_add (parallel::distributed::Vector<double> &dst,
                     const parallel::distributed::Vector<double> &src) const
    {
      vmult_add(dst,src);
    }

    void vmult_add (parallel::distributed::Vector<double> &dst,
                    const parallel::distributed::Vector<double> &src) const
    {
      if (pure_dirichlet_bc)
        {
          parallel::distributed::Vector<double> temp1(src);
          ns_operation->apply_P(temp1);
          ns_operation->apply_pressure(temp1,dst,level);
          ns_operation->apply_P(dst);
        }
      else
        {
          ns_operation->apply_pressure(src,dst,level);
        }
    }

    const parallel::distributed::Vector<double> &get_inverse_diagonal() const
    {
      return inverse_diagonal;
    }

    void initialize_dof_vector(parallel::distributed::Vector<double> &src) const
    {
      ns_operation->get_data(level).initialize_dof_vector(src,1);
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p> *ns_operation;
    unsigned int level;
    parallel::distributed::Vector<double> diagonal;
    parallel::distributed::Vector<double> inverse_diagonal;
  };

  /*  template <int dim, int fe_degree, int fe_degree_p>
    struct NavierStokesViscousMatrix
    {
      NavierStokesViscousMatrix(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op)
      :
        ns_op(ns_op)
      {}

      void vmult (parallel::distributed::Vector<double> &dst,
          const parallel::distributed::Vector<double> &src) const
      {
        ns_op.apply_viscous(src,dst);
      }

      const NavierStokesOperation<dim,fe_degree, fe_degree_p> ns_op;
    }; */

  template <int dim, int fe_degree, int fe_degree_p>
  struct NavierStokesViscousMatrix : public Subscriptor
  {
    void initialize(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op, unsigned int lvl)
    {
      ns_operation = &ns_op;
      level = lvl;
      ns_operation->get_data(level).initialize_dof_vector(diagonal,0);
      ns_operation->calculate_diagonal_viscous(diagonal,level);

      // initialize inverse diagonal
      inverse_diagonal = diagonal;
      for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
        if ( std::abs(inverse_diagonal.local_element(i)) > 1.0e-10 )
          inverse_diagonal.local_element(i) = 1.0/diagonal.local_element(i);
    }

    unsigned int m() const
    {
      return ns_operation->get_data(level).get_vector_partitioner(0)->size();
    }

    double el(const unsigned int row,const unsigned int /*col*/) const
    {
      return diagonal(row);
    }

    void vmult (parallel::distributed::Vector<double> &dst,
                const parallel::distributed::Vector<double> &src) const
    {
      dst = 0;
      vmult_add(dst,src);
    }

    void Tvmult (parallel::distributed::Vector<double> &dst,
                 const parallel::distributed::Vector<double> &src) const
    {
      dst = 0;
      vmult_add(dst,src);
    }

    void Tvmult_add (parallel::distributed::Vector<double> &dst,
                     const parallel::distributed::Vector<double> &src) const
    {
      vmult_add(dst,src);
    }

    void vmult_add (parallel::distributed::Vector<double> &dst,
                    const parallel::distributed::Vector<double> &src) const
    {
      ns_operation->apply_viscous(src,dst,level);
    }

    const parallel::distributed::Vector<double> &get_inverse_diagonal() const
    {
      return inverse_diagonal;
    }

    void initialize_dof_vector(parallel::distributed::Vector<double> &src) const
    {
      ns_operation->get_data(level).initialize_dof_vector(src,0);
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p> *ns_operation;
    unsigned int level;
    parallel::distributed::Vector<double> diagonal;
    parallel::distributed::Vector<double> inverse_diagonal;
  };

  template <int dim, int fe_degree, int fe_degree_p>
  struct PreconditionerInverseMassMatrix
  {
    PreconditionerInverseMassMatrix(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op)
      :
      ns_op(ns_op)
    {}

    void vmult (parallel::distributed::Vector<double> &dst,
                const parallel::distributed::Vector<double> &src) const
    {
      ns_op.apply_inverse_mass_matrix(src,dst);
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p> &ns_op;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  struct PreconditionerJacobi
  {
    PreconditionerJacobi(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op)
      :
      ns_operation(ns_op)
    {
      ns_operation.get_data().initialize_dof_vector(diagonal,1);
      ns_operation.calculate_laplace_diagonal(diagonal);
    }

    void vmult (parallel::distributed::Vector<double> &dst,
                const parallel::distributed::Vector<double> &src) const
    {
      for (unsigned int i=0; i<src.local_size(); ++i)
        {
          dst.local_element(i) = src.local_element(i)/diagonal.local_element(i);
        }
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p> &ns_operation;
    parallel::distributed::Vector<double> diagonal;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  do_timestep (const double &cur_time,const double  &delta_t, const unsigned int &time_step_number)
  {
    if (time_step_number == 1)
      check_time_integrator();

    const unsigned int output_solver_info_every_timesteps = 1e2;
    const unsigned int output_solver_info_details = 1e4;

    time = cur_time;
    time_step = delta_t;

    Timer timer;
    timer.restart();

    /***************** STEP 1: convective (nonlinear) term ********************/
    rhs_convection(solution_n,rhs_convection_n);
    compute_rhs(f);
    for (unsigned int d=0; d<dim; ++d)
      {
        velocity_temp[d].equ(beta[0],rhs_convection_n[d],beta[1],rhs_convection_nm[d],1.,f[d]); // Stokes problem: velocity_temp[d] = f[d];
        velocity_temp[d].sadd(time_step,alpha[0],solution_n[d],alpha[1],solution_nm[d]);
      }
    rhs_convection_nm = rhs_convection_n;

    computing_times[0] += timer.wall_time();
    /*************************************************************************/

    /************ STEP 2: solve poisson equation for pressure ****************/
    timer.restart();

    rhs_pressure(velocity_temp,solution_np);
    solution_np[dim] *= -1.0/time_step;

    // set maximum number of iterations, tolerance
    ReductionControl solver_control (1e5, 1.e-12, 1e-5);
    //SolverControl solver_control(1e5, 1.e-10);
    SolverCG<parallel::distributed::Vector<double> > solver (solver_control);

//  Timer cg_timer;
//  cg_timer.restart();

    // start CG-iterations with pressure solution at time t_n
    parallel::distributed::Vector<value_type> solution(solution_n[dim]);

    // CG-Solver without preconditioning
    solver.solve (mg_matrices_pressure[mg_matrices_pressure.max_level()], solution, solution_np[dim], PreconditionIdentity());

//    times_cg_pressure[0] += cg_timer.wall_time();
//    iterations_cg_pressure[0] += solver_control.last_step();
//    cg_timer.restart();
//    solution = solution_n[dim];

    // PCG-Solver with GMG + Chebyshev smoother as a preconditioner
//  mg::Matrix<parallel::distributed::Vector<double> > mgmatrix_pressure(mg_matrices_pressure);
//  Multigrid<parallel::distributed::Vector<double> > mg_pressure(data.back().get_dof_handler(1),
//                             mgmatrix_pressure,
//                               mg_coarse_pressure,
//                               mg_transfer_pressure,
//                               mg_smoother_pressure,
//                               mg_smoother_pressure);
//  PreconditionMG<dim, parallel::distributed::Vector<double>, MGTransferMF<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p> > >
//  preconditioner_pressure(data.back().get_dof_handler(1), mg_pressure, mg_transfer_pressure);
//  try
//  {
//    solver.solve (mg_matrices_pressure[mg_matrices_pressure.max_level()], solution, solution_np[dim], preconditioner_pressure);
//  }
//  catch (SolverControl::NoConvergence)
//  {
//    pcout<<"Multigrid failed. Try CG ..." << std::endl;
//    solution=solution_n[dim];
//    SolverControl solver_control (1e5, 1.e-8);
//    SolverCG<parallel::distributed::Vector<double> > solver (solver_control);
//    solver.solve (mg_matrices_pressure[mg_matrices_pressure.max_level()], solution, solution_np[dim], PreconditionIdentity());
//  }

//    times_cg_pressure[1] += cg_timer.wall_time();
//    iterations_cg_pressure[1] += solver_control.last_step();

//    if(time_step_number%output_solver_info_details == 0)
//    pcout << std::endl << "Solve pressure Poisson equation: Number of timesteps: " << time_step_number << std::endl
//          << "CG (no preconditioning):  wall time: " << times_cg_pressure[0]/time_step_number << " Iterations: " << (double)iterations_cg_pressure[0]/time_step_number << std::endl
//          << "PCG (GMG with Chebyshev): wall time: " << times_cg_pressure[1]/time_step_number << " Iterations: " << (double)iterations_cg_pressure[1]/time_step_number << std::endl
//          << std::endl;

//  if (false)
//  {
//    parallel::distributed::Vector<double> check1(mg_matrices_pressure[mg_matrices_pressure.max_level()].m()),
//        check2(check1), tmp(check1),
//        check3(check1);
//    for (unsigned int i=0; i<check1.size(); ++i)
//      check1(i) = (double)rand()/RAND_MAX;
//    mg_matrices_pressure[mg_matrices_pressure.max_level()].vmult(tmp, check1);
//    tmp *= -1.0;
//    preconditioner_pressure.vmult(check2, tmp);
//    check2 += check1;
//
//    mg_smoother_pressure.smooth(mg_matrices_pressure.max_level(), check3, tmp);
//    //check3 += check1;
//
//    DataOut<dim> data_out;
//    data_out.attach_dof_handler (data.back().get_dof_handler(1));
//    data_out.add_data_vector (check1, "initial");
//    data_out.add_data_vector (check2, "mg_cycle");
//    data_out.add_data_vector (check3, "smoother");
//    data_out.build_patches (data.back().get_dof_handler(1).get_fe().degree*3);
//    std::ostringstream filename;
//    filename << "smoothing-"
//         << Utilities::int_to_string(data.back().get_dof_handler(1).get_tria().n_active_cells(), 6)
//         << ".vtk";
//    std::ofstream output (filename.str().c_str());
//    data_out.write_vtk(output);
//    std::abort();
//  }

    if (pure_dirichlet_bc)
      {
        shift_pressure(solution);
      }
    solution_np[dim] = solution;

    if (time_step_number%output_solver_info_every_timesteps == 0)
      {
        pcout << std::endl << "Number of timesteps: " << time_step_number << std::endl;
        pcout << "Solve Poisson equation for p: PCG iterations: " << std::setw(3) << solver_control.last_step() << "  Wall time: " << timer.wall_time() << std::endl;
      }

    computing_times[1] += timer.wall_time();
    /*************************************************************************/

    /********************** STEP 3: projection *******************************/
    timer.restart();

    apply_projection(solution_np,velocity_temp2);
    for (unsigned int d=0; d<dim; ++d)
      {
        velocity_temp2[d].sadd(time_step,1.0,velocity_temp[d]);
      }
    computing_times[2] += timer.wall_time();
    /*************************************************************************/

    /************************ STEP 4: viscous term ***************************/
    timer.restart();

    rhs_viscous(velocity_temp2,solution_np);

    // set maximum number of iterations, tolerance
    ReductionControl solver_control_velocity (1e5, 1.e-12, 1e-5);
    //SolverControl solver_control_velocity (1e4, 1.e-8);

    SolverCG<parallel::distributed::Vector<double> > solver_velocity (solver_control_velocity);
    //NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p> ns_viscous_matrix(*this);
    for (unsigned int d=0; d<dim; ++d)
      {
        double wall_time_temp = timer.wall_time();

//    Timer cg_timer_viscous;
//    cg_timer_viscous.restart();

        // start CG-iterations with solution_n
        parallel::distributed::Vector<value_type> solution(solution_n[d]);

        // CG-Solver without preconditioning
        //solver_velocity.solve (ns_viscous_matrix, solution, solution_np[d], PreconditionIdentity());
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], PreconditionIdentity());

//    times_cg_velo[0] += cg_timer_viscous.wall_time();
//    iterations_cg_velo[0] += solver_control_velocity.last_step();
//    cg_timer_viscous.restart();
//    solution = solution_n[d];

        // PCG-Solver with inverse mass matrix as a preconditioner
        // solver_velocity.solve (ns_viscous_matrix, solution, solution_np[d], preconditioner);
        PreconditionerInverseMassMatrix<dim,fe_degree, fe_degree_p> preconditioner(*this);
        solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], preconditioner);

//    times_cg_velo[1] += cg_timer_viscous.wall_time();
//    iterations_cg_velo[1] += solver_control_velocity.last_step();
//    cg_timer_viscous.restart();
//    solution = solution_n[d];

        // PCG-Solver with GMG + Chebyshev smoother as a preconditioner
//    mg::Matrix<parallel::distributed::Vector<double> > mgmatrix_viscous(mg_matrices_viscous);
//    Multigrid<parallel::distributed::Vector<double> > mg_viscous(data.back().get_dof_handler(0),
//                               mgmatrix_viscous,
//                               mg_coarse_viscous,
//                               mg_transfer_viscous,
//                               mg_smoother_viscous,
//                               mg_smoother_viscous);
//    PreconditionMG<dim, parallel::distributed::Vector<double>, MGTransferMF<NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p> > >
//    preconditioner_viscous(data.back().get_dof_handler(0), mg_viscous, mg_transfer_viscous);
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], preconditioner_viscous);

        // PCG-Solver with Chebyshev preconditioner
//    PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p>,parallel::distributed::Vector<value_type> > precondition_chebyshev;
//    typename PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p>,parallel::distributed::Vector<value_type> >::AdditionalData smoother_data;
//    smoother_data.smoothing_range = 30;
//    smoother_data.degree = 5;
//    smoother_data.eig_cg_n_iterations = 30;
//    precondition_chebyshev.initialize(mg_matrices_viscous[mg_matrices_viscous.max_level()], smoother_data);
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], precondition_chebyshev);

//    times_cg_velo[2] += cg_timer_viscous.wall_time();
//    iterations_cg_velo[2] += solver_control_velocity.last_step();

        solution_np[d] = solution;

        if (time_step_number%output_solver_info_every_timesteps == 0)
          {
            pcout << "Solve viscous step for u" << d+1 <<":    PCG iterations: " << std::setw(3) << solver_control_velocity.last_step() << "  Wall time: " << timer.wall_time()-wall_time_temp << std::endl;
          }
      }

//  if(time_step_number%output_solver_info_details == 0)
//    pcout << "Solve viscous step for u: Number of timesteps: " << time_step_number << std::endl
//        << "CG (no preconditioning):  wall time: " << times_cg_velo[0]/time_step_number << " Iterations: " << (double)iterations_cg_velo[0]/time_step_number/dim << std::endl
//        << "PCG (inv mass precond.):  wall time: " << times_cg_velo[1]/time_step_number << " Iterations: " << (double)iterations_cg_velo[1]/time_step_number/dim << std::endl
//        << "PCG (GMG with Chebyshev): wall time: " << times_cg_velo[2]/time_step_number << " Iterations: " << (double)iterations_cg_velo[2]/time_step_number/dim << std::endl
//        << std::endl;

    computing_times[3] += timer.wall_time();
    /*************************************************************************/

    // solution at t_n -> solution at t_n-1    and    solution at t_n+1 -> solution at t_n
    solution_nm.swap(solution_n);
    solution_n.swap(solution_np);

    vorticity_nm = vorticity_n;
    compute_vorticity(solution_n,vorticity_n);

//  compute_lift_and_drag();
//  compute_pressure_difference();

    if (time_step_number == 1)
      {
        update_time_integrator();
        clear_files = false;
      }

  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  update_time_integrator ()
  {
    gamma0 = 3.0/2.0;
    alpha[0] = 2.0;
    alpha[1] = -0.5;
    beta[0] = 2.0;
    beta[1] = -1.0;
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  check_time_integrator()
  {
    if (std::abs(gamma0-1.0)>1.e-12 || std::abs(alpha[0]-1.0)>1.e-12 || std::abs(alpha[1]-0.0)>1.e-12 || std::abs(beta[0]-1.0)>1.e-12 || std::abs(beta[1]-0.0)>1.e-12)
      {
        pcout << "Time integrator constants invalid!" << std::endl;
        std::abort();
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  analyse_computing_times()
  {
    double time=0.0;
    for (unsigned int i=0; i<4; ++i)
      time+=computing_times[i];
    pcout<<std::endl<<"Computing times:"
         <<std::endl<<"Step 1: Convection:\t"<<computing_times[0]/time
         <<std::endl<<"Step 2: Pressure:\t"<<computing_times[1]/time
         <<std::endl<<"Step 3: Projection:\t"<<computing_times[2]/time
         <<std::endl<<"Step 4: Viscous:\t"<<computing_times[3]/time
         <<std::endl<<"Time (Step 1-4):\t"<<time<<std::endl;
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_penalty_parameter(double &factor) const
  {
    // triangular/tetrahedral elements: penalty parameter = stab_factor*(p+1)(p+d)/dim * surface/volume
//  factor = stab_factor * (fe_degree +1.0) * (fe_degree + dim) / dim;

    // quadrilateral/hexahedral elements: penalty parameter = stab_factor*(p+1)(p+1) * surface/volume
    factor = stab_factor * (fe_degree +1.0) * (fe_degree + 1.0);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_penalty_parameter_pressure(double &factor) const
  {
    // triangular/tetrahedral elements: penalty parameter = stab_factor*(p+1)(p+d)/dim * surface/volume
//  factor = stab_factor * (fe_degree_p +1.0) * (fe_degree_p + dim) / dim;

    // quadrilateral/hexahedral elements: penalty parameter = stab_factor*(p+1)(p+1) * surface/volume
    factor = stab_factor * (fe_degree_p +1.0) * (fe_degree_p + 1.0);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  compute_lift_and_drag()
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval_velocity(data.back(),true,0,0);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree+1,1,value_type> fe_eval_pressure(data.back(),true,1,0);

    Tensor<1,dim,value_type> Force;
    for (unsigned int d=0; d<dim; ++d)
      Force[d] = 0.0;

    for (unsigned int face=data.back().n_macro_inner_faces(); face<(data.back().n_macro_inner_faces()+data.back().n_macro_boundary_faces()); face++)
      {
        fe_eval_velocity.reinit (face);
        fe_eval_velocity.read_dof_values(solution_n,0);
        fe_eval_velocity.evaluate(false,true);

        fe_eval_pressure.reinit (face);
        fe_eval_pressure.read_dof_values(solution_n,dim);
        fe_eval_pressure.evaluate(true,false);

        if (data.back().get_boundary_indicator(face) == 2)
          {
            for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
              {
                VectorizedArray<value_type> pressure = fe_eval_pressure.get_value(q);
                Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
                Tensor<2,dim,VectorizedArray<value_type> > velocity_gradient = fe_eval_velocity.get_gradient(q);
                fe_eval_velocity.submit_value(pressure*normal -  make_vectorized_array<value_type>(viscosity)*
                                              (velocity_gradient+transpose(velocity_gradient))*normal,q);
              }
            Tensor<1,dim,VectorizedArray<value_type> > Force_local = fe_eval_velocity.integrate_value();

            // sum over all entries of VectorizedArray
            for (unsigned int d=0; d<dim; ++d)
              for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                Force[d] += Force_local[d][n];
          }
      }
    Force = Utilities::MPI::sum(Force,MPI_COMM_WORLD);

    // compute lift and drag coefficients (c = (F/rho)/(1/2 U² D)
    const double U = Um * (dim==2 ? 2./3. : 4./9.);
    const double H = 0.41;
    if (dim == 2)
      Force *= 2.0/pow(U,2.0)/D;
    else if (dim == 3)
      Force *= 2.0/pow(U,2.0)/D/H;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::string filename_drag, filename_lift;
        filename_drag = "drag_refine" + Utilities::int_to_string(data.back().get_dof_handler(1).get_tria().n_levels()-1) + ".txt"; //filename_drag = "drag.txt";
        filename_lift = "lift_refine" + Utilities::int_to_string(data.back().get_dof_handler(1).get_tria().n_levels()-1) + ".txt"; //filename_lift = "lift.txt";

        std::ofstream f_drag,f_lift;
        if (clear_files)
          {
            f_drag.open(filename_drag.c_str(),std::ios::trunc);
            f_lift.open(filename_lift.c_str(),std::ios::trunc);
          }
        else
          {
            f_drag.open(filename_drag.c_str(),std::ios::app);
            f_lift.open(filename_lift.c_str(),std::ios::app);
          }
        f_drag<<std::scientific<<std::setprecision(6)<<time+time_step<<"\t"<<Force[0]<<std::endl;
        f_drag.close();
        f_lift<<std::scientific<<std::setprecision(6)<<time+time_step<<"\t"<<Force[1]<<std::endl;
        f_lift.close();
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  compute_pressure_difference()
  {
    double pressure_1 = 0.0, pressure_2 = 0.0;
    unsigned int counter_1 = 0, counter_2 = 0;

    Point<dim> point_1, point_2;
    if (dim == 2)
      {
        Point<dim> point_1_2D(0.45,0.2), point_2_2D(0.55,0.2);
        point_1 = point_1_2D;
        point_2 = point_2_2D;
      }
    else if (dim == 3)
      {
        Point<dim> point_1_3D(0.45,0.2,0.205), point_2_3D(0.55,0.2,0.205);
        point_1 = point_1_3D;
        point_2 = point_2_3D;
      }

    // serial computation
//  Vector<double> value_1(1), value_2(1);
//  VectorTools::point_value(mapping,data.back().get_dof_handler(1),solution_n[dim],point_1,value_1);
//  pressure_1 = value_1(0);
//  VectorTools::point_value(mapping,data.back().get_dof_handler(1),solution_n[dim],point_2,value_2);
//  pressure_2 = value_2(0);

    // parallel computation
    const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
    cell_point_1 = GridTools::find_active_cell_around_point (mapping,data.back().get_dof_handler(1), point_1);
    if (cell_point_1.first->is_locally_owned())
      {
        counter_1 = 1;
        //std::cout<< "Point 1 found on Processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;

        Vector<double> value(1);
        my_point_value(mapping,data.back().get_dof_handler(1),solution_n[dim],cell_point_1,value);
        pressure_1 = value(0);
      }
    counter_1 = Utilities::MPI::sum(counter_1,MPI_COMM_WORLD);
    pressure_1 = Utilities::MPI::sum(pressure_1,MPI_COMM_WORLD);
    pressure_1 = pressure_1/counter_1;

    const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
    cell_point_2 = GridTools::find_active_cell_around_point (mapping,data.back().get_dof_handler(1), point_2);
    if (cell_point_2.first->is_locally_owned())
      {
        counter_2 = 1;
        //std::cout<< "Point 2 found on Processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;

        Vector<double> value(1);
        my_point_value(mapping,data.back().get_dof_handler(1),solution_n[dim],cell_point_2,value);
        pressure_2 = value(0);
      }
    counter_2 = Utilities::MPI::sum(counter_2,MPI_COMM_WORLD);
    pressure_2 = Utilities::MPI::sum(pressure_2,MPI_COMM_WORLD);
    pressure_2 = pressure_2/counter_2;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::string filename = "pressure_difference_refine" + Utilities::int_to_string(data.back().get_dof_handler(1).get_tria().n_levels()-1) + ".txt"; // filename = "pressure_difference.txt";

        std::ofstream f;
        if (clear_files)
          {
            f.open(filename.c_str(),std::ios::trunc);
          }
        else
          {
            f.open(filename.c_str(),std::ios::app);
          }
        f << std::scientific << std::setprecision(6) << time+time_step << "\t" << pressure_1-pressure_2 << std::endl;
        f.close();
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  my_point_value(const Mapping<dim> &mapping,
                 const DoFHandler<dim> &dof_handler,
                 const parallel::distributed::Vector<double> &solution,
                 const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > &cell_point,
                 Vector<double> &value)
  {
    const FiniteElement<dim> &fe = dof_handler.get_fe();
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,ExcInternalError());

    const Quadrature<dim> quadrature (GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
    fe_values.reinit(cell_point.first);

    // then use this to get at the values of
    // the given fe_function at this point
    std::vector<Vector<double> > u_value(1, Vector<double> (fe.n_components()));
    fe_values.get_function_values(solution, u_value);
    value = u_value[0];
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal) const
  {
    parallel::distributed::Vector<value_type> src(laplace_diagonal);
    data.back().loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal,
                      &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal_face,
                      &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal_boundary_face,
                      this, laplace_diagonal, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal, unsigned int level) const
  {
    parallel::distributed::Vector<value_type> src(laplace_diagonal);
    data[level].loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal,
                      &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal_face,
                      &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal_boundary_face,
                      this, laplace_diagonal, src);

    if (pure_dirichlet_bc)
      {
        parallel::distributed::Vector<value_type> vec1(laplace_diagonal);
        for (unsigned int i=0; i<vec1.local_size(); ++i)
          vec1.local_element(i) = 1.;
        parallel::distributed::Vector<value_type> d;
        d.reinit(laplace_diagonal);
        apply_pressure(vec1,d,level);
        double length = vec1*vec1;
        double factor = vec1*d;
        laplace_diagonal.add(-2./length,d,factor/pow(length,2.),vec1);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_laplace_diagonal (const MatrixFree<dim,value_type>        &data,
                          parallel::distributed::Vector<double>     &dst,
                          const parallel::distributed::Vector<double> &,
                          const std::pair<unsigned int,unsigned int>    &cell_range) const
  {
    FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        pressure.reinit (cell);

        VectorizedArray<value_type> local_diagonal_vector[pressure.tensor_dofs_per_cell];
        for (unsigned int j=0; j<pressure.dofs_per_cell; ++j)
          {
            for (unsigned int i=0; i<pressure.dofs_per_cell; ++i)
              pressure.begin_dof_values()[i] = make_vectorized_array(0.);
            pressure.begin_dof_values()[j] = make_vectorized_array(1.);
            pressure.evaluate (false,true,false);
            for (unsigned int q=0; q<pressure.n_q_points; ++q)
              {
                pressure.submit_gradient (pressure.get_gradient(q), q);
              }
            pressure.integrate (false,true);
            local_diagonal_vector[j] = pressure.begin_dof_values()[j];
          }
        for (unsigned int j=0; j<pressure.dofs_per_cell; ++j)
          pressure.begin_dof_values()[j] = local_diagonal_vector[j];
        pressure.distribute_local_to_global (dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_laplace_diagonal_face (const MatrixFree<dim,value_type>       &data,
                               parallel::distributed::Vector<double>   &dst,
                               const parallel::distributed::Vector<double> &,
                               const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval_neighbor(data,false,1,1);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);
        fe_eval_neighbor.reinit (face);

        /*VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
                 std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
          (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5   *stab_factor; */

        double factor = 1.;
        calculate_penalty_parameter_pressure(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction())+std::abs(fe_eval_neighbor.get_normal_volume_fraction()))/2.0 * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = std::max(fe_eval.read_cell_data(array_penalty_parameter[level]),fe_eval_neighbor.read_cell_data(array_penalty_parameter[level])) * (value_type)factor;

        // element-
        VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          {
            for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
            for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
              fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);

            fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);

            fe_eval.evaluate(true,true);
            fe_eval_neighbor.evaluate(true,true);

            for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
              {
                VectorizedArray<value_type> valueM = fe_eval.get_value(q);
                VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);

                VectorizedArray<value_type> jump_value = valueM - valueP;
                VectorizedArray<value_type> average_gradient =
                  ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
                average_gradient = average_gradient - jump_value * sigmaF;

                fe_eval.submit_normal_gradient(-0.5*jump_value,q);
                fe_eval.submit_value(-average_gradient,q);
              }
            fe_eval.integrate(true,true);
            local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
          }
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
        fe_eval.distribute_local_to_global(dst);

        // neighbor (element+)
        VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
        for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
          {
            for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
            for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
              fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);

            fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array(1.);

            fe_eval.evaluate(true,true);
            fe_eval_neighbor.evaluate(true,true);

            for (unsigned int q=0; q<fe_eval_neighbor.n_q_points; ++q)
              {
                VectorizedArray<value_type> valueM = fe_eval.get_value(q);
                VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);

                VectorizedArray<value_type> jump_value = valueM - valueP;
                VectorizedArray<value_type> average_gradient =
                  ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
                average_gradient = average_gradient - jump_value * sigmaF;

                fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
                fe_eval_neighbor.submit_value(average_gradient,q);
              }
            fe_eval_neighbor.integrate(true,true);
            local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
          }
        for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
          fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];
        fe_eval_neighbor.distribute_local_to_global(dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_laplace_diagonal_boundary_face (const MatrixFree<dim,value_type>        &data,
                                        parallel::distributed::Vector<double>     &dst,
                                        const parallel::distributed::Vector<double> &,
                                        const std::pair<unsigned int,unsigned int>    &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);

        //VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
        //  (value_type)(fe_degree * (fe_degree + 1.0))  *stab_factor;

        double factor = 1.;
        calculate_penalty_parameter_pressure(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = fe_eval.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

        VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          {
            for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              {
                fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
              }
            fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);
            fe_eval.evaluate(true,true);

            for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
              {
                if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
                  {
                    //set pressure gradient in normal direction to zero, i.e. pressure+ = pressure-, grad+ = -grad-
                    VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
                    VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
                    average_gradient = average_gradient - jump_value * sigmaF;

                    fe_eval.submit_normal_gradient(-0.5*jump_value,q);
                    fe_eval.submit_value(-average_gradient,q);
                  }
                else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // outflow boundaries
                  {
                    //set pressure to zero, i.e. pressure+ = - pressure- , grad+ = grad-
                    VectorizedArray<value_type> valueM = fe_eval.get_value(q);

                    VectorizedArray<value_type> jump_value = 2.0*valueM;
                    VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
                    average_gradient = average_gradient - jump_value * sigmaF;

                    fe_eval.submit_normal_gradient(-0.5*jump_value,q);
                    fe_eval.submit_value(-average_gradient,q);
                  }
              }
            fe_eval.integrate(true,true);
            local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
          }
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
        fe_eval.distribute_local_to_global(dst);
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_diagonal_viscous(parallel::distributed::Vector<value_type> &diagonal, unsigned int level) const
  {
    parallel::distributed::Vector<value_type> src(diagonal);
    data[level].loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_diagonal_viscous,
                      &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_diagonal_viscous_face,
                      &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_diagonal_viscous_boundary_face,
                      this, diagonal, src);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_diagonal_viscous (const MatrixFree<dim,value_type>        &data,
                          parallel::distributed::Vector<double>   &dst,
                          const parallel::distributed::Vector<double> &src,
                          const std::pair<unsigned int,unsigned int>  &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> velocity (data,0,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);

        VectorizedArray<value_type> local_diagonal_vector[velocity.tensor_dofs_per_cell];
        for (unsigned int j=0; j<velocity.dofs_per_cell; ++j)
          {
            for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
              velocity.begin_dof_values()[i] = make_vectorized_array(0.);
            velocity.begin_dof_values()[j] = make_vectorized_array(1.);
            velocity.evaluate (true,true,false);
            for (unsigned int q=0; q<velocity.n_q_points; ++q)
              {
                velocity.submit_value (gamma0/time_step*velocity.get_value(q), q);
                velocity.submit_gradient (make_vectorized_array<value_type>(viscosity)*velocity.get_gradient(q), q);
              }
            velocity.integrate (true,true);
            local_diagonal_vector[j] = velocity.begin_dof_values()[j];
          }
        for (unsigned int j=0; j<velocity.dofs_per_cell; ++j)
          velocity.begin_dof_values()[j] = local_diagonal_vector[j];
        velocity.distribute_local_to_global (dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_diagonal_viscous_face (const MatrixFree<dim,value_type>       &data,
                               parallel::distributed::Vector<double>   &dst,
                               const parallel::distributed::Vector<double> &src,
                               const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,0,0);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);
        fe_eval_neighbor.reinit (face);

        double factor = 1.;
        calculate_penalty_parameter(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction())+std::abs(fe_eval_neighbor.get_normal_volume_fraction()))/2.0 * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = std::max(fe_eval.read_cell_data(array_penalty_parameter[level]),fe_eval_neighbor.read_cell_data(array_penalty_parameter[level])) * (value_type)factor;

        // element-
        VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          {
            for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
            for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
              fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);

            fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);

            fe_eval.evaluate(true,true);
            fe_eval_neighbor.evaluate(true,true);

            for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
              {
                VectorizedArray<value_type> uM = fe_eval.get_value(q);
                VectorizedArray<value_type> uP = fe_eval_neighbor.get_value(q);

                VectorizedArray<value_type> jump_value = uM - uP;
                VectorizedArray<value_type> average_gradient =
                  ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * make_vectorized_array<value_type>(0.5);
                average_gradient = average_gradient - jump_value * sigmaF;

                fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
                fe_eval.submit_value(-viscosity*average_gradient,q);
              }
            fe_eval.integrate(true,true);
            local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
          }
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
        fe_eval.distribute_local_to_global(dst);

        // neighbor (element+)
        VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
        for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
          {
            for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
            for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
              fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);

            fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array(1.);

            fe_eval.evaluate(true,true);
            fe_eval_neighbor.evaluate(true,true);

            for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
              {
                VectorizedArray<value_type> uM = fe_eval.get_value(q);
                VectorizedArray<value_type> uP = fe_eval_neighbor.get_value(q);

                VectorizedArray<value_type> jump_value = uM - uP;
                VectorizedArray<value_type> average_gradient =
                  ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * make_vectorized_array<value_type>(0.5);
                average_gradient = average_gradient - jump_value * sigmaF;

                fe_eval_neighbor.submit_normal_gradient(-0.5*viscosity*jump_value,q);
                fe_eval_neighbor.submit_value(viscosity*average_gradient,q);
              }
            fe_eval_neighbor.integrate(true,true);
            local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
          }
        for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
          fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];
        fe_eval_neighbor.distribute_local_to_global(dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_diagonal_viscous_boundary_face (const MatrixFree<dim,value_type>      &data,
                                        parallel::distributed::Vector<double>   &dst,
                                        const parallel::distributed::Vector<double> &src,
                                        const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);

        double factor = 1.;
        calculate_penalty_parameter(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = fe_eval.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

        VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          {
            for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              {
                fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
              }
            fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);
            fe_eval.evaluate(true,true);

            for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
              {
                if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
                  {
                    // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
                    VectorizedArray<value_type> uM = fe_eval.get_value(q);
                    VectorizedArray<value_type> uP = -uM;

                    VectorizedArray<value_type> jump_value = uM - uP;
                    VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
                    average_gradient = average_gradient - jump_value * sigmaF;

                    fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
                    fe_eval.submit_value(-viscosity*average_gradient,q);
                  }
                else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
                  {
                    // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
                    VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
                    VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
                    fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
                    fe_eval.submit_value(-viscosity*average_gradient,q);
                  }
              }
            fe_eval.integrate(true,true);
            local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
          }
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
        fe_eval.distribute_local_to_global(dst);
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  rhs_convection (const std::vector<parallel::distributed::Vector<value_type> >   &src,
                  std::vector<parallel::distributed::Vector<value_type> >     &dst)
  {
    for (unsigned int d=0; d<dim; ++d)
      dst[d] = 0;

    // data.loop
    data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_convection,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_convection_face,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_convection_boundary_face,
                        this, dst, src);
    // data.cell_loop
    data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_mass_matrix,
                          this, dst, dst);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  compute_rhs (std::vector<parallel::distributed::Vector<value_type> >  &dst)
  {
    for (unsigned int d=0; d<dim; ++d)
      dst[d] = 0;

    // data.loop
    data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_compute_rhs,this, dst, dst);
    // data.cell_loop
    data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_mass_matrix,
                          this, dst, dst);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_viscous (const parallel::distributed::Vector<value_type>  &src,
                 parallel::distributed::Vector<value_type>     &dst) const
  {
    dst = 0;

    data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous_face,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous_boundary_face,
                        this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_viscous (const parallel::distributed::Vector<value_type>  &src,
                 parallel::distributed::Vector<value_type>     &dst,
                 const unsigned int                &level) const
  {
    //dst = 0;
    data[level].loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous_face,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous_boundary_face,
                        this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  rhs_viscous (const std::vector<parallel::distributed::Vector<value_type> >  &src,
               std::vector<parallel::distributed::Vector<value_type> >     &dst)
  {
    for (unsigned int d=0; d<dim; ++d)
      dst[d] = 0;

    data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_viscous,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_viscous_face,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_viscous_boundary_face,
                        this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_convection (const MatrixFree<dim,value_type>              &data,
                        std::vector<parallel::distributed::Vector<double> >     &dst,
                        const std::vector<parallel::distributed::Vector<double> > &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const
  {
    // inexact integration  (data,0,0) : second argument: which dof-handler, third argument: which quadrature
//  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);

    // exact integration of convective term
    FEEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> velocity (data,0,2);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values(src,0);
        velocity.evaluate (true,false,false);

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            // nonlinear convective flux F(u) = uu
            Tensor<1,dim,VectorizedArray<value_type> > u = velocity.get_value(q);
            Tensor<2,dim,VectorizedArray<value_type> > F;
            outer_product(F,u,u);
            velocity.submit_gradient (F, q);
          }
        velocity.integrate (false,true);
        velocity.distribute_local_to_global (dst,0);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_convection_face (const MatrixFree<dim,value_type>               &data,
                             std::vector<parallel::distributed::Vector<double> >     &dst,
                             const std::vector<parallel::distributed::Vector<double> > &src,
                             const std::pair<unsigned int,unsigned int>          &face_range) const
  {
    // inexact integration
//  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);
//  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval_neighbor(data,false,0,0);

    // exact integration
    FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval(data,true,0,2);
    FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_neighbor(data,false,0,2);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);
        fe_eval_neighbor.reinit (face);

        fe_eval.read_dof_values(src,0);
        fe_eval.evaluate(true,false);
        fe_eval_neighbor.read_dof_values(src,0);
        fe_eval_neighbor.evaluate(true,false);

//    VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);
//    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
//    {
//      Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
//      Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
//      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
//      VectorizedArray<value_type> uM_n = uM*normal;
//      VectorizedArray<value_type> uP_n = uP*normal;
//      VectorizedArray<value_type> lambda_qpoint = std::max(std::abs(uM_n), std::abs(uP_n));
//      lambda = std::max(lambda_qpoint,lambda);
//    }

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
          {
            Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
            Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
            Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
            VectorizedArray<value_type> uM_n = uM*normal;
            VectorizedArray<value_type> uP_n = uP*normal;
            VectorizedArray<value_type> lambda;

            // calculation of lambda according to Hesthaven & Warburton
//      for(unsigned int k=0;k<lambda.n_array_elements;++k)
//        lambda[k] = std::abs(uM_n[k]) > std::abs(uP_n[k]) ? std::abs(uM_n[k]) : std::abs(uP_n[k]);//lambda = std::max(std::abs(uM_n), std::abs(uP_n));
            // calculation of lambda according to Hesthaven & Warburton

            // calculation of lambda according to Shahbazi et al.
            Tensor<2,dim,VectorizedArray<value_type> > unity_tensor;
            for (unsigned int d=0; d<dim; ++d)
              unity_tensor[d][d] = 1.0;
            Tensor<2,dim,VectorizedArray<value_type> > flux_jacobian_M, flux_jacobian_P;
            outer_product(flux_jacobian_M,uM,normal);
            outer_product(flux_jacobian_P,uP,normal);
            flux_jacobian_M += uM_n*unity_tensor;
            flux_jacobian_P += uP_n*unity_tensor;

            // calculate maximum absolute eigenvalue of flux_jacobian_M: max |lambda(flux_jacobian_M)|
            VectorizedArray<value_type> lambda_max_m = make_vectorized_array<value_type>(0.0);
            for (unsigned int n=0; n<lambda_max_m.n_array_elements; ++n)
              {
                LAPACKFullMatrix<value_type> FluxJacobianM(dim);
                for (unsigned int i=0; i<dim; ++i)
                  for (unsigned int j=0; j<dim; ++j)
                    FluxJacobianM(i,j) = flux_jacobian_M[i][j][n];
                FluxJacobianM.compute_eigenvalues();
                for (unsigned int d=0; d<dim; ++d)
                  lambda_max_m[n] = std::max(lambda_max_m[n],std::abs(FluxJacobianM.eigenvalue(d)));
              }

            // calculate maximum absolute eigenvalue of flux_jacobian_P: max |lambda(flux_jacobian_P)|
            VectorizedArray<value_type> lambda_max_p = make_vectorized_array<value_type>(0.0);
            for (unsigned int n=0; n<lambda_max_p.n_array_elements; ++n)
              {
                LAPACKFullMatrix<value_type> FluxJacobianP(dim);
                for (unsigned int i=0; i<dim; ++i)
                  for (unsigned int j=0; j<dim; ++j)
                    FluxJacobianP(i,j) = flux_jacobian_P[i][j][n];
                FluxJacobianP.compute_eigenvalues();
                for (unsigned int d=0; d<dim; ++d)
                  lambda_max_p[n] = std::max(lambda_max_p[n],std::abs(FluxJacobianP.eigenvalue(d)));
              }
            // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
            lambda = std::max(lambda_max_m, lambda_max_p);
            // calculation of lambda according to Shahbazi et al.

            Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
            Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
            Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

            fe_eval.submit_value(-lf_flux,q);
            fe_eval_neighbor.submit_value(lf_flux,q);
          }
        fe_eval.integrate(true,false);
        fe_eval.distribute_local_to_global(dst,0);
        fe_eval_neighbor.integrate(true,false);
        fe_eval_neighbor.distribute_local_to_global(dst,0);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_convection_boundary_face (const MatrixFree<dim,value_type>            &data,
                                      std::vector<parallel::distributed::Vector<double> >    &dst,
                                      const std::vector<parallel::distributed::Vector<double> >  &src,
                                      const std::pair<unsigned int,unsigned int>         &face_range) const
  {
    // inexact integration
//    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);

    // exact integration
    FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval(data,true,0,2);

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);
        fe_eval.read_dof_values(src,0);
        fe_eval.evaluate(true,false);

        /*  VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);
          if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
          {
            for(unsigned int q=0;q<fe_eval.n_q_points;++q)
            {
              Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);

              Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
              Tensor<1,dim,VectorizedArray<value_type> > g_n;
              for(unsigned int d=0;d<dim;++d)
              {
                AnalyticalSolution<dim> dirichlet_boundary(d,time);
                value_type array [VectorizedArray<value_type>::n_array_elements];
                for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                {
                  Point<dim> q_point;
                  for (unsigned int d=0; d<dim; ++d)
                  q_point[d] = q_points[d][n];
                  array[n] = dirichlet_boundary.value(q_point);
                }
                g_n[d].load(&array[0]);
              }
              Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g_n;
              Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
              VectorizedArray<value_type> uM_n = uM*normal;
              VectorizedArray<value_type> uP_n = uP*normal;
              VectorizedArray<value_type> lambda_qpoint = std::max(std::abs(uM_n), std::abs(uP_n));
              lambda = std::max(lambda_qpoint,lambda);
            }
          } */

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
          {
            if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
              {
                // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
                Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);

                Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
                Tensor<1,dim,VectorizedArray<value_type> > g_n;
                for (unsigned int d=0; d<dim; ++d)
                  {
                    AnalyticalSolution<dim> dirichlet_boundary(d,time);
                    value_type array [VectorizedArray<value_type>::n_array_elements];
                    for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                      {
                        Point<dim> q_point;
                        for (unsigned int d=0; d<dim; ++d)
                          q_point[d] = q_points[d][n];
                        array[n] = dirichlet_boundary.value(q_point);
                      }
                    g_n[d].load(&array[0]);
                  }

                Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g_n;
                Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
                VectorizedArray<value_type> uM_n = uM*normal;
                VectorizedArray<value_type> uP_n = uP*normal;
                VectorizedArray<value_type> lambda;
                // calculation of lambda according to Hesthaven & Warburton
//        for(unsigned int k=0;k<lambda.n_array_elements;++k)
//          lambda[k] = std::abs(uM_n[k]) > std::abs(uP_n[k]) ? std::abs(uM_n[k]) : std::abs(uP_n[k]);
                // calculation of lambda according to Hesthaven & Warburton

                // calculation of lambda according to Shahbazi et al.
                Tensor<2,dim,VectorizedArray<value_type> > unity_tensor;
                for (unsigned int d=0; d<dim; ++d)
                  unity_tensor[d][d] = 1.0;
                Tensor<2,dim,VectorizedArray<value_type> > flux_jacobian_M, flux_jacobian_P;
                outer_product(flux_jacobian_M,uM,normal);
                outer_product(flux_jacobian_P,uP,normal);
                flux_jacobian_M += uM_n*unity_tensor;
                flux_jacobian_P += uP_n*unity_tensor;

                // calculate maximum absolute eigenvalue of flux_jacobian_M: max |lambda(flux_jacobian_M)|
                VectorizedArray<value_type> lambda_max_m = make_vectorized_array<value_type>(0.0);
                for (unsigned int n=0; n<lambda_max_m.n_array_elements; ++n)
                  {
                    LAPACKFullMatrix<value_type> FluxJacobianM(dim);
                    for (unsigned int i=0; i<dim; ++i)
                      for (unsigned int j=0; j<dim; ++j)
                        FluxJacobianM(i,j) = flux_jacobian_M[i][j][n];
                    FluxJacobianM.compute_eigenvalues();
                    for (unsigned int d=0; d<dim; ++d)
                      lambda_max_m[n] = std::max(lambda_max_m[n],std::abs(FluxJacobianM.eigenvalue(d)));
                  }

                // calculate maximum absolute eigenvalue of flux_jacobian_P: max |lambda(flux_jacobian_P)|
                VectorizedArray<value_type> lambda_max_p = make_vectorized_array<value_type>(0.0);
                for (unsigned int n=0; n<lambda_max_p.n_array_elements; ++n)
                  {
                    LAPACKFullMatrix<value_type> FluxJacobianP(dim);
                    for (unsigned int i=0; i<dim; ++i)
                      for (unsigned int j=0; j<dim; ++j)
                        FluxJacobianP(i,j) = flux_jacobian_P[i][j][n];
                    FluxJacobianP.compute_eigenvalues();
                    for (unsigned int d=0; d<dim; ++d)
                      lambda_max_p[n] = std::max(lambda_max_p[n],std::abs(FluxJacobianP.eigenvalue(d)));
                  }
                // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
                lambda = std::max(lambda_max_m, lambda_max_p);
                // calculation of lambda according to Shahbazi et al.

                Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
                Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
                Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

                fe_eval.submit_value(-lf_flux,q);
              }
            else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
              {
                // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
                Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
                Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
                VectorizedArray<value_type> uM_n = uM*normal;
                VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);

                Tensor<1,dim,VectorizedArray<value_type> > jump_value;
                for (unsigned d=0; d<dim; ++d)
                  jump_value[d] = 0.0;
                Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = uM*uM_n;
                Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

                fe_eval.submit_value(-lf_flux,q);
              }
          }

        fe_eval.integrate(true,false);
        fe_eval.distribute_local_to_global(dst,0);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_compute_rhs (const MatrixFree<dim,value_type>             &data,
                     std::vector<parallel::distributed::Vector<double> >     &dst,
                     const std::vector<parallel::distributed::Vector<double> > &,
                     const std::pair<unsigned int,unsigned int>          &cell_range) const
  {
    // (data,0,0) : second argument: which dof-handler, third argument: which quadrature
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            Point<dim,VectorizedArray<value_type> > q_points = velocity.quadrature_point(q);
            Tensor<1,dim,VectorizedArray<value_type> > rhs;
            for (unsigned int d=0; d<dim; ++d)
              {
                RHS<dim> f(d,time+time_step);
                value_type array [VectorizedArray<value_type>::n_array_elements];
                for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                  {
                    Point<dim> q_point;
                    for (unsigned int d=0; d<dim; ++d)
                      q_point[d] = q_points[d][n];
                    array[n] = f.value(q_point);
                  }
                rhs[d].load(&array[0]);
              }
            velocity.submit_value (rhs, q);
          }
        velocity.integrate (true,false);
        velocity.distribute_local_to_global (dst,0);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_viscous (const MatrixFree<dim,value_type>       &data,
                       parallel::distributed::Vector<double>   &dst,
                       const parallel::distributed::Vector<double> &src,
                       const std::pair<unsigned int,unsigned int>  &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> velocity (data,0,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values(src);
        velocity.evaluate (true,true,false);

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            velocity.submit_value (gamma0/time_step*velocity.get_value(q), q);
            velocity.submit_gradient (make_vectorized_array<value_type>(viscosity)*velocity.get_gradient(q), q);
          }
        velocity.integrate (true,true);
        velocity.distribute_local_to_global (dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_viscous_face (const MatrixFree<dim,value_type>      &data,
                            parallel::distributed::Vector<double>   &dst,
                            const parallel::distributed::Vector<double> &src,
                            const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,0,0);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);
        fe_eval_neighbor.reinit (face);

        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true,true);
        fe_eval_neighbor.read_dof_values(src);
        fe_eval_neighbor.evaluate(true,true);

//      VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
//               std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
//        (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5    *stab_factor;

        double factor = 1.;
        calculate_penalty_parameter(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction())+std::abs(fe_eval_neighbor.get_normal_volume_fraction()))/2.0 * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = std::max(fe_eval.read_cell_data(array_penalty_parameter[level]),fe_eval_neighbor.read_cell_data(array_penalty_parameter[level])) * (value_type)factor;

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
          {
            VectorizedArray<value_type> uM = fe_eval.get_value(q);
            VectorizedArray<value_type> uP = fe_eval_neighbor.get_value(q);

            VectorizedArray<value_type> jump_value = uM - uP;
            VectorizedArray<value_type> average_gradient =
              ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * make_vectorized_array<value_type>(0.5);
            average_gradient = average_gradient - jump_value * sigmaF;

            fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
            fe_eval_neighbor.submit_normal_gradient(-0.5*viscosity*jump_value,q);
            fe_eval.submit_value(-viscosity*average_gradient,q);
            fe_eval_neighbor.submit_value(viscosity*average_gradient,q);
          }
        fe_eval.integrate(true,true);
        fe_eval.distribute_local_to_global(dst);
        fe_eval_neighbor.integrate(true,true);
        fe_eval_neighbor.distribute_local_to_global(dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_viscous_boundary_face (const MatrixFree<dim,value_type>       &data,
                                     parallel::distributed::Vector<double>   &dst,
                                     const parallel::distributed::Vector<double> &src,
                                     const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);

        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true,true);

//    VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
//      (value_type)(fe_degree * (fe_degree + 1.0))   *stab_factor;

        double factor = 1.;
        calculate_penalty_parameter(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = fe_eval.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
          {
            if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
              {
                // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
                VectorizedArray<value_type> uM = fe_eval.get_value(q);
                VectorizedArray<value_type> uP = -uM;

                VectorizedArray<value_type> jump_value = uM - uP;
                VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
                average_gradient = average_gradient - jump_value * sigmaF;

                fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
                fe_eval.submit_value(-viscosity*average_gradient,q);
              }
            else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
              {
                // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
                VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
                VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
                fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
                fe_eval.submit_value(-viscosity*average_gradient,q);
              }
          }
        fe_eval.integrate(true,true);
        fe_eval.distribute_local_to_global(dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_viscous (const MatrixFree<dim,value_type>               &data,
                     std::vector<parallel::distributed::Vector<double> >     &dst,
                     const std::vector<parallel::distributed::Vector<double> > &src,
                     const std::pair<unsigned int,unsigned int>          &cell_range) const
  {
    // (data,0,0) : second argument: which dof-handler, third argument: which quadrature
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values(src,0);
        velocity.evaluate (true,false,false);

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            Tensor<1,dim,VectorizedArray<value_type> > u = velocity.get_value(q);
            velocity.submit_value (make_vectorized_array<value_type>(1.0/time_step)*u, q);
          }
        velocity.integrate (true,false);
        velocity.distribute_local_to_global (dst,0);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_viscous_face (const MatrixFree<dim,value_type>                &data,
                          std::vector<parallel::distributed::Vector<double> >     &dst,
                          const std::vector<parallel::distributed::Vector<double> > &src,
                          const std::pair<unsigned int,unsigned int>          &face_range) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_viscous_boundary_face (const MatrixFree<dim,value_type>             &data,
                                   std::vector<parallel::distributed::Vector<double> >    &dst,
                                   const std::vector<parallel::distributed::Vector<double> >  &src,
                                   const std::pair<unsigned int,unsigned int>         &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);

        /* VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
          (value_type)(fe_degree * (fe_degree + 1.0))   *stab_factor; */

        double factor = 1.;
        calculate_penalty_parameter(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = fe_eval.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
          {
            if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
              {
                // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
                Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
                Tensor<1,dim,VectorizedArray<value_type> > g_np;
                for (unsigned int d=0; d<dim; ++d)
                  {
                    AnalyticalSolution<dim> dirichlet_boundary(d,time+time_step);
                    value_type array [VectorizedArray<value_type>::n_array_elements];
                    for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                      {
                        Point<dim> q_point;
                        for (unsigned int d=0; d<dim; ++d)
                          q_point[d] = q_points[d][n];
                        array[n] = dirichlet_boundary.value(q_point);
                      }
                    g_np[d].load(&array[0]);
                  }

                VectorizedArray<value_type> nu = make_vectorized_array<value_type>(viscosity);
                fe_eval.submit_normal_gradient(-nu*g_np,q);
                fe_eval.submit_value(2.0*nu*sigmaF*g_np,q);
              }
            else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
              {
                // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
                Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
                Tensor<1,dim,VectorizedArray<value_type> > h;
                for (unsigned int d=0; d<dim; ++d)
                  {
                    NeumannBoundaryVelocity<dim> neumann_boundary(d,time+time_step);
                    value_type array [VectorizedArray<value_type>::n_array_elements];
                    for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                      {
                        Point<dim> q_point;
                        for (unsigned int d=0; d<dim; ++d)
                          q_point[d] = q_points[d][n];
                        array[n] = neumann_boundary.value(q_point);
                      }
                    h[d].load(&array[0]);
                  }
                Tensor<1,dim,VectorizedArray<value_type> > jump_value;
                for (unsigned d=0; d<dim; ++d)
                  jump_value[d] = 0.0;

                VectorizedArray<value_type> nu = make_vectorized_array<value_type>(viscosity);
                fe_eval.submit_normal_gradient(jump_value,q);
                fe_eval.submit_value(nu*h,q);
              }
          }

        fe_eval.integrate(true,true);
        fe_eval.distribute_local_to_global(dst,0);
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_inverse_mass_matrix (const parallel::distributed::Vector<value_type>  &src,
                             parallel::distributed::Vector<value_type>     &dst) const
  {
    dst = 0;

    data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_mass_matrix,
                          this, dst, src);

    dst *= time_step/gamma0;
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_mass_matrix(const MatrixFree<dim,value_type>                &data,
                          std::vector<parallel::distributed::Vector<value_type> >   &dst,
                          const std::vector<parallel::distributed::Vector<value_type> > &src,
                          const std::pair<unsigned int,unsigned int>          &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> phi(data,0,0);

    const unsigned int dofs_per_cell = phi.dofs_per_cell;

    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, dim, value_type> inverse(phi);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src,0);

        inverse.fill_inverse_JxW_values(coefficients);
        inverse.apply(coefficients,dim,phi.begin_dof_values(),phi.begin_dof_values());

        phi.set_dof_values(dst,0);
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_mass_matrix(const MatrixFree<dim,value_type>          &data,
                          parallel::distributed::Vector<value_type>     &dst,
                          const parallel::distributed::Vector<value_type> &src,
                          const std::pair<unsigned int,unsigned int>    &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi(data,0,0);

    const unsigned int dofs_per_cell = phi.dofs_per_cell;

    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.fill_inverse_JxW_values(coefficients);
        inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());

        phi.set_dof_values(dst);
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  compute_vorticity (const std::vector<parallel::distributed::Vector<value_type> >  &src,
                     std::vector<parallel::distributed::Vector<value_type> >     &dst)
  {
    for (unsigned int d=0; d<number_vorticity_components; ++d)
      dst[d] = 0;
    // data.loop
    data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_compute_vorticity,this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_compute_vorticity(const MatrixFree<dim,value_type>                  &data,
                          std::vector<parallel::distributed::Vector<value_type> >     &dst,
                          const std::vector<parallel::distributed::Vector<value_type> > &src,
                          const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
    FEEvaluation<dim,fe_degree,fe_degree+1,number_vorticity_components,value_type> phi(data,0,0);

    const unsigned int dofs_per_cell = phi.dofs_per_cell;

    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit(cell);
        velocity.read_dof_values(src,0);
        velocity.evaluate (false,true,false);

        phi.reinit(cell);

        for (unsigned int q=0; q<phi.n_q_points; ++q)
          {
            Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
            phi.submit_value (omega, q);
          }
        phi.integrate (true,false);

        inverse.fill_inverse_JxW_values(coefficients);
        inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());

        phi.set_dof_values(dst,0);
      }
  }

  template <int dim, typename FEEval>
  struct CurlCompute
  {
    static
    Tensor<1,dim,VectorizedArray<typename FEEval::number_type> >
    compute(const FEEval    &fe_eval,
            const unsigned int  q_point)
    {
      return fe_eval.get_curl(q_point);
    }
  };

  template <typename FEEval>
  struct CurlCompute<2,FEEval>
  {
    static
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> >
    compute(const FEEval    &fe_eval,
            const unsigned int  q_point)
    {
      Tensor<1,2,VectorizedArray<typename FEEval::number_type> > rot;
      Tensor<1,2,VectorizedArray<typename FEEval::number_type> > temp = fe_eval.get_gradient(q_point);
      rot[0] = temp[1];
      rot[1] = - temp[0];
      return rot;
    }
  };

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_P (parallel::distributed::Vector<value_type> &vector) const
  {
    parallel::distributed::Vector<value_type> vec1(vector);
    for (unsigned int i=0; i<vec1.local_size(); ++i)
      vec1.local_element(i) = 1.;
    double scalar = vec1*vector;
    double length = vec1*vec1;
    vector.add(-scalar/length,vec1);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  shift_pressure (parallel::distributed::Vector<value_type> &pressure)
  {
    parallel::distributed::Vector<value_type> vec1(pressure);
    for (unsigned int i=0; i<vec1.local_size(); ++i)
      vec1.local_element(i) = 1.;
    AnalyticalSolution<dim> analytical_solution(dim,time+time_step);
    double exact = analytical_solution.value(first_point);
    double current = 0.;
    if (pressure.locally_owned_elements().is_element(dof_index_first_point))
      current = pressure(dof_index_first_point);
    current = Utilities::MPI::sum(current, MPI_COMM_WORLD);
    pressure.add(exact-current,vec1);
  }


  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_pressure (const parallel::distributed::Vector<value_type>   &src,
                  parallel::distributed::Vector<value_type>     &dst) const
  {
    dst = 0;

    data.loop ( &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure,
                &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure_face,
                &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure_boundary_face,
                this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_pressure (const parallel::distributed::Vector<value_type>   &src,
                  parallel::distributed::Vector<value_type>     &dst,
                  const unsigned int                &level) const
  {
    //dst = 0;
    data[level].loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure_face,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure_boundary_face,
                        this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_pressure (const MatrixFree<dim,value_type>        &data,
                        parallel::distributed::Vector<double>     &dst,
                        const parallel::distributed::Vector<double>   &src,
                        const std::pair<unsigned int,unsigned int>    &cell_range) const
  {
    FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        pressure.reinit (cell);
        pressure.read_dof_values(src);
        pressure.evaluate (false,true,false);
        for (unsigned int q=0; q<pressure.n_q_points; ++q)
          {
            pressure.submit_gradient (pressure.get_gradient(q), q);
          }
        pressure.integrate (false,true);
        pressure.distribute_local_to_global (dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_pressure_face (const MatrixFree<dim,value_type>       &data,
                             parallel::distributed::Vector<double>   &dst,
                             const parallel::distributed::Vector<double> &src,
                             const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval_neighbor(data,false,1,1);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);
        fe_eval_neighbor.reinit (face);

        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true,true);
        fe_eval_neighbor.read_dof_values(src);
        fe_eval_neighbor.evaluate(true,true);
//      VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
//               std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
//        (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5;//   *stab_factor;

        double factor = 1.;
        calculate_penalty_parameter_pressure(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction())+std::abs(fe_eval_neighbor.get_normal_volume_fraction()))/2.0 * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = std::max(fe_eval.read_cell_data(array_penalty_parameter[level]),fe_eval_neighbor.read_cell_data(array_penalty_parameter[level])) * (value_type)factor;

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
          {
            VectorizedArray<value_type> valueM = fe_eval.get_value(q);
            VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);

            VectorizedArray<value_type> jump_value = valueM - valueP;
            VectorizedArray<value_type> average_gradient =
              ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
            average_gradient = average_gradient - jump_value * sigmaF;

            fe_eval.submit_normal_gradient(-0.5*jump_value,q);
            fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
            fe_eval.submit_value(-average_gradient,q);
            fe_eval_neighbor.submit_value(average_gradient,q);
          }
        fe_eval.integrate(true,true);
        fe_eval.distribute_local_to_global(dst);
        fe_eval_neighbor.integrate(true,true);
        fe_eval_neighbor.distribute_local_to_global(dst);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_pressure_boundary_face (const MatrixFree<dim,value_type>          &data,
                                      parallel::distributed::Vector<double>     &dst,
                                      const parallel::distributed::Vector<double>   &src,
                                      const std::pair<unsigned int,unsigned int>    &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);

        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true,true);

//    VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
//      (value_type)(fe_degree * (fe_degree + 1.0));//  *stab_factor;

        double factor = 1.;
        calculate_penalty_parameter_pressure(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = fe_eval.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
          {
            if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
              {
                //set pressure gradient in normal direction to zero, i.e. pressure+ = pressure-, grad+ = -grad-
                VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
                VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
                average_gradient = average_gradient - jump_value * sigmaF;

                fe_eval.submit_normal_gradient(-0.5*jump_value,q);
                fe_eval.submit_value(-average_gradient,q);
              }
            else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // outflow boundaries
              {
                //set pressure to zero, i.e. pressure+ = - pressure- , grad+ = grad-
                VectorizedArray<value_type> valueM = fe_eval.get_value(q);

                VectorizedArray<value_type> jump_value = 2.0*valueM;
                VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
                average_gradient = average_gradient - jump_value * sigmaF;

                fe_eval.submit_normal_gradient(-0.5*jump_value,q);
                fe_eval.submit_value(-average_gradient,q);
              }
          }
        fe_eval.integrate(true,true);
        fe_eval.distribute_local_to_global(dst);
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  rhs_pressure (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
    dst[dim] = 0;
    // data.loop
    data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_pressure,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_pressure_face,
                        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_pressure_boundary_face,
                        this, dst, src);

    if (pure_dirichlet_bc)
      {
        apply_P(dst[dim]);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_pressure (const MatrixFree<dim,value_type>                &data,
                      std::vector<parallel::distributed::Vector<double> >     &dst,
                      const std::vector<parallel::distributed::Vector<double> > &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity (data,0,1);
    FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        pressure.reinit (cell);
        velocity.read_dof_values(src,0);
        velocity.evaluate (false,true,false);
        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            VectorizedArray<value_type> divergence = velocity.get_divergence(q);
            pressure.submit_value (divergence, q);
          }
        pressure.integrate (true,false);
        pressure.distribute_local_to_global (dst,dim);
      }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_pressure_face (const MatrixFree<dim,value_type>               &data,
                           std::vector<parallel::distributed::Vector<double> >     &dst,
                           const std::vector<parallel::distributed::Vector<double> > &src,
                           const std::pair<unsigned int,unsigned int>          &face_range) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_pressure_boundary_face (const MatrixFree<dim,value_type>              &data,
                                    std::vector<parallel::distributed::Vector<double> >     &dst,
                                    const std::vector<parallel::distributed::Vector<double> > &src,
                                    const std::pair<unsigned int,unsigned int>          &face_range) const
  {
    // inexact integration
//  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure(data,true,1,1);
//  FEFaceEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity_n(data,true,0,1);
//  FEFaceEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity_nm(data,true,0,1);
//  FEFaceEvaluation<dim,fe_degree,fe_degree_p+1,number_vorticity_components,value_type> omega_n(data,true,0,1);
//  FEFaceEvaluation<dim,fe_degree,fe_degree_p+1,number_vorticity_components,value_type> omega_nm(data,true,0,1);

    // exact integration
    FEFaceEvaluation<dim,fe_degree_p,fe_degree+(fe_degree+2)/2,1,value_type> pressure(data,true,1,2);
    FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> velocity_n(data,true,0,2);
    FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> velocity_nm(data,true,0,2);
    FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> omega_n(data,true,0,2);
    FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> omega_nm(data,true,0,2);

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for (unsigned int face=face_range.first; face<face_range.second; face++)
      {
        pressure.reinit (face);
        velocity_n.reinit (face);
        velocity_n.read_dof_values(solution_n,0);
        velocity_n.evaluate (true,true);
        velocity_nm.reinit (face);
        velocity_nm.read_dof_values(solution_nm,0);
        velocity_nm.evaluate (true,true);

        omega_n.reinit (face);
        omega_n.read_dof_values(vorticity_n,0);
        omega_n.evaluate (false,true);
        omega_nm.reinit (face);
        omega_nm.read_dof_values(vorticity_nm,0);
        omega_nm.evaluate (false,true);

        //VectorizedArray<value_type> sigmaF = (std::abs( pressure.get_normal_volume_fraction()) ) *
        //  (value_type)(fe_degree * (fe_degree + 1.0)) *stab_factor;

        double factor = 1.;
        calculate_penalty_parameter_pressure(factor);
        //VectorizedArray<value_type> sigmaF = std::abs(pressure.get_normal_volume_fraction()) * (value_type)factor;
        //VectorizedArray<value_type> sigmaF = make_vectorized_array<value_type>(1./h_min) * (value_type)factor;
        VectorizedArray<value_type> sigmaF = pressure.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

        for (unsigned int q=0; q<pressure.n_q_points; ++q)
          {
            if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Inflow and wall boundaries
              {
                // p+ =  p-
                Point<dim,VectorizedArray<value_type> > q_points = pressure.quadrature_point(q);
                VectorizedArray<value_type> h;

//        NeumannBoundaryPressure<dim> neumann_boundary(1,time+time_step);
//        value_type array [VectorizedArray<value_type>::n_array_elements];
//        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//        {
//          Point<dim> q_point;
//          for (unsigned int d=0; d<dim; ++d)
//          q_point[d] = q_points[d][n];
//          array[n] = neumann_boundary.value(q_point);
//        }
//        h.load(&array[0]);

//          Tensor<1,dim,VectorizedArray<value_type> > dudt_n, rhs_n;
//          for(unsigned int d=0;d<dim;++d)
//          {
//            PressureBC_dudt<dim> neumann_boundary_pressure(d,time);
//            RHS<dim> f(d,time);
//            value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
//            value_type array_f [VectorizedArray<value_type>::n_array_elements];
//            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//            {
//              Point<dim> q_point;
//              for (unsigned int d=0; d<dim; ++d)
//              q_point[d] = q_points[d][n];
//              array_dudt[n] = neumann_boundary_pressure.value(q_point);
//              array_f[n] = f.value(q_point);
//            }
//            dudt_n[d].load(&array_dudt[0]);
//            rhs_n[d].load(&array_f[0]);
//          }
//          Tensor<1,dim,VectorizedArray<value_type> > dudt_nm, rhs_nm;
//          for(unsigned int d=0;d<dim;++d)
//          {
//            PressureBC_dudt<dim> neumann_boundary_pressure(d,time-time_step);
//            RHS<dim> f(d,time-time_step);
//            value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
//            value_type array_f [VectorizedArray<value_type>::n_array_elements];
//            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//            {
//              Point<dim> q_point;
//              for (unsigned int d=0; d<dim; ++d)
//              q_point[d] = q_points[d][n];
//              array_dudt[n] = neumann_boundary_pressure.value(q_point);
//              array_f[n] = f.value(q_point);
//            }
//            dudt_nm[d].load(&array_dudt[0]);
//            rhs_nm[d].load(&array_f[0]);
//          }

                Tensor<1,dim,VectorizedArray<value_type> > dudt_np, rhs_np;
                for (unsigned int d=0; d<dim; ++d)
                  {
                    PressureBC_dudt<dim> neumann_boundary_pressure(d,time+time_step);
                    RHS<dim> f(d,time+time_step);
                    value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
                    value_type array_f [VectorizedArray<value_type>::n_array_elements];
                    for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                      {
                        Point<dim> q_point;
                        for (unsigned int d=0; d<dim; ++d)
                          q_point[d] = q_points[d][n];
                        array_dudt[n] = neumann_boundary_pressure.value(q_point);
                        array_f[n] = f.value(q_point);
                      }
                    dudt_np[d].load(&array_dudt[0]);
                    rhs_np[d].load(&array_f[0]);
                  }

                Tensor<1,dim,VectorizedArray<value_type> > normal = pressure.get_normal_vector(q);
                Tensor<1,dim,VectorizedArray<value_type> > u_n = velocity_n.get_value(q);
                Tensor<2,dim,VectorizedArray<value_type> > grad_u_n = velocity_n.get_gradient(q);
                Tensor<1,dim,VectorizedArray<value_type> > conv_n = grad_u_n * u_n;
                Tensor<1,dim,VectorizedArray<value_type> > u_nm = velocity_nm.get_value(q);
                Tensor<2,dim,VectorizedArray<value_type> > grad_u_nm = velocity_nm.get_gradient(q);
                Tensor<1,dim,VectorizedArray<value_type> > conv_nm = grad_u_nm * u_nm;
                Tensor<1,dim,VectorizedArray<value_type> > rot_n = CurlCompute<dim,decltype(omega_n)>::compute(omega_n,q);
                Tensor<1,dim,VectorizedArray<value_type> > rot_nm = CurlCompute<dim,decltype(omega_nm)>::compute(omega_nm,q);

                // 2nd order extrapolation
//        h = - normal * (make_vectorized_array<value_type>(beta[0])*(dudt_n + conv_n + make_vectorized_array<value_type>(viscosity)*rot_n - rhs_n)
//                + make_vectorized_array<value_type>(beta[1])*(dudt_nm + conv_nm + make_vectorized_array<value_type>(viscosity)*rot_nm - rhs_nm));

                h = - normal * (dudt_np - rhs_np + make_vectorized_array<value_type>(beta[0])*(conv_n + make_vectorized_array<value_type>(viscosity)*rot_n)
                                + make_vectorized_array<value_type>(beta[1])*(conv_nm + make_vectorized_array<value_type>(viscosity)*rot_nm));

                // 1st order extrapolation
//        h = - normal * (dudt_np - rhs_np + conv_n + make_vectorized_array<value_type>(viscosity)*rot_n);

                // Stokes
//        h = - normal * (dudt_np - rhs_np + make_vectorized_array<value_type>(beta[0])*(make_vectorized_array<value_type>(viscosity)*rot_n)
//                        + make_vectorized_array<value_type>(beta[1])*(make_vectorized_array<value_type>(viscosity)*rot_nm));

                pressure.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
                pressure.submit_value(-time_step*h,q);
              }
            else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
              {
                // p+ = - p- + 2g
                Point<dim,VectorizedArray<value_type> > q_points = pressure.quadrature_point(q);
                VectorizedArray<value_type> g;

                AnalyticalSolution<dim> dirichlet_boundary(dim,time+time_step);
                value_type array [VectorizedArray<value_type>::n_array_elements];
                for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
                  {
                    Point<dim> q_point;
                    for (unsigned int d=0; d<dim; ++d)
                      q_point[d] = q_points[d][n];
                    array[n] = dirichlet_boundary.value(q_point);
                  }
                g.load(&array[0]);

                pressure.submit_normal_gradient(time_step*g,q);
                pressure.submit_value(-time_step * 2.0 *sigmaF * g,q);
              }
          }
        pressure.integrate(true,true);
        pressure.distribute_local_to_global(dst,dim);
      }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                    std::vector<parallel::distributed::Vector<value_type> >     &dst)
  {
    for (unsigned int d=0; d<dim; ++d)
      dst[d] = 0;
    // data.cell_loop
    data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_projection,this, dst, src);
    // data.cell_loop
    data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_mass_matrix,
                          this, dst, dst);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_projection (const MatrixFree<dim,value_type>              &data,
                    std::vector<parallel::distributed::Vector<double> >     &dst,
                    const std::vector<parallel::distributed::Vector<double> > &src,
                    const std::pair<unsigned int,unsigned int>          &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity (data,0,1);
    FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);
        pressure.reinit (cell);
        pressure.read_dof_values(src,dim);
        pressure.evaluate (false,true,false);
        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            Tensor<1,dim,VectorizedArray<value_type> > pressure_gradient = pressure.get_gradient(q);
            velocity.submit_value (-pressure_gradient, q);
          }
        velocity.integrate (true,false);
        velocity.distribute_local_to_global (dst,0);
      }
  }

  namespace
  {
    template <int dim>
    Point<dim> get_direction()
    {
      Point<dim> direction;
      direction[dim-1] = 1.;
      return direction;
    }

    template <int dim>
    Point<dim> get_center()
    {
      Point<dim> center;
      center[0] = 0.5;
      center[1] = 0.2;
      return center;
    }
  }

  template<int dim>
  class NavierStokesProblem
  {
  public:
    typedef typename NavierStokesOperation<dim, fe_degree, fe_degree_p>::value_type value_type;
    NavierStokesProblem(const unsigned int n_refinements);
    void run();

  private:
    void make_grid_and_dofs ();
    void write_output(std::vector<parallel::distributed::Vector<value_type>>  &solution_n,
                      std::vector<parallel::distributed::Vector<value_type>>   &vorticity,
                      const unsigned int                     timestep_number);
    void calculate_error(std::vector<parallel::distributed::Vector<value_type>> &solution_n, const double delta_t=0.0);
    void calculate_time_step();

    ConditionalOStream pcout;

    double time, time_step;

    std_cxx11::shared_ptr<Manifold<dim> > cylinder_manifold;

    parallel::distributed::Triangulation<dim> triangulation;
    FE_DGQArbitraryNodes<dim> fe;
    FE_DGQArbitraryNodes<dim> fe_p;
    MappingQ<dim>   mapping; // use higher order mapping for geometry interpolation, support points are Gauss-Lobatto points
    DoFHandler<dim> dof_handler;
    DoFHandler<dim> dof_handler_p;

    MatrixFree<dim,value_type> matrix_free_data;

    const double cfl;
    const unsigned int n_refinements;
    const double output_interval_time;

    std::set<types::boundary_id> dirichlet_boundary;
    std::set<types::boundary_id> neumann_boundary;
  };

  template<int dim>
  NavierStokesProblem<dim>::NavierStokesProblem(const unsigned int refine_steps):
    pcout (std::cout,
           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
    time(START_TIME),
    triangulation(MPI_COMM_WORLD, Triangulation<dim>::maximum_smoothing,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    fe(QGaussLobatto<1>(fe_degree+1)),
    fe_p(QGaussLobatto<1>(fe_degree_p+1)),
    mapping(fe_degree),
    dof_handler(triangulation),
    dof_handler_p(triangulation),
    cylinder_manifold(dim == 2 ?
                      static_cast<Manifold<dim>*>(new HyperBallBoundary<dim>(get_center<dim>(), 0.05)) :
                      static_cast<Manifold<dim>*>(new CylindricalManifold<dim>(get_direction<dim>(), get_center<dim>()))),
    cfl(CFL/pow(fe_degree,2.0)),
    n_refinements(refine_steps),
    output_interval_time(OUTPUT_INTERVAL_TIME)
  {
    pcout << std::endl << std::endl << std::endl
          << "/******************************************************************/" << std::endl
          << "/*                                                                */" << std::endl
          << "/*     Solver for the incompressible Navier-Stokes equations      */" << std::endl
          << "/*                                                                */" << std::endl
          << "/******************************************************************/" << std::endl
          << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::make_grid_and_dofs ()
  {
    /* --------------- Generate grid ------------------- */

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
//    const double left = -1.0, right = 1.0;
//    GridGenerator::hyper_cube(triangulation,left,right);

//    // set boundary indicator
//    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
//    for(;cell!=endc;++cell)
//    {
//    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
//    {
//    //  if ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12)||
//    //      (std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
//     if ((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
//        cell->face(face_number)->set_boundary_indicator (1);
//    }
//    }
//    triangulation.refine_global(n_refinements);
//    dirichlet_boundary.insert(0);
//    neumann_boundary.insert(1);

    // vortex problem
    const double left = -0.5, right = 0.5;
    GridGenerator::subdivided_hyper_cube(triangulation,2,left,right);

    triangulation.refine_global(n_refinements);

    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for (; cell!=endc; ++cell)
      {
        for (unsigned int face_number=0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
          {
            if (((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12) && (cell->face(face_number)->center()(1)<0))||
                ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12) && (cell->face(face_number)->center()(1)>0))||
                ((std::fabs(cell->face(face_number)->center()(1) - left)< 1e-12) && (cell->face(face_number)->center()(0)<0))||
                ((std::fabs(cell->face(face_number)->center()(1) - right)< 1e-12) && (cell->face(face_number)->center()(0)>0)))
              cell->face(face_number)->set_boundary_indicator (1);
          }
      }
    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
    // vortex problem

    // flow past cylinder
//  create_triangulation(triangulation);
//  triangulation.set_manifold(10, *cylinder_manifold);
//
//  triangulation.refine_global(n_refinements);
//  dirichlet_boundary.insert(0);
//  dirichlet_boundary.insert(2);
//  neumann_boundary.insert(1);
    // flow past cylinder

    pcout << std::endl << "Generating grid for " << dim << "-dimensional problem:" << std::endl << std::endl
          << "  element shape:" <<  (dim == 2 ? " quadrilateral elements" : " hexahedral elements") << std::endl
          << "  number of refinements:" << std::setw(10) << n_refinements << std::endl
          << "  number of cells:      " << std::setw(10) << triangulation.n_global_active_cells() << std::endl
          << "  number of faces:      " << std::setw(10) << triangulation.n_active_faces() << std::endl
          << "  number of vertices:   " << std::setw(10) << triangulation.n_vertices() << std::endl;

    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);
    dof_handler_p.distribute_dofs(fe_p);
    dof_handler.distribute_mg_dofs(fe);
    dof_handler_p.distribute_mg_dofs(fe_p);

    float ndofs_per_cell_velocity = pow(float(fe_degree+1),dim)*dim;
    float ndofs_per_cell_pressure = pow(float(fe_degree_p+1),dim);
    pcout << std::endl << "Discontinuous finite element discretisation:" << std::endl << std::endl
          << "  Velocity:" << std::endl
          << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree << std::endl
          << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_velocity << std::endl
          << "  number of dofs (velocity):\t" << std::setw(10) << dof_handler.n_dofs()*dim << std::endl << std::endl
          << "  Pressure:" << std::endl
          << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree_p << std::endl
          << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_pressure << std::endl
          << "  number of dofs (pressure):\t" << std::setw(10) << dof_handler_p.n_dofs() << std::endl;

    pcout << std::endl << "Symmetric interior penalty method:" << std::endl << std::endl
          << "  stabilisation factor: " << stab_factor << std::endl;
  }

  void create_triangulation(Triangulation<2> &tria,
                            const bool compute_in_2d = true)
  {
    HyperBallBoundary<2> boundary(Point<2>(0.5,0.2), 0.05);
    Triangulation<2> left, middle, right, tmp, tmp2;
    GridGenerator::subdivided_hyper_rectangle(left, std::vector<unsigned int>({3U, 4U}),
                                              Point<2>(), Point<2>(0.3, 0.41), false);
    GridGenerator::subdivided_hyper_rectangle(right, std::vector<unsigned int>({18U, 4U}),
                                              Point<2>(0.7, 0), Point<2>(2.5, 0.41), false);

    // create middle part first as a hyper shell
    GridGenerator::hyper_shell(middle, Point<2>(0.5, 0.2), 0.05, 0.2, 4, true);
    middle.set_manifold(0, boundary);
    middle.refine_global(1);

    // then move the vertices to the points where we want them to be to create a
    // slightly asymmetric cube with a hole
    for (Triangulation<2>::cell_iterator cell = middle.begin();
         cell != middle.end(); ++cell)
      for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
          Point<2> &vertex = cell->vertex(v);
          if (std::abs(vertex[0] - 0.7) < 1e-10 &&
              std::abs(vertex[1] - 0.2) < 1e-10)
            vertex = Point<2>(0.7, 0.205);
          else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                   std::abs(vertex[1] - 0.3) < 1e-10)
            vertex = Point<2>(0.7, 0.41);
          else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                   std::abs(vertex[1] - 0.1) < 1e-10)
            vertex = Point<2>(0.7, 0);
          else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                   std::abs(vertex[1] - 0.4) < 1e-10)
            vertex = Point<2>(0.5, 0.41);
          else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                   std::abs(vertex[1] - 0.0) < 1e-10)
            vertex = Point<2>(0.5, 0.0);
          else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                   std::abs(vertex[1] - 0.3) < 1e-10)
            vertex = Point<2>(0.3, 0.41);
          else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                   std::abs(vertex[1] - 0.1) < 1e-10)
            vertex = Point<2>(0.3, 0);
          else if (std::abs(vertex[0] - 0.3) < 1e-10 &&
                   std::abs(vertex[1] - 0.2) < 1e-10)
            vertex = Point<2>(0.3, 0.205);
          else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                   std::abs(vertex[1] - 0.13621) < 1e-4)
            vertex = Point<2>(0.59, 0.11);
          else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                   std::abs(vertex[1] - 0.26379) < 1e-4)
            vertex = Point<2>(0.59, 0.29);
          else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                   std::abs(vertex[1] - 0.13621) < 1e-4)
            vertex = Point<2>(0.41, 0.11);
          else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                   std::abs(vertex[1] - 0.26379) < 1e-4)
            vertex = Point<2>(0.41, 0.29);
        }

    // refine once to create the same level of refinement as in the
    // neighboring domains
    middle.refine_global(1);

    // must copy the triangulation because we cannot merge triangulations with
    // refinement...
    GridGenerator::flatten_triangulation(middle, tmp2);

    if (compute_in_2d)
      GridGenerator::merge_triangulations (tmp2, right, tria);
    else
      {
        GridGenerator::merge_triangulations (left, tmp2, tmp);
        GridGenerator::merge_triangulations (tmp, right, tria);
      }

    // Set the cylinder boundary  to 2, outflow to 1, the rest to 0.
    for (Triangulation<2>::active_cell_iterator cell=tria.begin() ;
         cell != tria.end(); ++cell)
      for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
          {
            if (std::abs(cell->face(f)->center()[0] - (compute_in_2d ? 0.3 : 0)) < 1e-12)
              cell->face(f)->set_all_boundary_ids(0);
            else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
              cell->face(f)->set_all_boundary_ids(1);
            else if (Point<2>(0.5,0.2).distance(cell->face(f)->center())<=0.05)
              {
                cell->face(f)->set_all_manifold_ids(10);
                cell->face(f)->set_all_boundary_ids(2);
              }
            else
              cell->face(f)->set_all_boundary_ids(0);
          }
  }

  void create_triangulation(Triangulation<3> &tria)
  {
    Triangulation<2> tria_2d;
    create_triangulation(tria_2d, false);
    GridGenerator::extrude_triangulation(tria_2d, 5, 0.41, tria);

    // Set the cylinder boundary  to 2, outflow to 1, the rest to 0.
    for (Triangulation<3>::active_cell_iterator cell=tria.begin() ;
         cell != tria.end(); ++cell)
      for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
          {
            if (std::abs(cell->face(f)->center()[0]) < 1e-12)
              cell->face(f)->set_all_boundary_ids(0);
            else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
              cell->face(f)->set_all_boundary_ids(1);
            else if (Point<3>(0.5,0.2,cell->face(f)->center()[2]).distance(cell->face(f)->center())<=0.05)
              {
                cell->face(f)->set_all_manifold_ids(10);
                cell->face(f)->set_all_boundary_ids(2);
              }
            else
              cell->face(f)->set_all_boundary_ids(0);
          }
  }

  template<int dim>
  void NavierStokesProblem<dim>::
  write_output(std::vector<parallel::distributed::Vector<value_type>>   &solution_n,
               std::vector<parallel::distributed::Vector<value_type>>  &vorticity,
               const unsigned int                    output_number)
  {

    // velocity
    const FESystem<dim> joint_fe (fe, dim);
    DoFHandler<dim> joint_dof_handler (dof_handler.get_tria());
    joint_dof_handler.distribute_dofs (joint_fe);
    parallel::distributed::Vector<double> joint_velocity (joint_dof_handler.n_dofs());
    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
        loc_vel_dof_indices (fe.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator joint_cell = joint_dof_handler.begin_active(), joint_endc = joint_dof_handler.end(), vel_cell = dof_handler.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell)
      {
        if (joint_cell->is_locally_owned())
          {
            joint_cell->get_dof_indices (loc_joint_dof_indices);
            vel_cell->get_dof_indices (loc_vel_dof_indices);
            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
              switch (joint_fe.system_to_base_index(i).first.first)
                {
                case 0:
                  Assert (joint_fe.system_to_base_index(i).first.second < dim,
                          ExcInternalError());
                  joint_velocity (loc_joint_dof_indices[i]) =
                    solution_n[ joint_fe.system_to_base_index(i).first.second ]
                    (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
                  break;
                default:
                  Assert (false, ExcInternalError());
                }
          }
      }

    DataOut<dim> data_out;

    std::vector<std::string> velocity_name (dim, "velocity");
    std::vector< DataComponentInterpretation::DataComponentInterpretation > component_interpretation(dim,DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (joint_dof_handler,joint_velocity, velocity_name, component_interpretation);

    // vorticity
    parallel::distributed::Vector<double> joint_vorticity (joint_dof_handler.n_dofs());
    if (dim==2)
      {
        data_out.add_data_vector (dof_handler,vorticity[0], "vorticity");
      }
    else if (dim==3)
      {
        std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
            loc_vel_dof_indices (fe.dofs_per_cell);
        typename DoFHandler<dim>::active_cell_iterator joint_cell = joint_dof_handler.begin_active(), joint_endc = joint_dof_handler.end(), vel_cell = dof_handler.begin_active();
        for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell)
          {
            if (joint_cell->is_locally_owned())
              {
                joint_cell->get_dof_indices (loc_joint_dof_indices);
                vel_cell->get_dof_indices (loc_vel_dof_indices);
                for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
                  switch (joint_fe.system_to_base_index(i).first.first)
                    {
                    case 0:
                      Assert (joint_fe.system_to_base_index(i).first.second < dim,
                              ExcInternalError());
                      joint_vorticity (loc_joint_dof_indices[i]) =
                        vorticity[ joint_fe.system_to_base_index(i).first.second ]
                        (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
                      break;
                    default:
                      Assert (false, ExcInternalError());
                    }
              }
          }
        std::vector<std::string> vorticity_name (dim, "vorticity");
        std::vector< DataComponentInterpretation::DataComponentInterpretation > component_interpretation(dim,DataComponentInterpretation::component_is_part_of_vector);
        data_out.add_data_vector (joint_dof_handler,joint_vorticity, vorticity_name, component_interpretation);
      }
    data_out.add_data_vector (dof_handler_p,solution_n[dim], "pressure");
    Vector<double> owner(triangulation.n_active_cells());
    owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector(owner,"ProcessorNumber");
    data_out.build_patches (mapping, 0); //data_out.build_patches (mapping, 3);

    std::string filename = "solution_Proc" + Utilities::int_to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
                           "_" + Utilities::int_to_string(output_number) + ".vtu";

    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);

    if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          {
            std::string filename = "solution_Proc" + Utilities::int_to_string(i) +
                                   "_" + Utilities::int_to_string(output_number) + ".vtu";
            filenames.push_back(filename.c_str());
          }
        std::string master_name = "solution_" + Utilities::int_to_string(output_number) + ".pvtu";
        std::ofstream master_output (master_name.c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }
  }

  template<int dim>
  void NavierStokesProblem<dim>::
  calculate_error(std::vector<parallel::distributed::Vector<value_type>>  &solution_n,
                  const double                        delta_t)
  {
    for (unsigned int d=0; d<dim; ++d)
      {
        Vector<double> norm_per_cell (triangulation.n_active_cells());
        VectorTools::integrate_difference (mapping,
                                           dof_handler,
                                           solution_n[d],
                                           AnalyticalSolution<dim>(d,time+delta_t),
                                           norm_per_cell,
                                           QGauss<dim>(fe.degree+2),
                                           VectorTools::L2_norm);
        double solution_norm =
          std::sqrt(Utilities::MPI::sum (norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
        pcout << "error (L2-norm) velocity u" << d+1 << ":"
              << std::setprecision(5) << std::setw(10) << solution_norm
              << std::endl;
      }
    Vector<double> norm_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (mapping,
                                       dof_handler_p,
                                       solution_n[dim],
                                       AnalyticalSolution<dim>(dim,time+delta_t),
                                       norm_per_cell,
                                       QGauss<dim>(fe.degree+2),
                                       VectorTools::L2_norm);
    double solution_norm =
      std::sqrt(Utilities::MPI::sum (norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
    pcout << "error (L2-norm) pressure p: "
          << std::setprecision(5) << std::setw(10) << solution_norm
          << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::calculate_time_step()
  {
    typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
                                                      endc = triangulation.end();

    double diameter = 0.0, min_cell_diameter = std::numeric_limits<double>::max();
    Tensor<1,dim, value_type> velocity;
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          // calculate minimum diameter
          //diameter = cell->diameter()/std::sqrt(dim); // diameter is the largest diagonal -> divide by sqrt(dim)
          diameter = cell->minimum_vertex_distance();
          if (diameter < min_cell_diameter)
            min_cell_diameter = diameter;
        }
    const double global_min_cell_diameter = -Utilities::MPI::max(-min_cell_diameter, MPI_COMM_WORLD);

    pcout << std::endl << "Temporal discretisation:" << std::endl << std::endl
          << "  High order dual splitting scheme (2nd order)" << std::endl << std::endl
          << "Calculation of time step size:" << std::endl << std::endl
          << "  h_min: " << std::setw(10) << global_min_cell_diameter << std::endl
          << "  u_max: " << std::setw(10) << MAX_VELOCITY << std::endl
          << "  CFL:   " << std::setw(7) << CFL << "/p²" << std::endl;

    // cfl = U_max * time_step / d_min
    time_step = cfl * global_min_cell_diameter / MAX_VELOCITY;

    // decrease time_step in order to exactly hit END_TIME
    time_step = (END_TIME-START_TIME)/(1+int((END_TIME-START_TIME)/time_step));

    //time_step = 0.2/pow(2.0,8);//2.0e-4;//0.1/pow(2.0,8);

    pcout << std::endl << "  time step size: " << std::setw(10) << time_step << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::run()
  {
    make_grid_and_dofs();

    calculate_time_step();

    NavierStokesOperation<dim, fe_degree, fe_degree_p>  navier_stokes_operation(mapping, dof_handler, dof_handler_p, time_step, dirichlet_boundary, neumann_boundary);

    // prescribe initial conditions
    for (unsigned int d=0; d<dim; ++d)
      VectorTools::interpolate(mapping, dof_handler, AnalyticalSolution<dim>(d,time), navier_stokes_operation.solution_n[d]);
    VectorTools::interpolate(mapping, dof_handler_p, AnalyticalSolution<dim>(dim,time), navier_stokes_operation.solution_n[dim]);
    navier_stokes_operation.solution_nm = navier_stokes_operation.solution_n;
//std::cout<<"processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" at line: "<<__LINE__<<std::endl;
    // compute vorticity from initial data at time t = START_TIME
    navier_stokes_operation.compute_vorticity(navier_stokes_operation.solution_n,navier_stokes_operation.vorticity_n);
    navier_stokes_operation.vorticity_nm = navier_stokes_operation.vorticity_n;

    unsigned int output_number = 0;
    write_output(navier_stokes_operation.solution_n,
                 navier_stokes_operation.vorticity_n,
                 output_number++);
    pcout << std::endl << "Write output at START_TIME t = " << START_TIME << std::endl;
    calculate_error(navier_stokes_operation.solution_n);

    const double EPSILON = 1.0e-10;
    unsigned int time_step_number = 1;

    for (; time<(END_TIME-EPSILON); time+=time_step,++time_step_number)
      {
        navier_stokes_operation.do_timestep(time,time_step,time_step_number);

        if ( (time+time_step) > (output_number*output_interval_time-EPSILON) )
          {
            write_output(navier_stokes_operation.solution_n,
                         navier_stokes_operation.vorticity_n,
                         output_number++);
            pcout << std::endl << "Write output at TIME t = " << time+time_step << std::endl;
            calculate_error(navier_stokes_operation.solution_n,time_step);
          }
      }
    navier_stokes_operation.analyse_computing_times();
  }
}

int main (int argc, char **argv)
{
  try
    {
      using namespace DG_NavierStokes;
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1); // parallel: -1

      deallog.depth_console(0);

      for (unsigned int refine_steps = refine_steps_min; refine_steps <= refine_steps_max; ++refine_steps)
        {
          NavierStokesProblem<dimension> navier_stokes_problem(refine_steps);
          navier_stokes_problem.run();
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
