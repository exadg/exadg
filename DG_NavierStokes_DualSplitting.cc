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
#include <deal.II/lac/parallel_block_vector.h>
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
#include <deal.II/distributed/tria.h>
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

#include <deal.II/integrators/laplace.h>

#include <fstream>
#include <sstream>

//#define XWALL
#define SYMMETRIC //(VISCOUS TERM)

namespace DG_NavierStokes
{
  using namespace dealii;

  const unsigned int fe_degree = 3;
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int fe_degree_xwall = 1;
  const unsigned int n_q_points_1d_xwall = 12;
  const unsigned int dimension = 3; // dimension >= 2
  const unsigned int refine_steps_min = 3;
  const unsigned int refine_steps_max = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 30.0; // Poisseuille 5.0;  Kovasznay 1.0
  const double OUTPUT_INTERVAL_TIME = 0.01;
  const double CFL = 1.0;

  const double VISCOSITY = 1./180.0;//0.005; // Taylor vortex: 0.01; vortex problem (Hesthaven): 0.025; Poisseuille 0.005; Kovasznay 0.025; Stokes 1.0

  const double MAX_VELOCITY = 30.0; // Taylor vortex: 1; vortex problem (Hesthaven): 1.5; Poisseuille 1.0; Kovasznay 4.0
  const double stab_factor = 8.0;
  const double CS = 0.17; // Smagorinsky constant

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  bool pure_dirichlet_bc = true;

  const std::string output_prefix = "solution_ch180_8_p3_gt18_f8_cs17_newvisc_bdf3_cfl1";

  const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);

  template<int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
  AnalyticalSolution (const unsigned int   component,
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

  virtual ~AnalyticalSolution(){};

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
  /*  const double T = 0.1;
    if(component == 0 && (std::abs(p[1]-1.0)<1.0e-15))
      result = t<T? (t/T) : 1.0; */
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
    //turbulent channel flow
    if(component == 0)
    {
      if(p[1]<0.99&&p[1]>-0.99)
        result = -22.0*(pow(p[1],2.0)-1.0);//*(1.0+((double)rand()/RAND_MAX)*0.0005);//*1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0*(t<T? (t/T) : 1.0);
      else
        result = 0.0;
    }
    if(component == 1|| component == 2)
    {
      result = 0.;
    }
      if(component == dim)
    result = 0.0;//(p[0]-1.0)*pressure_gradient*(t<T? (t/T) : 1.0);
    if(component >dim)
      result = 0.0;

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
//    const double pi = numbers::PI;
//    if(component == 0)
//      result = -std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    else if(component == 1)
//      result = std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    else if(component == dim)
//      result = -std::cos(2*pi*p[0])*std::cos(2*pi*p[1])*std::exp(-8.0*pi*pi*VISCOSITY*t);
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
    /*const double pi = numbers::PI;
    const double a = 0.25*pi;
    const double d = 2*a;
    if (component == 0)
      result = -a*(std::exp(a*p[0])*std::sin(a*p[1]+d*p[2]) + std::exp(a*p[2])*std::cos(a*p[0]+d*p[1]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == 1)
      result = -a*(std::exp(a*p[1])*std::sin(a*p[2]+d*p[0]) + std::exp(a*p[0])*std::cos(a*p[1]+d*p[2]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == 2)
      result = -a*(std::exp(a*p[2])*std::sin(a*p[0]+d*p[1]) + std::exp(a*p[1])*std::cos(a*p[2]+d*p[0]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == dim)
        result = -a*a*0.5*(std::exp(2*a*p[0]) + std::exp(2*a*p[1]) + std::exp(2*a*p[2]) +
                           2*std::sin(a*p[0]+d*p[1])*std::cos(a*p[2]+d*p[0])*std::exp(a*(p[1]+p[2])) +
                           2*std::sin(a*p[1]+d*p[2])*std::cos(a*p[0]+d*p[1])*std::exp(a*(p[2]+p[0])) +
                           2*std::sin(a*p[2]+d*p[0])*std::cos(a*p[1]+d*p[2])*std::exp(a*(p[0]+p[1]))) * std::exp(-2*VISCOSITY*d*d*t);*/
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

  return result;
  }

  template<int dim>
  class NeumannBoundaryVelocity : public Function<dim>
  {
  public:
    NeumannBoundaryVelocity (const unsigned int   component,
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

    virtual ~NeumannBoundaryVelocity(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double NeumannBoundaryVelocity<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
//    double t = this->get_time();
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
//    const double pi = numbers::PI;
//    if(component==0)
//    {
//      if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
//        result = 2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//      else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
//        result = -2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    }
//    else if(component==1)
//    {
//      if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
//        result = -2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//      else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
//        result = 2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    }
    return result;
  }

  template<int dim>
  class NeumannBoundaryPressure : public Function<dim>
  {
  public:
  NeumannBoundaryPressure (const unsigned int   n_components = 1,
                 const double       time = 0.) : Function<dim>(n_components, time) {}

    virtual ~NeumannBoundaryPressure(){};

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

  template<int dim>
  class RHS : public Function<dim>
  {
  public:
    RHS (const unsigned int   component,
      const double     time = 0.) : Function<dim>(1, time),time(time),component(component) {}

    virtual ~RHS(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const double time;
    const unsigned int component;
  };

  template<int dim>
  double RHS<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {

    //channel flow with periodic bc
    if(component==0)
      if(time<0.01)
        return 1.0*(1.0+((double)rand()/RAND_MAX)*0.01);
      else
        return 1.0;
    else
      return 0.0;
//  double t = this->get_time();
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
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

    virtual ~PressureBC_dudt(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double PressureBC_dudt<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
//  double t = this->get_time();
  double result = 0.0;

  //Taylor vortex (Shahbazi et al.,2007)
//  const double pi = numbers::PI;
//  if(component == 0)
//    result = (2.0*pi*pi*VISCOSITY*std::cos(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//  else if(component == 1)
//    result = (-2.0*pi*pi*VISCOSITY*std::sin(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);

  // vortex problem (Hesthaven)
//  const double pi = numbers::PI;
//  if(component == 0)
//    result = 4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//  else if(component == 1)
//    result = -4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);

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

  return result;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall> struct NavierStokesPressureMatrix;
  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall> struct NavierStokesViscousMatrix;
  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall> struct PreconditionerJacobiPressure;
  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall> struct PreconditionerJacobiPressureCoarse;
//template<int dim, int fe_degree, int fe_degree_p> struct PreconditionerJacobiViscous;

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
  
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  class MGCoarsePressure : public MGCoarseGridBase<parallel::distributed::Vector<double> >
  {
  public:
    MGCoarsePressure() {}

    void initialize(const NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &pressure,
        const PreconditionerJacobiPressureCoarse<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> & preconditioner)
    {
      ns_pressure_coarse = &pressure;
      jacobi_preconditioner_pressure_coarse = &preconditioner;
    }

    virtual void operator() (const unsigned int   level,
                             parallel::distributed::Vector<double> &dst,
                             const parallel::distributed::Vector<double> &src) const
    {
//      SolverControl solver_control (1e3, 1e-6);
      ReductionControl solver_control (1e3, 1.e-12,1e-6);
      SolverCG<parallel::distributed::Vector<double> > solver_coarse (solver_control);
      solver_coarse.solve (*ns_pressure_coarse, dst, src, *jacobi_preconditioner_pressure_coarse);
//      solver_coarse.solve (*ns_pressure_coarse, dst, src, PreconditionIdentity());
    }

    const  NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_pressure_coarse;
    const  PreconditionerJacobiPressureCoarse<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *jacobi_preconditioner_pressure_coarse;
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  class MGCoarseViscous : public MGCoarseGridBase<parallel::distributed::BlockVector<double> >
  {
  public:
     MGCoarseViscous() {}

     void initialize(const NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &viscous)
     {
       ns_viscous_coarse = &viscous;
     }

    virtual void operator() (const unsigned int   /*level*/,
                              parallel::distributed::BlockVector<double> &dst,
                              const parallel::distributed::BlockVector<double> &src) const
     {
       SolverControl solver_control (1e3, 1e-6);
       SolverCG<parallel::distributed::BlockVector<double> > solver_coarse (solver_control);
       solver_coarse.solve (*ns_viscous_coarse, dst, src, PreconditionIdentity());
     }

     const  NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_viscous_coarse;
  };

  struct SimpleSpaldingsLaw
  {
    static
  double SpaldingsLaw(double dist, double utau)
  {
    //watch out, this is not exactly Spalding's law but psi=u_+*k, which saves quite some multiplications
    const double yplus=dist*utau/VISCOSITY;
    double psi=0.0;


    if(yplus>11.0)//this is approximately where the intersection of log law and linear region lies
      psi=log(yplus)+5.17*0.41;
    else
      psi=yplus*0.41;

    double inc=10.0;
    double fn=10.0;
    int count=0;
    bool converged = false;
    while(not converged)
    {
      const double psiquad=psi*psi;
      const double exppsi=std::exp(psi);
      const double expmkmb=std::exp(-0.41*5.17);
             fn=-yplus + psi*(1./0.41)+(expmkmb)*(exppsi-(1.0)-psi-psiquad*(0.5) - psiquad*psi/(6.0) - psiquad*psiquad/(24.0));
             double dfn= 1/0.41+expmkmb*(exppsi-(1.0)-psi-psiquad*(0.5) - psiquad*psi/(6.0));

      inc=fn/dfn;

      psi-=inc;

      bool test=false;
      //do loop for all if one of the values is not converged
        if((std::abs(inc)>1.0E-14 && abs(fn)>1.0E-14&&1000>count++))
            test=true;

      converged = not test;
    }

    return psi;

    //Reichardt's law 1951
    // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
  }
  };

  template <int dim, int n_q_points_1d, typename Number>
    class EvaluationXWall
    {

    public:
    EvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                        const parallel::distributed::Vector<double>& wdist,
                        const parallel::distributed::Vector<double>& tauw):
                          mydata(matrix_free),
                          wdist(wdist),
                          tauw(tauw),
                          evaluate_value(true),
                          evaluate_gradient(true),
                          evaluate_hessian(false),
                          k(0.41),
                          km1(1.0/k),
                          B(5.17),
                          expmkmb(exp(-k*B))
      {};

    virtual ~EvaluationXWall(){};

    virtual void reinit(AlignedVector<VectorizedArray<Number> > qp_wdist,
        AlignedVector<VectorizedArray<Number> > qp_tauw,
        AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_gradwdist,
        AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_gradtauw,
        unsigned int n_q_points,
        std::vector<bool> enriched_components)
    {

      qp_enrichment.resize(n_q_points);
      qp_grad_enrichment.resize(n_q_points);
      for(unsigned int q=0;q<n_q_points;++q)
      {
        qp_enrichment[q] =  EnrichmentShapeDer(qp_wdist[q], qp_tauw[q],
            qp_gradwdist[q], qp_gradtauw[q],&(qp_grad_enrichment[q]), enriched_components);

        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        {
          if(not enriched_components.at(v))
          {
            qp_enrichment[q][v] = 0.0;
            for (unsigned int d = 0; d<dim; d++)
              qp_grad_enrichment[q][d][v] = 0.0;
          }

        }
      }

    };

    virtual void evaluate(const bool evaluate_val,
               const bool evaluate_grad,
               const bool evaluate_hess = false)
    {
      evaluate_value = evaluate_val;
      evaluate_gradient = evaluate_grad;
      //second derivative not implemented yet
      evaluate_hessian = evaluate_hess;
      Assert(not evaluate_hessian,ExcInternalError());
    }
    VectorizedArray<Number> enrichment(unsigned int q){return qp_enrichment[q];}
    Tensor<1,dim,VectorizedArray<Number> > enrichment_gradient(unsigned int q){return qp_grad_enrichment[q];}
    protected:
    VectorizedArray<Number> EnrichmentShapeDer(VectorizedArray<Number> wdist, VectorizedArray<Number> tauw,
        Tensor<1,dim,VectorizedArray<Number> > gradwdist, Tensor<1,dim,VectorizedArray<Number> > gradtauw,
        Tensor<1,dim,VectorizedArray<Number> >* gradpsi, std::vector<bool> enriched_components)
      {
           VectorizedArray<Number> density = make_vectorized_array(1.0);
//        //calculate transformation ---------------------------------------


//         LINALG::Matrix<my::numderiv2_,1> der2wdist(true);
//         if(evaluate_hessian)
//           der2wdist.Multiply(derxy2_,ewdist_);
//         LINALG::Matrix<my::numderiv2_,1> der2tauw(true);
//         if(evaluate_hessian)
//           der2tauw.Multiply(derxy2_,etauw_);
         Tensor<1,dim,VectorizedArray<Number> > gradtrans;
//         LINALG::Matrix<my::numderiv2_,1> der2trans_1(true);
//         LINALG::Matrix<my::numderiv2_,1> der2trans_2(true);
//
//         if(tauw<1.0e-10)
//           std::cerr << "tauw is almost zero"<< std::endl;;
//         if(density<1.0e-10)
//           std::cerr << "density is almost zero"<< std::endl;;
//
         const VectorizedArray<Number> utau=std::sqrt(tauw*make_vectorized_array(1.0)/density);
         const VectorizedArray<Number> fac=make_vectorized_array(0.5)/std::sqrt(density*tauw);
         const VectorizedArray<Number> wdistfac=wdist*fac;
//
         for(unsigned int sdm=0;sdm < dim;++sdm)
           gradtrans[sdm]=(utau*gradwdist[sdm]+wdistfac*gradtauw[sdm])*make_vectorized_array(1.0/VISCOSITY);

         //second derivative, first part: to be multiplied with der2psigpsc
         //second derivative, second part: to be multiplied with derpsigpsc
//         if(evaluate_hessian)
//         {
//           const Number wdistfactauwtwoinv=wdistfac/(tauw*2.0);
//
//           for(int sdm=0;sdm < my::numderiv2_;++sdm)
//           {
//             const int i[6]={0, 1, 2, 0, 0, 1};
//             const int j[6]={0, 1, 2, 1, 2, 2};
//
//             der2trans_1(sdm)=dertrans(i[sdm])*dertrans(j[sdm]);
//
//             der2trans_2(sdm)=(derwdist(j[sdm])*fac*dertauw(i[sdm])
//                               +wdistfac*der2tauw(sdm)
//                               -wdistfactauwtwoinv*dertauw(i[sdm])*dertauw(j[sdm])
//                               +dertauw(j[sdm])*fac*derwdist(i[sdm])
//                               +utau*der2wdist(sdm))*viscinv_;
//           }
//         }
         //calculate transformation done ----------------------------------

         //get enrichment function and scalar derivatives
           VectorizedArray<Number> psigp = SpaldingsLaw(wdist, utau, enriched_components)*make_vectorized_array(1.0);
           VectorizedArray<Number> derpsigpsc=DerSpaldingsLaw(psigp)*make_vectorized_array(1.0);
//         const Number der2psigpsc=Der2SpaldingsLaw(wdist, utau, psigp,derpsigpsc);
//
//         //calculate final derivatives
         Tensor<1,dim,VectorizedArray<Number> > gradpsiq;
         for(int sdm=0;sdm < dim;++sdm)
         {
           gradpsiq[sdm]=derpsigpsc*gradtrans[sdm];
         }

         (*gradpsi)=gradpsiq;
//         if(evaluate_hessian)
//           for(int sdm=0;sdm < my::numderiv2_;++sdm)
//           {
//             der2psigp(sdm)=der2psigpsc*der2trans_1(sdm);
//             der2psigp(sdm)+=derpsigpsc*der2trans_2(sdm);
//           }

        return psigp;
      }

      const MatrixFree<dim,Number> &mydata;

    const parallel::distributed::Vector<double>& wdist;
    const parallel::distributed::Vector<double>& tauw;

    private:

    bool evaluate_value;
    bool evaluate_gradient;
    bool evaluate_hessian;

    const Number k;
    const Number km1;
    const Number B;
    const Number expmkmb;

    AlignedVector<VectorizedArray<Number> > qp_enrichment;
    AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_grad_enrichment;


      VectorizedArray<Number> SpaldingsLaw(VectorizedArray<Number> dist, VectorizedArray<Number> utau, std::vector<bool> enriched_components)
      {
        //watch out, this is not exactly Spalding's law but psi=u_+*k, which saves quite some multiplications
        const VectorizedArray<Number> yplus=dist*utau*make_vectorized_array(1.0/VISCOSITY);
        VectorizedArray<Number> psi=make_vectorized_array(0.0);

        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        {
          if(enriched_components.at(v))
          {
            if(yplus[v]>11.0)//this is approximately where the intersection of log law and linear region lies
              psi[v]=log(yplus[v])+B*k;
            else
              psi[v]=yplus[v]*k;
          }
          else
            psi[v] = 0.0;
        }

        VectorizedArray<Number> inc=make_vectorized_array(10.0);
        VectorizedArray<Number> fn=make_vectorized_array(10.0);
        int count=0;
        bool converged = false;
        while(not converged)
        {
          VectorizedArray<Number> psiquad=psi*psi;
          VectorizedArray<Number> exppsi=std::exp(psi);
                 fn=-yplus + psi*make_vectorized_array(km1)+make_vectorized_array(expmkmb)*(exppsi-make_vectorized_array(1.0)-psi-psiquad*make_vectorized_array(0.5) - psiquad*psi/make_vectorized_array(6.0) - psiquad*psiquad/make_vectorized_array(24.0));
                 VectorizedArray<Number> dfn= km1+expmkmb*(exppsi-make_vectorized_array(1.0)-psi-psiquad*make_vectorized_array(0.5) - psiquad*psi/make_vectorized_array(6.0));

          inc=fn/dfn;

          psi-=inc;

          bool test=false;
          //do loop for all if one of the values is not converged
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(enriched_components.at(v))
              if((std::abs(inc[v])>1.0E-14 && abs(fn[v])>1.0E-14&&1000>count++))
                test=true;
          }
          converged = not test;
        }

        return psi;

        //Reichardt's law 1951
        // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
      }

      VectorizedArray<Number> DerSpaldingsLaw(VectorizedArray<Number> psi)
      {
        //derivative with respect to y+!
        //spaldings law according to paper (derivative)
        return make_vectorized_array(1.0)/(make_vectorized_array(1.0/k)+make_vectorized_array(expmkmb)*(std::exp(psi)-make_vectorized_array(1.0)-psi-psi*psi*make_vectorized_array(0.5)-psi*psi*psi/make_vectorized_array(6.0)));

      // Reichardt's law
      //  double yplus=dist*utau*viscinv_;
      //  return (0.4/(k_*(1.0+0.4*yplus))+7.8*(1.0/11.0*exp(-yplus/11.0)-1.0/11.0*exp(-yplus/3.0)+yplus/33.0*exp(-yplus/3.0)))*k_;
      }

      Number Der2SpaldingsLaw(Number psi,Number derpsi)
      {
        //derivative with respect to y+!
        //spaldings law according to paper (2nd derivative)
        return -make_vectorized_array(expmkmb)*(exp(psi)-make_vectorized_array(1.)-psi-psi*psi*make_vectorized_array(0.5))*derpsi*derpsi*derpsi;

        // Reichardt's law
      //  double yplus=dist*utau*viscinv_;
      //  return (-0.4*0.4/(k_*(1.0+0.4*yplus)*(1.0+0.4*yplus))+7.8*(-1.0/121.0*exp(-yplus/11.0)+(2.0/33.0-yplus/99.0)*exp(-yplus/3.0)))*k_;
      }
    };

  template <int dim, int fe_degree = 1, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
              int n_components_ = 1, typename Number = double >
    class FEEvaluationXWall : public EvaluationXWall<dim,n_q_points_1d, Number>
    {
      typedef FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> BaseClass;
      typedef Number                            number_type;
      typedef typename BaseClass::value_type    value_type;
      typedef typename BaseClass::gradient_type gradient_type;
//    private:
//    static const unsigned int n_q_points_wall_normal = 20;
//    static const unsigned int n_q_points_wall_parallel = 8;
//    public:
//    static const unsigned int n_q_points = n_q_points_wall_normal * n_q_points_wall_parallel * n_q_points_wall_parallel;
public:
    FEEvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                        const parallel::distributed::Vector<double>& wdist,
                        const parallel::distributed::Vector<double>& tauw,
                        const unsigned int            fe_no = 0,
                        const unsigned int            quad_no = 0):
                          EvaluationXWall<dim,n_q_points_1d, Number>::EvaluationXWall(matrix_free, wdist, tauw),
                          fe_eval(),
                          fe_eval_xwall(),
                          fe_eval_tauw(),
                          values(),
                          gradients(),
                          dofs_per_cell(0),
                          tensor_dofs_per_cell(0),
                          n_q_points(0),
                          enriched(false)
      {
        {
          FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> fe_eval_tmp(matrix_free,0,quad_no);
          fe_eval.resize(1,fe_eval_tmp);
        }
        {
          FEEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> fe_eval_xwall_tmp(matrix_free,3,quad_no);
          fe_eval_xwall.resize(1,fe_eval_xwall_tmp);
        }
        {
          FEEvaluation<dim,1,n_q_points_1d,1,double> fe_eval_tauw_tmp(matrix_free,2,quad_no);
          fe_eval_tauw.resize(1,fe_eval_tauw_tmp);
        }
        values.resize(fe_eval[0].n_q_points,value_type());
        gradients.resize(fe_eval[0].n_q_points,gradient_type());
        n_q_points = fe_eval[0].n_q_points;
      };

      void reinit(const unsigned int cell)
      {
#ifdef XWALL
        {
          enriched = false;
          values.resize(fe_eval[0].n_q_points,value_type());
          gradients.resize(fe_eval[0].n_q_points,gradient_type());
  //        decide if we have an enriched element via the y component of the cell center
          for (unsigned int v=0; v<EvaluationXWall<dim,n_q_points_1d, Number>::mydata.n_components_filled(cell); ++v)
          {
            typename DoFHandler<dim>::cell_iterator dcell = EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(cell, v);
//            std::cout << ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL))) << std::endl;
            if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
              enriched = true;
          }
          enriched_components.resize(VectorizedArray<Number>::n_array_elements);
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            enriched_components.at(v) = false;
          if(enriched)
          {
            //store, exactly which component of the vectorized array is enriched
            for (unsigned int v=0; v<EvaluationXWall<dim,n_q_points_1d, Number>::mydata.n_components_filled(cell); ++v)
            {
              typename DoFHandler<dim>::cell_iterator dcell = EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(cell, v);
              if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                  enriched_components.at(v) = true;
            }

            //initialize the enrichment function
            {
              fe_eval_tauw[0].reinit(cell);
              //get wall distance and wss at quadrature points
              fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
              fe_eval_tauw[0].evaluate(true, true);

              AlignedVector<VectorizedArray<Number> > cell_wdist;
              AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > cell_gradwdist;
              cell_wdist.resize(fe_eval_tauw[0].n_q_points);
              cell_gradwdist.resize(fe_eval_tauw[0].n_q_points);
              for(unsigned int q=0;q<fe_eval_tauw[0].n_q_points;++q)
              {
                cell_wdist[q] = fe_eval_tauw[0].get_value(q);
                cell_gradwdist[q] = fe_eval_tauw[0].get_gradient(q);
              }

              fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);

              fe_eval_tauw[0].evaluate(true, true);

              AlignedVector<VectorizedArray<Number> > cell_tauw;
              AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > cell_gradtauw;

              cell_tauw.resize(fe_eval_tauw[0].n_q_points);
              cell_gradtauw.resize(fe_eval_tauw[0].n_q_points);

              for(unsigned int q=0;q<fe_eval_tauw[0].n_q_points;++q)
              {
                cell_tauw[q] = fe_eval_tauw[0].get_value(q);
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                    Assert( fe_eval_tauw[0].get_value(q)[v] > 1.0e-9 ,ExcInternalError());
                }

                cell_gradtauw[q] = fe_eval_tauw[0].get_gradient(q);
              }
              EvaluationXWall<dim,n_q_points_1d, Number>::reinit(cell_wdist, cell_tauw, cell_gradwdist, cell_gradtauw, fe_eval_tauw[0].n_q_points,enriched_components);
            }
          }
          fe_eval_xwall[0].reinit(cell);
        }
#endif
        fe_eval[0].reinit(cell);
#ifdef XWALL
        if(enriched)
        {
          dofs_per_cell = fe_eval[0].dofs_per_cell + fe_eval_xwall[0].dofs_per_cell;
          tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell + fe_eval_xwall[0].tensor_dofs_per_cell;
        }
        else
        {
          dofs_per_cell = fe_eval[0].dofs_per_cell;
          tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell;
        }
#else
        dofs_per_cell = fe_eval[0].dofs_per_cell;
        tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell;
#endif
      }

      void read_dof_values (const parallel::distributed::Vector<double> &src, const parallel::distributed::Vector<double> &src_xwall)
      {

        fe_eval[0].read_dof_values(src);
#ifdef XWALL
//          if(enriched)
          {
            fe_eval_xwall[0].read_dof_values(src_xwall);
//            std::cout << "b" << std::endl;
//            for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell ; i++)
//              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//                if(not enriched_components.at(v))
//                  fe_eval_xwall[0].begin_dof_values()[i][v] = 0.0;
//              std::cout << "d" << std::endl;
//            std::cout << "e" << std::endl;
//            for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell ; i++)
//              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//                Assert(not isnan(fe_eval_xwall[0].begin_dof_values()[i][v]),ExcInternalError());
//            std::cout << "f" << std::endl;
          }

#endif
      }

      void read_dof_values (const std::vector<parallel::distributed::Vector<double> > &src, unsigned int i,const std::vector<parallel::distributed::Vector<double> > &src_xwall, unsigned int j)
      {
        fe_eval[0].read_dof_values(src,i);
#ifdef XWALL
//          if(enriched)
          {
            fe_eval_xwall[0].read_dof_values(src_xwall,j);
//            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//            {
//              if(not enriched_components.at(v))
//                for(unsigned int k = 0; k<fe_eval_xwall[0].dofs_per_cell ; k++)
//                  fe_eval_xwall[0].begin_dof_values()[k][v] = 0.0;
//            }
//            for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell ; i++)
//              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//              {
//                std::cout << (fe_eval_xwall[0].begin_dof_values()[i])[v] << " ";
//                Assert(not isnan((fe_eval_xwall[0].begin_dof_values()[i])[v]),ExcInternalError());
//              }
//            std::cout << std::endl;
          }
#endif
      }
      void read_dof_values (const parallel::distributed::BlockVector<double> &src, unsigned int i,const parallel::distributed::BlockVector<double> &src_xwall, unsigned int j)
      {
        fe_eval[0].read_dof_values(src,i);
#ifdef XWALL
        fe_eval_xwall[0].read_dof_values(src_xwall,j);
#endif
      }

      void evaluate(const bool evaluate_val,
                 const bool evaluate_grad,
                 const bool evaluate_hess = false)
      {
  fe_eval[0].evaluate(evaluate_val,evaluate_grad);
#ifdef XWALL
          if(enriched)
          {
            gradients.resize(fe_eval[0].n_q_points,gradient_type());
            values.resize(fe_eval[0].n_q_points,value_type());
            fe_eval_xwall[0].evaluate(true,evaluate_grad);
            //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
            EvaluationXWall<dim,n_q_points_1d,Number>::evaluate(evaluate_val,evaluate_grad,evaluate_hess);
            //evaluate gradient
            if(evaluate_grad)
            {
              //there are 2 parts due to chain rule
              //start with part 1
              AlignedVector<gradient_type> final_gradient;
              final_gradient.resize(fe_eval_xwall[0].n_q_points);

              val_enrgrad_to_grad(final_gradient);
              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              {
                final_gradient[q] += fe_eval_xwall[0].get_gradient(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
              }
              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              {
                gradient_type submitgradient = gradient_type();
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                  {
                    add_array_component_to_gradient(submitgradient,final_gradient[q],v);
                  }
                }
                gradients[q] = submitgradient;
              }
            }
            if(evaluate_val)
            {
              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              {
                value_type finalvalue = fe_eval_xwall[0].get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
                value_type submitvalue = value_type();
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                  {
                    add_array_component_to_value(submitvalue,finalvalue,v);
                  }
                }
                values[q]=submitvalue;
              }
            }
          }
#endif
      }

      void val_enrgrad_to_grad(AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& grad)
      {
        for(unsigned int j=0;j<dim;++j)
        {
          for(unsigned int i=0;i<dim;++i)
          {
            for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
            {
              grad[q][j][i] += fe_eval_xwall[0].get_value(q)[j]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
            }
          }
        }
      }
      void val_enrgrad_to_grad(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& grad)
      {
        for(unsigned int i=0;i<dim;++i)
        {
          for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
          {
            grad[q][i] += fe_eval_xwall[0].get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }


      void submit_value(const value_type val_in,
          const unsigned int q_point)
      {
        fe_eval[0].submit_value(val_in,q_point);
        values[q_point] = value_type();
#ifdef XWALL
          if(enriched)
          {
            value_type submitvalue = value_type();
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_value(submitvalue,val_in,v);
              }
            }
            values[q_point] = submitvalue;
          }
#endif
      }
      void submit_value(const Tensor<1,1,VectorizedArray<Number> > val_in,
          const unsigned int q_point)
      {
        fe_eval[0].submit_value(val_in[0],q_point);
        values[q_point] = value_type();
#ifdef XWALL
          if(enriched)
          {
            value_type submitvalue = value_type();
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_value(submitvalue,val_in[0],v);
              }
            }
            values[q_point] = submitvalue;
          }
#endif
      }

      void submit_gradient(const gradient_type grad_in,
          const unsigned int q_point)
      {
        fe_eval[0].submit_gradient(grad_in,q_point);
        gradients[q_point] = gradient_type();
#ifdef XWALL
          if(enriched)
          {
            gradient_type submitgradient = gradient_type();
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_gradient(submitgradient,grad_in,v);
              }
            }
            gradients[q_point] = submitgradient;
          }
#endif
      }

      void value_type_unit(VectorizedArray<Number>* test)
        {
          *test = make_vectorized_array(1.);
        }

      void value_type_unit(Tensor<1,n_components_,VectorizedArray<Number> >* test)
        {
          for(unsigned int i = 0; i< n_components_; i++)
            (*test)[i] = make_vectorized_array(1.);
        }

      void print_value_type_unit(VectorizedArray<Number> test)
        {
          std::cout << test[0] << std::endl;
        }

      void print_value_type_unit(Tensor<1,n_components_,VectorizedArray<Number> > test)
        {
          for(unsigned int i = 0; i< n_components_; i++)
            std::cout << test[i][0] << "  ";
          std::cout << std::endl;
        }

      value_type get_value(const unsigned int q_point)
      {
#ifdef XWALL
        {
          if(enriched)
          {
            value_type returnvalue = fe_eval[0].get_value(q_point);
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_value(returnvalue,values[q_point],v);
              }
            }
            return returnvalue;//fe_eval[0].get_value(q_point) + values[q_point];
          }
        }
#endif
          return fe_eval[0].get_value(q_point);
      }
      void add_array_component_to_value(VectorizedArray<Number>& val,const VectorizedArray<Number>& toadd, unsigned int v)
      {
        val[v] += toadd[v];
      }
      void add_array_component_to_value(Tensor<1,n_components_,VectorizedArray<Number> >& val,const Tensor<1,n_components_,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int d = 0; d<n_components_; d++)
          val[d][v] += toadd[d][v];
      }


      gradient_type get_gradient (const unsigned int q_point)
      {
#ifdef XWALL
        {
          if(enriched)
          {
            gradient_type returngradient = fe_eval[0].get_gradient(q_point);
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_gradient(returngradient,gradients[q_point],v);
              }
            }
            return returngradient;
          }
        }
#endif
        return fe_eval[0].get_gradient(q_point);
      }

      gradient_type get_symmetric_gradient (const unsigned int q_point)
      {
        return make_symmetric(get_gradient(q_point));
      }

      void add_array_component_to_gradient(Tensor<2,dim,VectorizedArray<Number> >& grad,const Tensor<2,dim,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int comp = 0; comp<dim; comp++)
          for (unsigned int d = 0; d<dim; d++)
            grad[comp][d][v] += toadd[comp][d][v];
      }
      void add_array_component_to_gradient(Tensor<1,dim,VectorizedArray<Number> >& grad,const Tensor<1,dim,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int d = 0; d<n_components_; d++)
          grad[d][v] += toadd[d][v];
      }

      Tensor<2,dim,VectorizedArray<Number> > make_symmetric(const Tensor<2,dim,VectorizedArray<Number> >& grad)
    {
        Tensor<2,dim,VectorizedArray<Number> > symgrad;
        for (unsigned int i = 0; i<dim; i++)
          for (unsigned int j = 0; j<dim; j++)
            symgrad[i][j] =  grad[i][j] + grad[j][i];
        return symgrad;
    }

    Tensor<1,dim,VectorizedArray<Number> > make_symmetric(const Tensor<1,dim,VectorizedArray<Number> >& grad)
      {
          Tensor<1,dim,VectorizedArray<Number> > symgrad;
          for (unsigned int j = 0; j<dim; j++)
            symgrad[j] = make_vectorized_array(0.);
          // symmetric gradient is not defined in that case
          Assert(false, ExcInternalError());
          return symgrad;
      }

      void integrate (const bool integrate_val,
                      const bool integrate_grad)
      {
#ifdef XWALL
        {
          if(enriched)
          {
            AlignedVector<value_type> tmp_values(fe_eval[0].n_q_points,value_type());
            if(integrate_val)
              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
                tmp_values[q]=values[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
            //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
            //the scalar product of the second part of the gradient is computed directly and added to the value
            if(integrate_grad)
            {
              //first, zero out all non-enriched vectorized array components
              grad_enr_to_val(tmp_values, gradients);

              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
                fe_eval_xwall[0].submit_gradient(gradients[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q),q);
            }

            for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              fe_eval_xwall[0].submit_value(tmp_values[q],q);
            //integrate
            fe_eval_xwall[0].integrate(true,integrate_grad);
          }
        }
#endif
        fe_eval[0].integrate(integrate_val, integrate_grad);
      }

      void grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
        {
          for(int j=0; j<dim;++j)//comp
          {
            for(int i=0; i<dim;++i)//dim
            {
              tmp_values[q][j] += gradient[q][i][j]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
            }
          }
        }
      }
      void grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
        {
          for(int i=0; i<dim;++i)//dim
          {
            tmp_values[q] += gradient[q][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }

      void sym_grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
        {
          for(int j=0; j<dim;++j)//comp
          {
            for(int i=0; i<dim;++i)//dim
            {
              tmp_values[q][j] += 0.5*gradient[q][i][j]*(EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i]);
              tmp_values[q][i] += 0.5*gradient[q][i][j]*(EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[j]);
            }
          }
        }
      }
      void sym_grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
      {
        Assert(false,ExcInternalError());
      }
      void distribute_local_to_global (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
      {
        fe_eval[0].distribute_local_to_global(dst);
//        for(unsigned int i = 0; i<fe_eval[0].dofs_per_cell ; i++)
//          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//            Assert(not isnan(fe_eval[0].begin_dof_values()[i][v]),ExcInternalError());
#ifdef XWALL
          if(enriched)
          {
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(not enriched_components.at(v))
                for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell*n_components_ ; i++)
                  fe_eval_xwall[0].begin_dof_values()[i][v] = 0.0;
            }
            fe_eval_xwall[0].distribute_local_to_global(dst_xwall);
//            for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell ; i++)
//              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//                Assert(not isnan(fe_eval_xwall[0].begin_dof_values()[i][v]),ExcInternalError());
          }
//          else
//          {
//            for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell ; i++)
//              fe_eval_xwall[0].begin_dof_values()[i] = make_vectorized_array(0.0);
//            fe_eval_xwall[0].distribute_local_to_global(dst_xwall);
//            for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell ; i++)
//              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//                Assert(not isnan(fe_eval_xwall[0].begin_dof_values()[i][v]),ExcInternalError());
//          }
#endif
      }

      void distribute_local_to_global (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i,std::vector<parallel::distributed::Vector<double> > &dst_xwall, unsigned int j)
      {
        fe_eval[0].distribute_local_to_global(dst,i);
//        for(unsigned int i = 0; i<fe_eval[0].dofs_per_cell ; i++)
//          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//            Assert(not isnan(fe_eval[0].begin_dof_values()[i][v]),ExcInternalError());
#ifdef XWALL
        if(enriched)
        {
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(not enriched_components.at(v))
              for(unsigned int k = 0; k<fe_eval_xwall[0].dofs_per_cell*n_components_ ; k++)
                fe_eval_xwall[0].begin_dof_values()[k][v] = 0.0;
          }
          fe_eval_xwall[0].distribute_local_to_global(dst_xwall,j);
//          for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell ; i++)
//            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//              Assert(not isnan(fe_eval_xwall[0].begin_dof_values()[i][v]),ExcInternalError());
        }
//        else
//        {
//          for(unsigned int k = 0; k<fe_eval_xwall[0].dofs_per_cell ; k++)
//            fe_eval_xwall[0].begin_dof_values()[k] = make_vectorized_array(0.0);
//          fe_eval_xwall[0].distribute_local_to_global(dst_xwall,j);
//          for(unsigned int i = 0; i<fe_eval_xwall[0].dofs_per_cell ; i++)
//            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//              Assert(not isnan(fe_eval_xwall[0].begin_dof_values()[i][v]),ExcInternalError());
//        }
#endif
      }

      void distribute_local_to_global (parallel::distributed::BlockVector<double> &dst, unsigned int i,parallel::distributed::BlockVector<double> &dst_xwall, unsigned int j)
      {
        fe_eval[0].distribute_local_to_global(dst,i);
#ifdef XWALL
        if(enriched)
        {
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(not enriched_components.at(v))
              for(unsigned int k = 0; k<fe_eval_xwall[0].dofs_per_cell*n_components_ ; k++)
                fe_eval_xwall[0].begin_dof_values()[k][v] = 0.0;
          }
          fe_eval_xwall[0].distribute_local_to_global(dst_xwall,j);
        }
#endif
      }

      void fill_JxW_values(AlignedVector<VectorizedArray<Number> > &JxW_values) const
      {
        fe_eval[0].fill_JxW_values(JxW_values);
#ifdef XWALL
          if(enriched)
            fe_eval_xwall[0].fill_JxW_values(JxW_values);
#endif
      }

      Point<dim,VectorizedArray<Number> > quadrature_point(unsigned int q)
      {
        return fe_eval[0].quadrature_point(q);
      }

      VectorizedArray<Number> get_divergence(unsigned int q)
    {
#ifdef XWALL
        if(enriched)
        {
          VectorizedArray<Number> div_enr= make_vectorized_array(0.0);
          for (unsigned int i=0;i<dim;i++)
            div_enr += gradients[q][i][i];
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(not enriched_components.at(v))
            {
              div_enr[v] = 0.0;
            }
          }
          return fe_eval[0].get_divergence(q) + div_enr;
        }
#endif
        return fe_eval[0].get_divergence(q);
    }

    Tensor<1,dim==2?1:dim,VectorizedArray<Number> >
    get_curl (const unsigned int q_point) const
     {
#ifdef XWALL
      if(enriched)
      {
        // copy from generic function into dim-specialization function
        const Tensor<2,dim,VectorizedArray<Number> > grad = gradients[q_point];
        Tensor<1,dim==2?1:dim,VectorizedArray<Number> > curl;
        switch (dim)
          {
          case 1:
            Assert (false,
                    ExcMessage("Computing the curl in 1d is not a useful operation"));
            break;
          case 2:
            curl[0] = grad[1][0] - grad[0][1];
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(not enriched_components.at(v))
              {
                curl[0][v]=0.0;
              }
            }
            break;
          case 3:
            curl[0] = grad[2][1] - grad[1][2];
            curl[1] = grad[0][2] - grad[2][0];
            curl[2] = grad[1][0] - grad[0][1];
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(not enriched_components.at(v))
              {
                curl[0][v]=0.0;
                curl[1][v]=0.0;
                curl[2][v]=0.0;
              }
            }
            break;
          default:
            Assert (false, ExcNotImplemented());
            break;
          }
        return fe_eval[0].get_curl(q_point) + curl;
      }
#endif
      return fe_eval[0].get_curl(q_point);
     }
    VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
    {
#ifdef XWALL
      if(enriched)
      {
        if(j<fe_eval[0].dofs_per_cell*n_components_)
          return fe_eval[0].begin_dof_values()[j];
        else
          return fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_];
      }
      else
        return fe_eval[0].begin_dof_values()[j];
#else

      return fe_eval[0].begin_dof_values()[j];
#endif
    }
    void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
    {
#ifdef XWALL
      if(enriched)
      {
        if(j<fe_eval[0].dofs_per_cell*n_components_)
          fe_eval[0].begin_dof_values()[j][v] = value;
        else
          fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_][v] = value;
      }
      else
        fe_eval[0].begin_dof_values()[j][v]=value;
      return;
#else
      fe_eval[0].begin_dof_values()[j][v]=value;
      return;
#endif
    }
    void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
    {
#ifdef XWALL
      if(enriched)
      {
        if(j<fe_eval[0].dofs_per_cell*n_components_)
          fe_eval[0].begin_dof_values()[j] = value;
        else
          fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_] = value;
      }
      else
        fe_eval[0].begin_dof_values()[j]=value;
      return;
#else
      fe_eval[0].begin_dof_values()[j]=value;
      return;
#endif
    }
    bool component_enriched(unsigned int v)
    {
      if(not enriched)
        return false;
      else
        return enriched_components.at(v);
    }

    void evaluate_eddy_viscosity(std::vector<parallel::distributed::Vector<double> > solution_n, unsigned int cell)
    {
      eddyvisc.resize(n_q_points);
      const VectorizedArray<Number> Cs = make_vectorized_array(CS);
      VectorizedArray<Number> hfac = make_vectorized_array(1.0/(double)fe_degree);
      fe_eval_tauw[0].reinit(cell);
      {
        VectorizedArray<Number> volume = make_vectorized_array(0.);
        {
          AlignedVector<VectorizedArray<Number> > JxW_values;
          JxW_values.resize(fe_eval_tauw[0].n_q_points);
          fe_eval_tauw[0].fill_JxW_values(JxW_values);
          for (unsigned int q=0; q<fe_eval_tauw[0].n_q_points; ++q)
            volume += JxW_values[q];
        }
        reinit(cell);
        read_dof_values(solution_n,0,solution_n,dim+1);
        evaluate (false,true,false);
        AlignedVector<VectorizedArray<Number> > wdist;
        wdist.resize(fe_eval_tauw[0].n_q_points);
        fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
        fe_eval_tauw[0].evaluate(true,false,false);
        for (unsigned int q=0; q<fe_eval_tauw[0].n_q_points; ++q)
          wdist[q] = fe_eval_tauw[0].get_value(q);
        fe_eval_tauw[0].reinit(cell);
        fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
        fe_eval_tauw[0].evaluate(true,false,false);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<Number> > s = get_symmetric_gradient(q);

          VectorizedArray<Number> snorm = make_vectorized_array(0.);
          for (unsigned int i = 0; i<dim ; i++)
            for (unsigned int j = 0; j<dim ; j++)
              snorm += make_vectorized_array(0.5)*(s[i][j])*(s[i][j]);
          //simple wall correction
          VectorizedArray<Number> fmu = (1.-std::exp(-wdist[q]/VISCOSITY*std::sqrt(fe_eval_tauw[0].get_value(q))/25.));
          VectorizedArray<Number> lm = Cs*std::pow(volume,1./3.)*hfac*fmu;
          eddyvisc[q]= make_vectorized_array(VISCOSITY) + std::pow(lm,2.)*std::sqrt(make_vectorized_array(2.)*snorm);
        }
      }

      return;
    }
    private:
      AlignedVector<FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> > fe_eval;
      AlignedVector<FEEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> > fe_eval_xwall;
      AlignedVector<FEEvaluation<dim,1,n_q_points_1d,1,double> > fe_eval_tauw;
      AlignedVector<value_type> values;
      AlignedVector<gradient_type> gradients;

    public:
      unsigned int n_q_points;
      unsigned int dofs_per_cell;
      unsigned int tensor_dofs_per_cell;
      bool enriched;
      std::vector<bool> enriched_components;
      AlignedVector<VectorizedArray<Number> > eddyvisc;

    };


  template <int dim, int fe_degree = 1, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
              int n_components_ = 1, typename Number = double >
    class FEFaceEvaluationXWall : public EvaluationXWall<dim,n_q_points_1d, Number>
    {
    public:
      typedef FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> BaseClass;
      typedef Number                            number_type;
      typedef typename BaseClass::value_type    value_type;
      typedef typename BaseClass::gradient_type gradient_type;
//    private:
//    static const unsigned int n_q_points_wall_normal = 20;
//    static const unsigned int n_q_points_wall_parallel = 8;
//    public:
//    static const unsigned int n_q_points = n_q_points_wall_normal * n_q_points_wall_parallel * n_q_points_wall_parallel;

    FEFaceEvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                        const parallel::distributed::Vector<double>& wdist,
                        const parallel::distributed::Vector<double>& tauw,
                        const bool                    is_left_face = true,
                        const unsigned int            fe_no = 0,
                        const unsigned int            quad_no = 0,
                        const bool                    no_gradients_on_faces = false):
                          EvaluationXWall<dim,n_q_points_1d, Number>::EvaluationXWall(matrix_free, wdist, tauw),
//TODO Benjamin: I always have to specify the quadrature rule here which fits to n_q_points_1d
                          fe_eval(matrix_free,is_left_face,0,quad_no,no_gradients_on_faces),
                          fe_eval_xwall(matrix_free,is_left_face,3,quad_no,no_gradients_on_faces),
                          fe_eval_tauw(matrix_free,is_left_face,2,quad_no,no_gradients_on_faces),
                          is_left_face(is_left_face),
                          values(fe_eval.n_q_points),
                          gradients(fe_eval.n_q_points),
                          dofs_per_cell(0),
                          tensor_dofs_per_cell(0),
                          n_q_points(fe_eval.n_q_points),
                          enriched(false)
      {
      };

      void reinit(const unsigned int f)
      {
#ifdef XWALL
        {
          enriched = false;
          values.resize(fe_eval.n_q_points,value_type());
          gradients.resize(fe_eval.n_q_points,gradient_type());
          if(is_left_face)
          {
  //        decide if we have an enriched element via the y component of the cell center
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements &&
              EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] != numbers::invalid_unsigned_int; ++v)
            {
              typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] / VectorizedArray<Number>::n_array_elements,
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] % VectorizedArray<Number>::n_array_elements);
                  if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                    enriched = true;
            }
          }
          else
          {
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements &&
              EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] != numbers::invalid_unsigned_int; ++v)
            {
              typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] / VectorizedArray<Number>::n_array_elements,
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] % VectorizedArray<Number>::n_array_elements);
                  if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                    enriched = true;
            }
          }
          enriched_components.resize(VectorizedArray<Number>::n_array_elements);
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            enriched_components.at(v) = false;
          if(enriched)
          {
            //store, exactly which component of the vectorized array is enriched
            if(is_left_face)
            {
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements&&
              EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] != numbers::invalid_unsigned_int; ++v)
              {
                typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                    EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] / VectorizedArray<Number>::n_array_elements,
                    EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] % VectorizedArray<Number>::n_array_elements);
                    if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                      enriched_components.at(v)=(true);
              }
            }
            else
            {
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements&&
              EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] != numbers::invalid_unsigned_int; ++v)
              {
                typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                    EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] / VectorizedArray<Number>::n_array_elements,
                    EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] % VectorizedArray<Number>::n_array_elements);
                    if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                      enriched_components.at(v)=(true);
              }
            }

            Assert(enriched_components.size()==VectorizedArray<Number>::n_array_elements,ExcInternalError());

            //initialize the enrichment function
            {
              fe_eval_tauw.reinit(f);
              //get wall distance and wss at quadrature points
              fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
              fe_eval_tauw.evaluate(true, true);

              AlignedVector<VectorizedArray<Number> > face_wdist;
              AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > face_gradwdist;
              face_wdist.resize(fe_eval_tauw.n_q_points);
              face_gradwdist.resize(fe_eval_tauw.n_q_points);
              for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
              {
                face_wdist[q] = fe_eval_tauw.get_value(q);
                face_gradwdist[q] = fe_eval_tauw.get_gradient(q);
              }

              fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
              fe_eval_tauw.evaluate(true, true);
              AlignedVector<VectorizedArray<Number> > face_tauw;
              AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > face_gradtauw;
              face_tauw.resize(fe_eval_tauw.n_q_points);
              face_gradtauw.resize(fe_eval_tauw.n_q_points);
              for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
              {
                face_tauw[q] = fe_eval_tauw.get_value(q);
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                    Assert( fe_eval_tauw.get_value(q)[v] > 1.0e-9 ,ExcInternalError());
                }

                face_gradtauw[q] = fe_eval_tauw.get_gradient(q);
              }
              EvaluationXWall<dim,n_q_points_1d, Number>::reinit(face_wdist, face_tauw, face_gradwdist, face_gradtauw, fe_eval_tauw.n_q_points,enriched_components);
            }
          }
          fe_eval_xwall.reinit(f);
        }
#endif
        fe_eval.reinit(f);
#ifdef XWALL
        if(enriched)
        {
          dofs_per_cell = fe_eval.dofs_per_cell + fe_eval_xwall.dofs_per_cell;
          tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell + fe_eval_xwall.tensor_dofs_per_cell;
        }
        else
        {
          dofs_per_cell = fe_eval.dofs_per_cell;
          tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell;
        }
#else
        dofs_per_cell = fe_eval.dofs_per_cell;
        tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell;
#endif
      }

      void read_dof_values (const parallel::distributed::Vector<double> &src, const parallel::distributed::Vector<double> &src_xwall)
      {
        fe_eval.read_dof_values(src);
#ifdef XWALL
        fe_eval_xwall.read_dof_values(src_xwall);
#endif
      }

      void read_dof_values (const std::vector<parallel::distributed::Vector<double> > &src, unsigned int i,const std::vector<parallel::distributed::Vector<double> > &src_xwall, unsigned int j)
      {
        fe_eval.read_dof_values(src,i);
#ifdef XWALL
        fe_eval_xwall.read_dof_values(src_xwall,j);
#endif
      }

      void read_dof_values (const parallel::distributed::BlockVector<double> &src, unsigned int i,const parallel::distributed::BlockVector<double> &src_xwall, unsigned int j)
      {
        fe_eval.read_dof_values(src,i);
#ifdef XWALL
        fe_eval_xwall.read_dof_values(src_xwall,j);
#endif
      }

      void evaluate(const bool evaluate_val,
                 const bool evaluate_grad,
                 const bool evaluate_hess = false)
      {
  fe_eval.evaluate(evaluate_val,evaluate_grad);
#ifdef XWALL
          if(enriched)
          {
            gradients.resize(fe_eval.n_q_points,gradient_type());
            values.resize(fe_eval.n_q_points,value_type());
            fe_eval_xwall.evaluate(true,evaluate_grad);
            //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
            EvaluationXWall<dim,n_q_points_1d,Number>::evaluate(evaluate_val,evaluate_grad,evaluate_hess);
            //evaluate gradient
            if(evaluate_grad)
            {
              //there are 2 parts due to chain rule
              //start with part 1
              AlignedVector<gradient_type> final_gradient;
              final_gradient.resize(fe_eval_xwall.n_q_points);

              val_enrgrad_to_grad(final_gradient);
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              {
                final_gradient[q] += fe_eval_xwall.get_gradient(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
              }
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              {
                gradient_type submitgradient = gradient_type();
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                  {
                    add_array_component_to_gradient(submitgradient,final_gradient[q],v);
                  }
                }
                gradients[q] = submitgradient;
              }
            }
            if(evaluate_val)
            {
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              {
                value_type finalvalue = fe_eval_xwall.get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
                value_type submitvalue = value_type();
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                  {
                    add_array_component_to_value(submitvalue,finalvalue,v);
                  }
                }
                values[q]=submitvalue;
              }
            }
          }
#endif
      }
      void val_enrgrad_to_grad(AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& grad)
      {
        for(unsigned int j=0;j<dim;++j)
        {
          for(unsigned int i=0;i<dim;++i)
          {
            for(unsigned int q=0;q<fe_eval.n_q_points;++q)
            {
              grad[q][j][i] += fe_eval_xwall.get_value(q)[j]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
            }
          }
        }
      }
      void val_enrgrad_to_grad(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& grad)
      {
        for(unsigned int i=0;i<dim;++i)
        {
          for(unsigned int q=0;q<fe_eval.n_q_points;++q)
          {
            grad[q][i] += fe_eval_xwall.get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }

      void submit_value(const value_type val_in,
          const unsigned int q_point)
      {
        fe_eval.submit_value(val_in,q_point);
        values[q_point] = value_type();
#ifdef XWALL
          if(enriched)
          {
            value_type submitvalue = value_type();
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_value(submitvalue,val_in,v);
              }
            }
            values[q_point] = submitvalue;
          }
#endif
      }

      void submit_gradient(const gradient_type grad_in,
          const unsigned int q_point)
      {
        fe_eval.submit_gradient(grad_in,q_point);
        gradients[q_point] = gradient_type();
#ifdef XWALL
          if(enriched)
          {
            gradient_type submitgradient = gradient_type();
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_gradient(submitgradient,grad_in,v);
              }
            }
            gradients[q_point] = submitgradient;
          }
#endif
      }

      value_type get_value(const unsigned int q_point)
      {
#ifdef XWALL
        {
          if(enriched)
          {
            value_type returnvalue = fe_eval.get_value(q_point);
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_value(returnvalue,values[q_point],v);
              }
            }
            return returnvalue;//fe_eval.get_value(q_point) + values[q_point];
          }
        }
#endif
          return fe_eval.get_value(q_point);
      }
      void add_array_component_to_value(VectorizedArray<Number>& val,const VectorizedArray<Number>& toadd, unsigned int v)
      {
        val[v] += toadd[v];
      }
      void add_array_component_to_value(Tensor<1,n_components_, VectorizedArray<Number> >& val,const Tensor<1,n_components_,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int d = 0; d<n_components_; d++)
          val[d][v] += toadd[d][v];
      }

      gradient_type get_gradient (const unsigned int q_point)
      {
#ifdef XWALL
        {
          if(enriched)
          {
            gradient_type returngradient = fe_eval.get_gradient(q_point);
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                add_array_component_to_gradient(returngradient,gradients[q_point],v);
              }
            }
            return returngradient;
          }
        }
#endif
        return fe_eval.get_gradient(q_point);
      }

      gradient_type get_symmetric_gradient (const unsigned int q_point)
      {
        return make_symmetric(get_gradient(q_point));
      }

      Tensor<2,dim,VectorizedArray<Number> > make_symmetric(const Tensor<2,dim,VectorizedArray<Number> >& grad)
    {
        Tensor<2,dim,VectorizedArray<Number> > symgrad;
        for (unsigned int i = 0; i<dim; i++)
          for (unsigned int j = 0; j<dim; j++)
            symgrad[i][j] = grad[i][j] + grad[j][i];
        return symgrad;
    }

      Tensor<1,dim,VectorizedArray<Number> > make_symmetric(const Tensor<1,dim,VectorizedArray<Number> >& grad)
    {
        Tensor<1,dim,VectorizedArray<Number> > symgrad;
        for (unsigned int j = 0; j<dim; j++)
          symgrad[j] = make_vectorized_array(0.);
        // symmetric gradient is not defined in that case
        Assert(false, ExcInternalError());
        return symgrad;
    }

      void add_array_component_to_gradient(Tensor<2,dim,VectorizedArray<Number> >& grad,const Tensor<2,dim,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int comp = 0; comp<dim; comp++)
          for (unsigned int d = 0; d<dim; d++)
            grad[comp][d][v] += toadd[comp][d][v];
      }
      void add_array_component_to_gradient(Tensor<1,dim,VectorizedArray<Number> >& grad,const Tensor<1,dim,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int d = 0; d<n_components_; d++)
          grad[d][v] += toadd[d][v];
      }

      VectorizedArray<Number> get_divergence(unsigned int q)
    {
#ifdef XWALL
        if(enriched)
        {
          VectorizedArray<Number> div_enr= make_vectorized_array(0.0);
          for (unsigned int i=0;i<dim;i++)
            div_enr += gradients[q][i][i];
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(not enriched_components.at(v))
            {
              div_enr[v] = 0.0;
            }
          }
          return fe_eval[0].get_divergence(q) + div_enr;
        }
#endif
        return fe_eval[0].get_divergence(q);
    }

      Tensor<1,dim,VectorizedArray<Number> > get_normal_vector(const unsigned int q_point) const
      {
        return fe_eval.get_normal_vector(q_point);
      }

      void integrate (const bool integrate_val,
                      const bool integrate_grad)
      {
#ifdef XWALL
        {
          if(enriched)
          {
            AlignedVector<value_type> tmp_values(fe_eval.n_q_points,value_type());
            if(integrate_val)
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
                tmp_values[q]=values[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
            //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
            //the scalar product of the second part of the gradient is computed directly and added to the value
            if(integrate_grad)
            {
              //first, zero out all non-enriched vectorized array components
              grad_enr_to_val(tmp_values,gradients);
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
                fe_eval_xwall.submit_gradient(gradients[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q),q);
            }

            for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              fe_eval_xwall.submit_value(tmp_values[q],q);
            //integrate
            fe_eval_xwall.integrate(true,integrate_grad);
          }
        }
#endif
        fe_eval.integrate(integrate_val, integrate_grad);
      }

      void grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {

          for(int j=0; j<dim;++j)//comp
          {
            for(int i=0; i<dim;++i)//dim
            {
              tmp_values[q][j] += gradient[q][i][j]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
            }
          }
        }
      }
      void grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          for(int i=0; i<dim;++i)//dim
          {
            tmp_values[q] += gradient[q][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }

      void sym_grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          for(int j=0; j<dim;++j)//comp
          {
            for(int i=0; i<dim;++i)//dim
            {
              tmp_values[q][j] += 0.5*gradient[q][i][j]*(EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i]);
              tmp_values[q][i] += 0.5*gradient[q][i][j]*(EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[j]);
            }
          }
        }
      }
      void sym_grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
      {
        Assert(false,ExcInternalError());
      }
      void distribute_local_to_global (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
      {
        fe_eval.distribute_local_to_global(dst);
//        for(unsigned int i = 0; i<fe_eval.dofs_per_cell ; i++)
//          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//            Assert(not isnan(fe_eval.begin_dof_values()[i][v]),ExcInternalError());
#ifdef XWALL
          if(enriched)
          {
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(not enriched_components.at(v))
                for(unsigned int i = 0; i<fe_eval_xwall.dofs_per_cell*n_components_ ; i++)
                  fe_eval_xwall.begin_dof_values()[i][v] = 0.0;
            }
            fe_eval_xwall.distribute_local_to_global(dst_xwall);

//            for(unsigned int i = 0; i<fe_eval_xwall.dofs_per_cell ; i++)
//              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//                Assert(not isnan(fe_eval_xwall.begin_dof_values()[i][v]),ExcInternalError());
          }
//          else
//          {
//            for(unsigned int i = 0; i<fe_eval_xwall.dofs_per_cell ; i++)
//              fe_eval_xwall.begin_dof_values()[i] = make_vectorized_array(0.0);
//            fe_eval_xwall.distribute_local_to_global(dst_xwall);
//            std::cout << "test4" << std::endl;
//            for(unsigned int i = 0; i<fe_eval_xwall.dofs_per_cell ; i++)
//              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//                Assert(not isnan(fe_eval_xwall.begin_dof_values()[i][v]),ExcInternalError());
//          }
#endif
      }

      void distribute_local_to_global (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i,std::vector<parallel::distributed::Vector<double> > &dst_xwall, unsigned int j)
      {
        fe_eval.distribute_local_to_global(dst,i);
#ifdef XWALL
        if(enriched)
        {
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(not enriched_components.at(v))
              for(unsigned int k = 0; k<fe_eval_xwall.dofs_per_cell*n_components_ ; k++)
                fe_eval_xwall.begin_dof_values()[k][v] = 0.0;
          }
          fe_eval_xwall.distribute_local_to_global(dst_xwall,j);
//          for(unsigned int i = 0; i<fe_eval_xwall.dofs_per_cell ; i++)
//            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//              Assert(not isnan(fe_eval_xwall.begin_dof_values()[i][v]),ExcInternalError());

        }
//        else
//        {
//          for(unsigned int k = 0; k<fe_eval_xwall.dofs_per_cell ; k++)
//            fe_eval_xwall.begin_dof_values()[k] = make_vectorized_array(0.0);
//          fe_eval_xwall.distribute_local_to_global(dst_xwall,j);
//          std::cout << "test12" << std::endl;
//          for(unsigned int i = 0; i<fe_eval_xwall.dofs_per_cell ; i++)
//            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//              Assert(not isnan(fe_eval_xwall.begin_dof_values()[i][v]),ExcInternalError());
//
//        }
#endif
      }


      void distribute_local_to_global (parallel::distributed::BlockVector<double> &dst, unsigned int i,parallel::distributed::BlockVector<double> &dst_xwall, unsigned int j)
      {
        fe_eval.distribute_local_to_global(dst,i);
#ifdef XWALL
        if(enriched)
        {
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(not enriched_components.at(v))
              for(unsigned int k = 0; k<fe_eval_xwall.dofs_per_cell*n_components_ ; k++)
                fe_eval_xwall.begin_dof_values()[k][v] = 0.0;
          }
          fe_eval_xwall.distribute_local_to_global(dst_xwall,j);

        }
#endif
      }

      Point<dim,VectorizedArray<Number> > quadrature_point(unsigned int q)
      {
        return fe_eval.quadrature_point(q);
      }

      VectorizedArray<Number> get_normal_volume_fraction()
      {
        return fe_eval.get_normal_volume_fraction();
      }

      VectorizedArray<Number> read_cell_data(const AlignedVector<VectorizedArray<Number> > &cell_data)
      {
        return fe_eval.read_cell_data(cell_data);
      }

      Tensor<1,n_components_,VectorizedArray<Number> > get_normal_gradient(const unsigned int q_point) const
      {
#ifdef XWALL
      {
        if(enriched)
        {
          Tensor<1,n_components_,VectorizedArray<Number> > grad_out;
          for (unsigned int comp=0; comp<n_components_; comp++)
          {
            grad_out[comp] = gradients[q_point][comp][0] *
                             fe_eval.get_normal_vector(q_point)[0];
            for (unsigned int d=1; d<dim; ++d)
              grad_out[comp] += gradients[q_point][comp][d] *
                               fe_eval.get_normal_vector(q_point)[d];
          }
          return fe_eval.get_normal_gradient(q_point) + grad_out;
        }
      }
#endif
        return fe_eval.get_normal_gradient(q_point);
      }
      VectorizedArray<Number> get_normal_gradient(const unsigned int q_point,bool test) const
      {
#ifdef XWALL
      {
        if(enriched)
        {
          VectorizedArray<Number> grad_out;
            grad_out = gradients[q_point][0] *
                             fe_eval.get_normal_vector(q_point)[0];
            for (unsigned int d=1; d<dim; ++d)
              grad_out += gradients[q_point][d] *
                               fe_eval.get_normal_vector(q_point)[d];
          return fe_eval.get_normal_gradient(q_point) + grad_out;
        }
      }
#endif
        return fe_eval.get_normal_gradient(q_point);
      }

      void submit_normal_gradient (const Tensor<1,n_components_,VectorizedArray<Number> > grad_in,
                                const unsigned int q)
      {
        fe_eval.submit_normal_gradient(grad_in,q);
        gradients[q]=gradient_type();
#ifdef XWALL

        if(enriched)
        {
          for (unsigned int comp=0; comp<n_components_; comp++)
            {
              for (unsigned int d=0; d<dim; ++d)
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                  {
                    gradients[q][comp][d][v] = grad_in[comp][v] *
                    fe_eval.get_normal_vector(q)[d][v];
                  }
                  else
                    gradients[q][comp][d][v] = 0.0;
                }
            }
        }
#endif
      }
      void submit_normal_gradient (const VectorizedArray<Number> grad_in,
                                const unsigned int q)
      {
        fe_eval.submit_normal_gradient(grad_in,q);
        gradients[q]=gradient_type();
#ifdef XWALL

        if(enriched)
        {
              for (unsigned int d=0; d<dim; ++d)
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                  {
                    gradients[q][d][v] = grad_in[v] *
                    fe_eval.get_normal_vector(q)[d][v];
                  }
                  else
                    gradients[q][d][v] = 0.0;
                }
        }
#endif
      }
      Tensor<1,dim==2?1:dim,VectorizedArray<Number> >
      get_curl (const unsigned int q_point) const
       {
  #ifdef XWALL
        if(enriched)
        {
          // copy from generic function into dim-specialization function
          const Tensor<2,dim,VectorizedArray<Number> > grad = gradients[q_point];
          Tensor<1,dim==2?1:dim,VectorizedArray<Number> > curl;
          switch (dim)
            {
            case 1:
              Assert (false,
                      ExcMessage("Computing the curl in 1d is not a useful operation"));
              break;
            case 2:
              curl[0] = grad[1][0] - grad[0][1];
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
              {
                if(not enriched_components.at(v))
                {
                  curl[0][v]=0.0;
                }
              }
              break;
            case 3:
              curl[0] = grad[2][1] - grad[1][2];
              curl[1] = grad[0][2] - grad[2][0];
              curl[2] = grad[1][0] - grad[0][1];
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
              {
                if(not enriched_components.at(v))
                {
                  curl[0][v]=0.0;
                  curl[1][v]=0.0;
                  curl[2][v]=0.0;
                }
              }
              break;
            default:
              Assert (false, ExcNotImplemented());
              break;
            }
          return fe_eval.get_curl(q_point) + curl;
        }
  #endif
        return fe_eval.get_curl(q_point);
       }

      VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
      {
  #ifdef XWALL
        if(enriched)
        {
          if(j<fe_eval.dofs_per_cell*n_components_)
            return fe_eval.begin_dof_values()[j];
          else
            return fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_];
        }
        else
          return fe_eval.begin_dof_values()[j];
  #else

        return fe_eval.begin_dof_values()[j];
  #endif
      }
      void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
      {
  #ifdef XWALL
        if(enriched)
        {
          if(j<fe_eval.dofs_per_cell*n_components_)
            fe_eval.begin_dof_values()[j][v] = value;
          else
            fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_][v] = value;
        }
        else
          fe_eval.begin_dof_values()[j][v]=value;
        return;
  #else
        fe_eval.begin_dof_values()[j][v]=value;
        return;
  #endif
      }
      void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
      {
  #ifdef XWALL
        if(enriched)
        {
          if(j<fe_eval.dofs_per_cell*n_components_)
            fe_eval.begin_dof_values()[j] = value;
          else
            fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_] = value;
        }
        else
          fe_eval.begin_dof_values()[j]=value;
        return;
  #else
        fe_eval.begin_dof_values()[j]=value;
        return;
  #endif
      }
      void evaluate_eddy_viscosity(std::vector<parallel::distributed::Vector<double> > solution_n, unsigned int face, const VectorizedArray<Number> volume)
      {
        eddyvisc.resize(n_q_points);
        const VectorizedArray<Number> Cs = make_vectorized_array(CS);
        VectorizedArray<Number> hfac = make_vectorized_array(1.0/(double)fe_degree);
        fe_eval_tauw.reinit(face);
        {
          reinit(face);
          read_dof_values(solution_n,0,solution_n,dim+1);
          evaluate (false,true,false);
          AlignedVector<VectorizedArray<Number> > wdist;
          wdist.resize(fe_eval_tauw.n_q_points);
          fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
          fe_eval_tauw.evaluate(true,false);
          for (unsigned int q=0; q<fe_eval_tauw.n_q_points; ++q)
            wdist[q] = fe_eval_tauw.get_value(q);
          fe_eval_tauw.reinit(face);
          fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
          fe_eval_tauw.evaluate(true,false);

          for (unsigned int q=0; q<n_q_points; ++q)
          {
            Tensor<2,dim,VectorizedArray<Number> > s = get_symmetric_gradient(q);

            VectorizedArray<Number> snorm = make_vectorized_array(0.);
            for (unsigned int i = 0; i<dim ; i++)
              for (unsigned int j = 0; j<dim ; j++)
                snorm += make_vectorized_array(0.5)*(s[i][j])*(s[i][j]);
            //simple wall correction
            VectorizedArray<Number> fmu = (1.-std::exp(-wdist[q]/VISCOSITY*std::sqrt(fe_eval_tauw.get_value(q))/25.));
            VectorizedArray<Number> lm = Cs*std::pow(volume,1./3.)*hfac*fmu;
            eddyvisc[q]= make_vectorized_array(VISCOSITY) + std::pow(lm,2.)*std::sqrt(make_vectorized_array(2.)*snorm);
          }
        }

        return;
      }
    private:
      FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> fe_eval;
      FEFaceEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> fe_eval_xwall;
      FEFaceEvaluation<dim,1,n_q_points_1d,1,Number> fe_eval_tauw;
      bool is_left_face;
      AlignedVector<value_type> values;
      AlignedVector<gradient_type> gradients;


    public:
      unsigned int dofs_per_cell;
      unsigned int tensor_dofs_per_cell;
      const unsigned int n_q_points;
      bool enriched;
      std::vector<bool> enriched_components;
      AlignedVector<VectorizedArray<Number> > eddyvisc;
    };



  template<int dim, int fe_degree, int fe_degree_xwall>
  class XWall
  {
  //time-integration-level routines for xwall
  public:
    XWall(const DoFHandler<dim> &dof_handler,
        std::vector<MatrixFree<dim,double> >* data,
        double visc,
        AlignedVector<VectorizedArray<double> > &element_volume);

    //initialize everything, e.g.
    //setup of wall distance
    //setup of communication of tauw to off-wall nodes
    //setup quadrature rules
    //possibly setup new matrixfree data object only including the xwall elements
    void initialize()
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "\nXWall Initialization:" << std::endl;

      //initialize wall distance and closest wall-node connectivity
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Initialize wall distance:...";
      InitWDist();
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << " done!" << std::endl;

      //initialize some vectors
      (*mydata).back().initialize_dof_vector(tauw, 2);
      tauw = 1.0;
    }

    //Update wall shear stress at the beginning of every time step
    void UpdateTauW(std::vector<parallel::distributed::Vector<double> > &solution_np);

    DoFHandler<dim>* ReturnDofHandlerWallDistance(){return &dof_handler_wall_distance;}
    const parallel::distributed::Vector<double>* ReturnWDist() const
        {return &wall_distance;}
    const parallel::distributed::Vector<double>* ReturnTauW() const
        {return &tauw;}
    const std::vector<MatrixFree<dim,double> >* ReturnData() const
        {return &mydata;}
  private:

    void InitWDist();

    //calculate wall shear stress based on current solution
    void CalculateWallShearStress(const std::vector<parallel::distributed::Vector<double> >   &src,
        parallel::distributed::Vector<double>      &dst);

    void L2Projection(){};

    //element-level routines
    void local_rhs_dummy (const MatrixFree<dim,double>                &data,
                          parallel::distributed::Vector<double>      &dst,
                          const std::vector<parallel::distributed::Vector<double> >    &src,
                          const std::pair<unsigned int,unsigned int>          &cell_range) const;

    void local_rhs_wss_boundary_face(const MatrixFree<dim,double>              &data,
                      parallel::distributed::Vector<double>      &dst,
                      const std::vector<parallel::distributed::Vector<double> >  &src,
                      const std::pair<unsigned int,unsigned int>          &face_range) const;

    void local_rhs_dummy_face (const MatrixFree<dim,double>              &data,
                  parallel::distributed::Vector<double>      &dst,
                  const std::vector<parallel::distributed::Vector<double> >  &src,
                  const std::pair<unsigned int,unsigned int>          &face_range) const;

    void local_rhs_normalization_boundary_face(const MatrixFree<dim,double>              &data,
                      parallel::distributed::Vector<double>      &dst,
                      const std::vector<parallel::distributed::Vector<double> >  &src,
                      const std::pair<unsigned int,unsigned int>          &face_range) const;

    //continuous vectors with linear interpolation
    FE_Q<dim> fe_wall_distance;
    DoFHandler<dim> dof_handler_wall_distance;
    parallel::distributed::Vector<double> wall_distance;
    parallel::distributed::Vector<double> tauw;
    std::vector<MatrixFree<dim,double> >* mydata;
    double viscosity;
//    parallel::distributed::Vector<double> &eddy_viscosity;
    AlignedVector<VectorizedArray<double> >& element_volume;

  public:

  };

  template<int dim, int fe_degree, int fe_degree_xwall>
  XWall<dim,fe_degree,fe_degree_xwall>::XWall(const DoFHandler<dim> &dof_handler,
      std::vector<MatrixFree<dim,double> >* data,
      double visc,
      AlignedVector<VectorizedArray<double> > &element_volume)
  :fe_wall_distance(1),
   dof_handler_wall_distance(dof_handler.get_tria()),
   mydata(data),
   viscosity(visc),
   element_volume(element_volume)
  {
    dof_handler_wall_distance.distribute_dofs(fe_wall_distance);
    dof_handler_wall_distance.distribute_mg_dofs(fe_wall_distance);
  }

  template<int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::InitWDist()
  {
    // compute wall distance
    (*mydata).back().initialize_dof_vector(wall_distance, 2);
    //TODO this first version doesn't work with periodic BC, I think...
//    //save all nodes on dirichlet boundaries in a map
//    std::map<unsigned int, Point<dim> > wallnodes_locations;
////
//    std::vector<types::global_dof_index> element_dof_indices((*mydata).back().get_dof_handler(2).get_fe().dofs_per_cell);
//    for (typename DoFHandler<dim>::active_cell_iterator cellw=dof_handler_wall_distance.begin_active();
//        cellw != dof_handler_wall_distance.end(); ++cellw)
//    {
//      if (cellw->is_locally_owned())
//      {
//        cellw->get_dof_indices(element_dof_indices);
//        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//        {
//          typename DoFHandler<dim>::face_iterator face=cellw->face(f);
//          //this is a face with dirichlet boundary
//          if(face->at_boundary())
//          {
//            unsigned int bid = face->boundary_id();
//            if(bid == 0)
//            {
//              for (unsigned int vw=0; vw<GeometryInfo<dim>::vertices_per_face; ++vw)
//              {
//                wallnodes_locations[element_dof_indices[vw]]=face->vertex(vw);
//              }
//            }
//          }
//        }
//      }
//    }
//
//    //look for the nearst wall node
//    //TODO Benjamin: for parallel computations, communicate wallnodes_locations to all procs
//    std::vector<types::global_dof_index> element_dof_indicesxw((*mydata).back().get_dof_handler(2).get_fe().dofs_per_cell);
//    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler_wall_distance.begin_active();
//        cell != dof_handler_wall_distance.end(); ++cell)
//      if (cell->is_locally_owned())
//      {
//        std::vector<double> tmpwdist(GeometryInfo<dim>::vertices_per_cell,1e9);
//
//        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
//        {
//          Point<dim> p = cell->vertex(v);
//
//          for(typename std::map<unsigned int, Point<dim> >::const_iterator pw = wallnodes_locations.begin(); pw != wallnodes_locations.end(); pw++)
//          {
//            double wdist=pw->second.distance(p);
//            if(wdist < tmpwdist.at(v))
//            {
//              tmpwdist.at(v) = wdist;
//            }
//          }
//        }
//        //TODO also store the connectivity of the enrichment nodes to these Dirichlet nodes
//        //to efficiently communicate the wall shear stress later on
//        cell->get_dof_indices(element_dof_indicesxw);
//        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
//        {
//          wall_distance(element_dof_indicesxw[v]) = tmpwdist.at(v);
//        }
//      }
//
//old version for serial case
//        std::vector<types::global_dof_index> element_dof_indices((*mydata).back().get_dof_handler(2).get_fe().dofs_per_cell);
//    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler_wall_distance.begin_active();
//        cell != dof_handler_wall_distance.end(); ++cell)
//      if (cell->is_locally_owned())
//      {
//                cell->get_dof_indices(element_dof_indices);
//        std::vector<double> tmpwdist(GeometryInfo<dim>::vertices_per_cell,1e9);
//        //TODO Benjamin
//        //this won't work in parallel
//        for (typename DoFHandler<dim>::active_cell_iterator cellw=dof_handler_wall_distance.begin_active();
//            cellw != dof_handler_wall_distance.end(); ++cellw)
//        {
//          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//          {
//            typename DoFHandler<dim>::face_iterator face=cellw->face(f);
//            //this is a face with dirichlet boundary
//            unsigned int bid = face->boundary_id();
//            if(bid == 0)
//            {
//              for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
//              {
//                Point<dim> p = cell->vertex(v);
//                for (unsigned int vw=0; vw<GeometryInfo<dim>::vertices_per_face; ++vw)
//                {
//                  Point<dim> pw =face->vertex(vw);
//                  double wdist=pw.distance(p);
//                  if(wdist < tmpwdist[v])
//                    tmpwdist[v]=wdist;
//                }
//              }
//            }
//          }
//        }
//        //TODO also store the connectivity of the enrichment nodes to these Dirichlet nodes
//        //to efficiently communicate the wall shear stress later on
//        cell->get_dof_indices(element_dof_indices);
//        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
//        {
//          wall_distance(element_dof_indices[v]) = tmpwdist[v];
//        }
//      }


    std::vector<types::global_dof_index> element_dof_indices((*mydata).back().get_dof_handler(2).get_fe().dofs_per_cell);
for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler_wall_distance.begin_active();
    cell != dof_handler_wall_distance.end(); ++cell)
  if (cell->is_locally_owned())
  {
    //TODO also store the connectivity of the enrichment nodes to these Dirichlet nodes
    //to efficiently communicate the wall shear stress later on
    cell->get_dof_indices(element_dof_indices);
    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      Point<dim> p = cell->vertex(v);
      // TODO: fill in wall distance function
      double wdist = 0.0;
      if(p[1]<0.0)
        wdist = 1.0+p[1];
      else
        wdist = 1.0-p[1];

      wall_distance(element_dof_indices[v]) = wdist;
    }
  }
  wall_distance.update_ghost_values();

//    wall_distance.print(std::cout);
//    Vector<double> local_distance_values(fe_wall_distance.dofs_per_cell);
//    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler_wall_distance.begin_active();
//        cell != dof_handler_wall_distance.end(); ++cell)
//      if (cell->is_locally_owned())
//      {
//        cell->get_dof_values(wall_distance, local_distance_values);
//        std::cout << "Element with center: " << cell->center() << ": ";
//        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
//        {
//          std::cout << local_distance_values[v] << " ";
//        }
//        std::cout << std::endl;
//      }
  }

  template<int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::UpdateTauW(std::vector<parallel::distributed::Vector<double> > &solution_np)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "\nCompute new tauw: ";
    CalculateWallShearStress(solution_np,tauw);
    //mean does not work currently because of all off-wall nodes in the vector
//    double tauwmean = tauw.mean_value();
//    std::cout << "mean = " << tauwmean << " ";

    double tauwmax = tauw.linfty_norm();
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "max = " << tauwmax << " ";

    double minloc = 1e9;
    for(unsigned int i = 0; i < tauw.local_size(); ++i)
    {
      if(tauw.local_element(i)>0.0)
      {
        if(minloc > tauw.local_element(i))
          minloc = tauw.local_element(i);
      }
    }
    const double minglob = Utilities::MPI::min(minloc, MPI_COMM_WORLD);

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "min = " << minglob << " ";
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "(set to 1.0 for now) ";
    tauw = 1.0;
    tauw.update_ghost_values();
#ifdef XWALL
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "L2-project... ";
    L2Projection();
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "done!" << std::endl;
#else
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << std::endl;
#endif
  }

  template<int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim, fe_degree,fe_degree_xwall>::
  CalculateWallShearStress (const std::vector<parallel::distributed::Vector<double> >   &src,
            parallel::distributed::Vector<double>      &dst)
  {
    parallel::distributed::Vector<double> normalization;
    (*mydata).back().initialize_dof_vector(normalization, 2);
    parallel::distributed::Vector<double> force;
    (*mydata).back().initialize_dof_vector(force, 2);

    // initialize
    force = 0.0;
    normalization = 0.0;


    (*mydata).back().loop (&XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy,
        &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy_face,
        &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_wss_boundary_face,
              this, force, src);

    (*mydata).back().loop (&XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy,
        &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy_face,
        &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_normalization_boundary_face,
              this, normalization, src);
    double mean = 0.0;
    unsigned int count = 0;
    for(unsigned int i = 0; i < force.local_size(); ++i)
    {
      if(normalization.local_element(i)>0.0)
      {
        dst.local_element(i) = force.local_element(i) / normalization.local_element(i);
        mean += dst.local_element(i);
        count++;
      }
    }
    mean = Utilities::MPI::sum(mean,MPI_COMM_WORLD);
    count = Utilities::MPI::sum(count,MPI_COMM_WORLD);
    mean /= (double)count;
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "mean = " << mean << " ";
  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  local_rhs_dummy (const MatrixFree<dim,double>                &data,
              parallel::distributed::Vector<double>      &dst,
              const std::vector<parallel::distributed::Vector<double> >  &src,
              const std::pair<unsigned int,unsigned int>           &cell_range) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  local_rhs_wss_boundary_face (const MatrixFree<dim,double>             &data,
                         parallel::distributed::Vector<double>    &dst,
                         const std::vector<parallel::distributed::Vector<double> >  &src,
                         const std::pair<unsigned int,unsigned int>          &face_range) const
  {
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,double> fe_eval_xwall(data,wall_distance,tauw,true,0,3);
    FEFaceEvaluation<dim,1,n_q_points_1d_xwall,1,double> fe_eval_tauw(data,true,2,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,double> fe_eval_xwall(data,wall_distance,tauw,true,0,0);
    FEFaceEvaluation<dim,1,fe_degree+1,1,double> fe_eval_tauw(data,true,2,0);
#endif
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        fe_eval_xwall.reinit (face);
        fe_eval_xwall.evaluate_eddy_viscosity(src,face,fe_eval_xwall.read_cell_data(element_volume));
        fe_eval_xwall.reinit (face);
        fe_eval_tauw.reinit (face);

        fe_eval_xwall.read_dof_values(src,0,src,dim+1);
        fe_eval_xwall.evaluate(false,true);
        if(fe_eval_xwall.n_q_points != fe_eval_tauw.n_q_points)
          std::cerr << "\nwrong number of quadrature points" << std::endl;

        for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
        {
          Tensor<1, dim, VectorizedArray<double> > average_gradient = fe_eval_xwall.get_normal_gradient(q);

          VectorizedArray<double> tauwsc = make_vectorized_array<double>(0.0);
          if(dim == 2)
            tauwsc = std::sqrt(average_gradient[0]*average_gradient[0] + average_gradient[1]*average_gradient[1]);
          else if(dim == 3)
            tauwsc = std::sqrt(average_gradient[0]*average_gradient[0] + average_gradient[1]*average_gradient[1] + average_gradient[2]*average_gradient[2]);

          tauwsc = tauwsc * fe_eval_xwall.eddyvisc[q];//(make_vectorized_array<double>(viscosity));
          fe_eval_tauw.submit_value(tauwsc,q);
        }
        fe_eval_tauw.integrate(true,false);
        fe_eval_tauw.distribute_local_to_global(dst);
      }
    }
  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  local_rhs_normalization_boundary_face (const MatrixFree<dim,double>             &data,
                         parallel::distributed::Vector<double>    &dst,
                         const std::vector<parallel::distributed::Vector<double> >  &src,
                         const std::pair<unsigned int,unsigned int>          &face_range) const
  {
    FEFaceEvaluation<dim,1,fe_degree+1,1,double> fe_eval_tauw(data,true,2,0);
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        fe_eval_tauw.reinit (face);

        for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
          fe_eval_tauw.submit_value(make_vectorized_array<double>(1.0),q);

        fe_eval_tauw.integrate(true,false);
        fe_eval_tauw.distribute_local_to_global(dst);
      }
    }
  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  local_rhs_dummy_face (const MatrixFree<dim,double>                 &data,
                parallel::distributed::Vector<double>      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const
  {

  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  class NavierStokesOperation
  {
  public:
  typedef double value_type;
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;

  NavierStokesOperation(const DoFHandler<dim> &dof_handler,const DoFHandler<dim> &dof_handler_p, const DoFHandler<dim> &dof_handler_xwall, const double time_step_size,
      const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs);

  void do_timestep (const double  &cur_time,const double  &delta_t, const unsigned int &time_step_number);

  void  rhs_convection (const std::vector<parallel::distributed::Vector<value_type> > &src,
                std::vector<parallel::distributed::Vector<value_type> >    &dst);

  void  compute_rhs (std::vector<parallel::distributed::Vector<value_type> >  &dst);

  void  apply_viscous (const parallel::distributed::BlockVector<value_type>     &src,
                   parallel::distributed::BlockVector<value_type>      &dst) const;

  void  rhs_viscous (const std::vector<parallel::distributed::Vector<value_type> >   &src,
                   std::vector<parallel::distributed::Vector<value_type> >  &dst);

  void  apply_pressure (const parallel::distributed::Vector<value_type>     &src,
                   parallel::distributed::Vector<value_type>      &dst) const;

  void  apply_pressure (const parallel::distributed::Vector<value_type>   &src,
                   parallel::distributed::Vector<value_type>    &dst,
                   const unsigned int                &level) const;

  void  apply_P (parallel::distributed::Vector<value_type> &dst) const;

  void  shift_pressure (parallel::distributed::Vector<value_type>  &pressure);

  void apply_inverse_mass_matrix(const parallel::distributed::BlockVector<value_type>  &src,
                parallel::distributed::BlockVector<value_type>    &dst) const;

  void precompute_inverse_mass_matrix();

  void  rhs_pressure (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                std::vector<parallel::distributed::Vector<value_type> >      &dst);

  void  apply_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                   std::vector<parallel::distributed::Vector<value_type> >      &dst);

  void compute_vorticity (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                      std::vector<parallel::distributed::Vector<value_type> >      &dst);

  void analyse_computing_times();
  void DistributeConstraintP(parallel::distributed::Vector<value_type> vec)
  {
    constraint_p_maxlevel.distribute(vec);
  }

  std::vector<parallel::distributed::Vector<value_type> > solution_nm2, solution_nm, solution_n, velocity_temp, velocity_temp2, solution_np;
  std::vector<parallel::distributed::Vector<value_type> > vorticity_nm2, vorticity_nm, vorticity_n;
  std::vector<parallel::distributed::Vector<value_type> > rhs_convection_nm2, rhs_convection_nm, rhs_convection_n;
  std::vector<parallel::distributed::Vector<value_type> > f;
  std::vector<parallel::distributed::Vector<value_type> > xwallstatevec;
//  parallel::distributed::Vector<value_type> eddy_viscosity;

  const MatrixFree<dim,value_type> & get_data() const
  {
    return data.back();
  }

  const MatrixFree<dim,value_type> & get_data(const unsigned int level) const
  {
    return data[level];
  }

  void calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal) const;

  void calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal, unsigned int level) const;

  void calculate_diagonal_viscous(std::vector<parallel::distributed::Vector<value_type> > &diagonal,
 unsigned int level) const;

  XWall<dim,fe_degree,fe_degree_xwall>* ReturnXWall(){return &xwall;}

  private:
  //MatrixFree<dim,value_type> data;
  std::vector<MatrixFree<dim,value_type> > data;

  double time, time_step;
  const double viscosity;
  double gamma0;
  double alpha[3], beta[3];
  std::vector<double> computing_times;
  std::vector<double> times_cg_velo;
  std::vector<unsigned int> iterations_cg_velo;
  std::vector<double> times_cg_pressure;
  std::vector<unsigned int> iterations_cg_pressure;

  //NavierStokesPressureMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> ns_pressure_matrix;
  MGLevelObject<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > mg_matrices_pressure;
//  MGTransferPrebuilt<parallel::distributed::Vector<double> > mg_transfer_pressure;
  MGTransferMF<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > mg_transfer_pressure;

  typedef PreconditionChebyshev<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                  parallel::distributed::Vector<double> > SMOOTHER_PRESSURE;
  // typename SMOOTHER_PRESSURE::AdditionalData smoother_data_pressure;
  MGLevelObject<typename SMOOTHER_PRESSURE::AdditionalData> smoother_data_pressure;

  MGSmootherPrecondition<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
    SMOOTHER_PRESSURE, parallel::distributed::Vector<double> > mg_smoother_pressure;
    MGCoarsePressure<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> mg_coarse_pressure;

    // PCG - solver for pressure with Chebyshev preconditioner
  PreconditionChebyshev<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                  parallel::distributed::Vector<value_type> > precondition_chebyshev_pressure;
  typename PreconditionChebyshev<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
                  parallel::distributed::Vector<value_type> >::AdditionalData smoother_data_chebyshev_pressure;

  PreconditionerJacobiPressure<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> jacobi_preconditioner_pressure;
  PreconditionerJacobiPressureCoarse<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> jacobi_preconditioner_pressure_coarse;
//  MGLevelObject<NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > mg_matrices_viscous;
//  MGTransferPrebuilt<parallel::distributed::BlockVector<double> > mg_transfer_viscous;

//  typedef PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
//                  parallel::distributed::BlockVector<double> > SMOOTHER_VISCOUS;
//  typename SMOOTHER_VISCOUS::AdditionalData smoother_data_viscous;
//  MGSmootherPrecondition<NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,
//    SMOOTHER_VISCOUS, parallel::distributed::BlockVector<double> > mg_smoother_viscous;
//    MGCoarseViscous<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> mg_coarse_viscous;

    std::vector< AlignedVector<VectorizedArray<value_type> > > array_penalty_parameter;

    AlignedVector<VectorizedArray<value_type> > element_volume;

    Point<dim> first_point;
    types::global_dof_index dof_index_first_point;

    XWall<dim,fe_degree,fe_degree_xwall> xwall;
    std::vector<Table<2,VectorizedArray<value_type> > > matrices;
    ConstraintMatrix constraint_p_maxlevel;

  void update_time_integrator();
  void check_time_integrator();

  // impulse equation
  void local_rhs_convection (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_convection_face (const MatrixFree<dim,value_type>              &data,
                  std::vector<parallel::distributed::Vector<double> >      &dst,
                  const std::vector<parallel::distributed::Vector<double> >  &src,
                  const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_convection_boundary_face(const MatrixFree<dim,value_type>              &data,
                      std::vector<parallel::distributed::Vector<double> >      &dst,
                      const std::vector<parallel::distributed::Vector<double> >  &src,
                      const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_compute_rhs (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_apply_viscous (const MatrixFree<dim,value_type>        &data,
                        parallel::distributed::BlockVector<double>      &dst,
                        const parallel::distributed::BlockVector<double>  &src,
                        const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_apply_viscous_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::BlockVector<double>      &dst,
                  const parallel::distributed::BlockVector<double>  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_apply_viscous_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::BlockVector<double>      &dst,
                      const parallel::distributed::BlockVector<double>  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_rhs_viscous (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_viscous_face (const MatrixFree<dim,value_type>              &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_viscous_boundary_face(const MatrixFree<dim,value_type>              &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  // poisson equation
  void local_apply_pressure (const MatrixFree<dim,value_type>          &data,
                            parallel::distributed::Vector<double>      &dst,
                            const parallel::distributed::Vector<double>  &src,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_apply_pressure_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::Vector<double>    &dst,
                  const parallel::distributed::Vector<double>  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_apply_pressure_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::Vector<double>    &dst,
                      const parallel::distributed::Vector<double>  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_laplace_diagonal(const MatrixFree<dim,value_type>        &data,
                            parallel::distributed::Vector<double>      &dst,
                            const parallel::distributed::Vector<double>  &src,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_laplace_diagonal_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::Vector<double>    &dst,
                  const parallel::distributed::Vector<double>  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_laplace_diagonal_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::Vector<double>    &dst,
                      const parallel::distributed::Vector<double>  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_diagonal_viscous(const MatrixFree<dim,value_type>        &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_diagonal_viscous_face (const MatrixFree<dim,value_type>      &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_diagonal_viscous_boundary_face(const MatrixFree<dim,value_type>      &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_rhs_pressure (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_pressure_face (const MatrixFree<dim,value_type>              &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_pressure_boundary_face(const MatrixFree<dim,value_type>              &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  // inverse mass matrix velocity
  void local_apply_mass_matrix(const MatrixFree<dim,value_type>                &data,
                      std::vector<parallel::distributed::Vector<value_type> >    &dst,
                      const std::vector<parallel::distributed::Vector<value_type> >  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range) const;
  // inverse mass matrix velocity
  void local_apply_mass_matrix(const MatrixFree<dim,value_type>                &data,
                      parallel::distributed::BlockVector<value_type>    &dst,
                      const parallel::distributed::BlockVector<value_type>  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range) const;
  // inverse mass matrix velocity
  void local_precompute_mass_matrix(const MatrixFree<dim,value_type>                &data,
                      std::vector<parallel::distributed::Vector<value_type> >    &,
                      const std::vector<parallel::distributed::Vector<value_type> >  &,
                      const std::pair<unsigned int,unsigned int>          &cell_range);

  void local_apply_mass_matrix(const MatrixFree<dim,value_type>          &data,
                      parallel::distributed::Vector<value_type>      &dst,
                      const std::vector<parallel::distributed::Vector<value_type> >   &src,
                      const std::pair<unsigned int,unsigned int>    &cell_range) const;

  // projection step
  void local_projection (const MatrixFree<dim,value_type>                &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >    &src,
                    const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_compute_vorticity (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >      &dst,
                            const std::vector<parallel::distributed::Vector<double> >    &src,
                            const std::pair<unsigned int,unsigned int>          &cell_range) const;

  //penalty parameter
  void calculate_penalty_parameter(double &factor) const;
  void calculate_penalty_parameter_pressure(double &factor) const;

  void compute_lift_and_drag();

  void compute_pressure_difference();
  void add_periodicity_constraints(const unsigned int level,
                                   const unsigned int target_level,
                                   const typename DoFHandler<dim>::face_iterator face1,
                                   const typename DoFHandler<dim>::face_iterator face2,
                                   ConstraintMatrix &constraints)
  {
    if (level == 0)
      {
        const unsigned int dofs_per_face = face1->get_fe(0).dofs_per_face;
        std::vector<types::global_dof_index> dofs_1(dofs_per_face);
        std::vector<types::global_dof_index> dofs_2(dofs_per_face);

        face1->get_mg_dof_indices(target_level, dofs_1, 0);
        face2->get_mg_dof_indices(target_level, dofs_2, 0);

        for (unsigned int i=0; i<dofs_per_face; ++i)
          {
            constraints.add_line(dofs_2[i]);
            constraints.add_entry(dofs_2[i], dofs_1[i], 1.);
          }
      }
    else
      {
        for (unsigned int c=0; c<face1->n_children(); ++c)
          add_periodicity_constraints(level-1, target_level, face1->child(c),face2->child(c),constraints);
      }
  };
  };


  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::NavierStokesOperation(const DoFHandler<dim> &dof_handler,
                                                                       const DoFHandler<dim> &dof_handler_p,
                                                                       const DoFHandler<dim> &dof_handler_xwall,
                                                                       const double time_step_size,
                                                                       const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs):
  time(0.0),
  time_step(time_step_size),
  viscosity(VISCOSITY),
  gamma0(1.0),
  alpha({1.0,0.0,0.0}),
  beta({1.0,0.0,0.0}),
  computing_times(4),
  times_cg_velo(3),
  iterations_cg_velo(3),
  times_cg_pressure(2),
  iterations_cg_pressure(2),
  mg_transfer_pressure(mg_matrices_pressure),
  jacobi_preconditioner_pressure(*this),
  jacobi_preconditioner_pressure_coarse(*this),
  element_volume(0),
  xwall(dof_handler,&data,viscosity,element_volume)
  {

  data.resize(dof_handler_p.get_tria().n_global_levels());
  //mg_matrices_pressure.resize(dof_handler_p.get_tria().n_levels()-2, dof_handler_p.get_tria().n_levels()-1);
  mg_matrices_pressure.resize(0, dof_handler_p.get_tria().n_global_levels()-1);
//  mg_matrices_viscous.resize(0, dof_handler.get_tria().n_levels()-1);
  gamma0 = 11.0/6.0;

  array_penalty_parameter.resize(dof_handler_p.get_tria().n_global_levels());

  for (unsigned int level=mg_matrices_pressure.min_level();level<=mg_matrices_pressure.max_level(); ++level)
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
    additional_data.periodic_face_pairs_level_0 = periodic_face_pairs;

    // collect the boundary indicators of periodic faces because their
    // weighting in the formula for the penalty parameter should be the one
    // for the interior not for the boundary
    std::set<types::boundary_id> periodic_boundary_ids;
    for (unsigned int i=0; i<periodic_face_pairs.size(); ++i)
      {
        periodic_boundary_ids.insert(periodic_face_pairs[i].cell[0]->face(periodic_face_pairs[i].face_idx[0])->boundary_id());
        periodic_boundary_ids.insert(periodic_face_pairs[i].cell[1]->face(periodic_face_pairs[i].face_idx[1])->boundary_id());
      }

    std::vector<const DoFHandler<dim> * >  dof_handler_vec;
    dof_handler_vec.push_back(&dof_handler);
    dof_handler_vec.push_back(&dof_handler_p);
    dof_handler_vec.push_back((xwall.ReturnDofHandlerWallDistance()));
    dof_handler_vec.push_back(&dof_handler_xwall);

    ConstraintMatrix constraint, constraint_p;
    for (typename std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >::const_iterator it=periodic_face_pairs.begin(); it!=periodic_face_pairs.end(); ++it)
      {
        typename Triangulation<dim>::face_iterator face1 = it->cell[1]->face(it->face_idx[1]);
        typename Triangulation<dim>::face_iterator face2 = it->cell[0]->face(it->face_idx[0]);
        typename DoFHandler<dim>::face_iterator dface1(&dof_handler_p.get_tria(), face1->level(), face1->index(), &dof_handler_p);
        typename DoFHandler<dim>::face_iterator dface2(&dof_handler_p.get_tria(), face2->level(), face2->index(), &dof_handler_p);
        add_periodicity_constraints(level, level,dface1,dface2,constraint_p);
        if(level == mg_matrices_pressure.max_level())
        {
          add_periodicity_constraints(level, level,dface1,dface2,constraint_p_maxlevel);
          constraint_p_maxlevel.close();
        }
      }


    constraint.close();
    constraint_p.close();

    std::vector<const ConstraintMatrix *> constraint_matrix_vec;
    constraint_matrix_vec.push_back(&constraint);
    constraint_matrix_vec.push_back(&constraint_p);
    constraint_matrix_vec.push_back(&constraint);
    constraint_matrix_vec.push_back(&constraint);

    std::vector<Quadrature<1> > quadratures;
    quadratures.push_back(QGauss<1>(fe_degree+1));
    quadratures.push_back(QGauss<1>(fe_degree_p+1));
    // quadrature formula 2: exact integration of convective term
    quadratures.push_back(QGauss<1>(fe_degree + (fe_degree+2)/2));
    quadratures.push_back(QGauss<1>(n_q_points_1d_xwall));

    const MappingQ<dim> mapping(fe_degree);

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
              const double factor = (cell->at_boundary(f) &&
                                     periodic_boundary_ids.find(cell->face(f)->boundary_id()) ==
                                     periodic_boundary_ids.end()) ? 1. : 0.5;
              for (unsigned int q=0; q<face_quadrature.size(); ++q)
                surface_area += fe_face_values.JxW(q) * factor;
            }
          array_penalty_parameter[level][i][v] = surface_area / volume;
          //pcout << "surface to volume ratio: " << array_penalty_parameter[level][i][v] << std::endl;
        }

    mg_matrices_pressure[level].initialize(*this, level);
//    mg_matrices_viscous[level].initialize(*this, level);
  }
  // PCG - solver for pressure with Chebyshev preconditioner
  smoother_data_chebyshev_pressure.smoothing_range = 30;
  smoother_data_chebyshev_pressure.degree = 0;
  smoother_data_chebyshev_pressure.eig_cg_n_iterations = 30;
  smoother_data_chebyshev_pressure.matrix_diagonal_inverse = mg_matrices_pressure[mg_matrices_pressure.max_level()].get_inverse_diagonal();
  precondition_chebyshev_pressure.initialize(mg_matrices_pressure[mg_matrices_pressure.max_level()], smoother_data_chebyshev_pressure);

  jacobi_preconditioner_pressure.initialize();
  jacobi_preconditioner_pressure_coarse.initialize();
  mg_transfer_pressure.build_matrices(dof_handler_p);
  mg_coarse_pressure.initialize(mg_matrices_pressure[mg_matrices_pressure.min_level()],jacobi_preconditioner_pressure_coarse);

//  mg_transfer_viscous.build_matrices(dof_handler);
//  mg_coarse_viscous.initialize(mg_matrices_viscous[mg_matrices_viscous.min_level()]);

  smoother_data_pressure.resize(0, dof_handler_p.get_tria().n_global_levels()-1);
  for(unsigned int level=0; level<dof_handler_p.get_tria().n_global_levels();++level)
  {
    smoother_data_pressure[level].smoothing_range = 30;
    smoother_data_pressure[level].degree = 7; //empirically: use degree = 3 - 6
    smoother_data_pressure[level].eig_cg_n_iterations = 20; //20

    smoother_data_pressure[level].matrix_diagonal_inverse = mg_matrices_pressure[level].get_inverse_diagonal();
  }

  mg_smoother_pressure.initialize(mg_matrices_pressure, smoother_data_pressure);


//  smoother_data_viscous.smoothing_range = 30;
//  smoother_data_viscous.degree = 5; //empirically: use degree = 3 - 6
//  smoother_data_viscous.eig_cg_n_iterations = 30;
//  mg_smoother_viscous.initialize(mg_matrices_viscous, smoother_data_viscous);
  gamma0 = 1.0;

  // initialize solution vectors
  solution_n.resize(dim+1+dim);
  data.back().initialize_dof_vector(solution_n[0], 0);
  for (unsigned int d=1;d<dim;++d)
  {
    solution_n[d] = solution_n[0];
  }
  data.back().initialize_dof_vector(solution_n[dim], 1);
  data.back().initialize_dof_vector(solution_n[dim+1], 3);
  for (unsigned int d=1;d<dim;++d)
  {
    solution_n[dim+d+1] = solution_n[dim+1];
  }
  solution_nm2 = solution_n;
  solution_nm = solution_n;
  solution_np = solution_n;

  velocity_temp.resize(2*dim);
  data.back().initialize_dof_vector(velocity_temp[0],0);
  data.back().initialize_dof_vector(velocity_temp[dim],3);
  for (unsigned int d=1;d<dim;++d)
  {
    velocity_temp[d] = velocity_temp[0];
    velocity_temp[d+dim] = velocity_temp[dim];
  }
  velocity_temp2 = velocity_temp;

  vorticity_n.resize(2*number_vorticity_components);
  data.back().initialize_dof_vector(vorticity_n[0]);
  for (unsigned int d=1;d<number_vorticity_components;++d)
  {
    vorticity_n[d] = vorticity_n[0];
  }
  data.back().initialize_dof_vector(vorticity_n[number_vorticity_components],3);
  for (unsigned int d=1;d<number_vorticity_components;++d)
  {
    vorticity_n[d+number_vorticity_components] = vorticity_n[number_vorticity_components];
  }
  vorticity_nm2 = vorticity_n;
  vorticity_nm = vorticity_n;
//  data.back().initialize_dof_vector(eddy_viscosity,0);

  rhs_convection_n.resize(2*dim);
  data.back().initialize_dof_vector(rhs_convection_n[0],0);
  data.back().initialize_dof_vector(rhs_convection_n[dim],3);
  for (unsigned int d=1;d<dim;++d)
  {
    rhs_convection_n[d] = rhs_convection_n[0];
    rhs_convection_n[d+dim] = rhs_convection_n[dim];
  }
  rhs_convection_nm2 = rhs_convection_n;
  rhs_convection_nm = rhs_convection_n;
  f = rhs_convection_n;

  dof_index_first_point = 0;
  for(unsigned int d=0;d<dim;++d)
    first_point[d] = 0.0;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    typename DoFHandler<dim>::active_cell_iterator first_cell;
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_p.begin_active(), endc = dof_handler_p.end();
    for(;cell!=endc;++cell)
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
  std::vector<types::global_dof_index>
  dof_indices(dof_handler_p.get_fe().dofs_per_cell);
  first_cell->get_dof_indices(dof_indices);
  dof_index_first_point = dof_indices[0];
  }
  dof_index_first_point = Utilities::MPI::sum(dof_index_first_point,MPI_COMM_WORLD);
  for(unsigned int d=0;d<dim;++d)
    first_point[d] = Utilities::MPI::sum(first_point[d],MPI_COMM_WORLD);
  xwall.initialize();
  xwallstatevec.push_back(*xwall.ReturnWDist());
  xwallstatevec.push_back(*xwall.ReturnTauW());
  //make sure that these vectors are available on the ghosted elements
  xwallstatevec[0].update_ghost_values();
  xwallstatevec[1].update_ghost_values();
  matrices.resize(data.back().n_macro_cells());
  const MappingQ<dim> mapping(fe_degree);
  QGauss<dim> quadrature(fe_degree+1);
  FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature, update_JxW_values);
  element_volume.resize(data.back().n_macro_cells()+data.back().n_macro_ghost_cells());
  for (unsigned int i=0; i<data.back().n_macro_cells()+data.back().n_macro_ghost_cells(); ++i)
    for (unsigned int v=0; v<data.back().n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data.back().get_cell_iterator(i,v);
        fe_values.reinit(cell);
        double volume = 0.;
        for (unsigned int q=0; q<quadrature.size(); ++q)
          volume += fe_values.JxW(q);
        element_volume[i][v] = volume;
        //pcout << "surface to volume ratio: " << array_penalty_parameter[level][i][v] << std::endl;
      }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct NavierStokesPressureMatrix : public Subscriptor
  {
    void initialize(const NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op, unsigned int lvl)
    {
      ns_operation = &ns_op;
      level = lvl;
      ns_operation->get_data(level).initialize_dof_vector(diagonal,1);
      ns_operation->calculate_laplace_diagonal(diagonal,level);
      inverse_diagonal = diagonal;
      for(unsigned int i=0; i<inverse_diagonal.local_size();++i)
        if( std::abs(inverse_diagonal.local_element(i)) > 1.0e-10 )
          inverse_diagonal.local_element(i) = 1.0/diagonal.local_element(i);
        else
        {
          inverse_diagonal.local_element(i) = 1.0;
          diagonal.local_element(i) = 1.0;
        }
      inverse_diagonal.update_ghost_values();
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
      if(pure_dirichlet_bc)
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

    const parallel::distributed::Vector<double> & get_inverse_diagonal() const
    {
    return inverse_diagonal;
    }

      void initialize_dof_vector(parallel::distributed::Vector<double> &src) const
      {
        if (!src.partitioners_are_compatible(*ns_operation->get_data(level).get_dof_info(1).vector_partitioner))
          ns_operation->get_data(level).initialize_dof_vector(src,1);
      }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
    unsigned int level;
    parallel::distributed::Vector<double> diagonal;
    parallel::distributed::Vector<double> inverse_diagonal;
  };

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct NavierStokesViscousMatrix : public Subscriptor
  {
    void initialize(NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
    {
      ns_operation = &ns_op;
    }
    void vmult (parallel::distributed::BlockVector<double> &dst,
        const parallel::distributed::BlockVector<double> &src) const
    {
      ns_operation->apply_viscous(src,dst);
    }

    NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
  };

//  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  struct NavierStokesViscousMatrix : public Subscriptor
//  {
//      void initialize(const NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op, unsigned int lvl)
//      {
//        ns_operation = &ns_op;
//        level = lvl;
//        ns_operation->get_data(level).initialize_dof_vector(diagonal.block(0),0);
//        ns_operation->get_data(level).initialize_dof_vector(diagonal.block(1),3);
//        std::vector<parallel::distributed::Vector<double> >  dst_tmp;
//        dst_tmp.resize(2);
//        ns_operation->calculate_diagonal_viscous(dst_tmp,level);
//        diagonal.block(0)=dst_tmp.at(0);
//        diagonal.block(1)=dst_tmp.at(1);
//      }
//
//      unsigned int m() const
//      {
//        return ns_operation->get_data(level).get_vector_partitioner(0)->size()+ns_operation->get_data(level).get_vector_partitioner(3)->size();
//      }
//
//      double el(const unsigned int row,const unsigned int /*col*/) const
//      {
//        return diagonal(row);
//      }
//
////      void vmult (parallel::distributed::Vector<double> &dst,
////          const parallel::distributed::Vector<double> &src) const
////      {
////        Assert(false,ExcInternalError());
//////        dst = 0;
//////        vmult_add(dst,src);
////      }
//      void vmult (parallel::distributed::BlockVector<double> &dst,
//          const parallel::distributed::BlockVector<double> &src) const
//      {
//        dst.block(0) = 0;
//        dst.block(1) = 0;
//        vmult_add(dst,src);
//      }
//
//      void Tvmult (parallel::distributed::BlockVector<double> &dst,
//          const parallel::distributed::BlockVector<double> &src) const
//      {
//        dst.block(0) = 0;
//        dst.block(1) = 0;
//        vmult_add(dst,src);
//      }
//
//      void Tvmult_add (parallel::distributed::BlockVector<double> &dst,
//          const parallel::distributed::BlockVector<double> &src) const
//      {
//        vmult_add(dst,src);
//      }
//
//      void vmult_add (parallel::distributed::BlockVector<double> &dst,
//          const parallel::distributed::BlockVector<double> &src) const
//      {
//        ns_operation->apply_viscous(src,dst,level);
//      }
//
//      const NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
//      unsigned int level;
//      parallel::distributed::BlockVector<double> diagonal;
//  };

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct PreconditionerInverseMassMatrix
  {
    PreconditionerInverseMassMatrix(NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
    :
      ns_op(ns_op)
    {}

    void vmult (parallel::distributed::BlockVector<double> &dst,
        const parallel::distributed::BlockVector<double> &src) const
    {
      ns_op.apply_inverse_mass_matrix(src,dst);
    }

    NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op;
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct PreconditionerJacobiPressure
  {
    PreconditionerJacobiPressure(const NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
    :
        ns_operation(ns_op)
    {

    }

    void initialize()
    {
      ns_operation.get_data().initialize_dof_vector(diagonal,1);
      ns_operation.calculate_laplace_diagonal(diagonal);
    }

    void vmult (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      for (unsigned int i=0;i<src.local_size();++i)
      {
        if( std::abs(diagonal.local_element(i)) > 1.0e-10 )
          dst.local_element(i) = src.local_element(i)/diagonal.local_element(i);
        else
          dst.local_element(i) = src.local_element(i);
      }
      dst.update_ghost_values();
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_operation;
    parallel::distributed::Vector<double> diagonal;
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct PreconditionerJacobiPressureCoarse
  {
    PreconditionerJacobiPressureCoarse(const NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
    :
        ns_operation(ns_op),
        level(0)
    {

    }

    void initialize()
    {
      ns_operation.get_data(level).initialize_dof_vector(diagonal,1);
      ns_operation.calculate_laplace_diagonal(diagonal,level);
    }

    void vmult (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      for (unsigned int i=0;i<src.local_size();++i)
      {
        if( std::abs(diagonal.local_element(i)) > 1.0e-10 )
          dst.local_element(i) = src.local_element(i)/diagonal.local_element(i);
        else
          dst.local_element(i) += src.local_element(i);
      }
      dst.update_ghost_values();
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_operation;
    parallel::distributed::Vector<double> diagonal;
    unsigned int level;
  };


  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  do_timestep (const double  &cur_time,const double  &delta_t, const unsigned int &time_step_number)
  {
  if(time_step_number == 1)
    check_time_integrator();

    const unsigned int output_solver_info_every_timesteps = 1e5;
    const unsigned int output_solver_info_details = 1e4;

    time = cur_time;
    time_step = delta_t;

    Timer timer;
    timer.restart();
    //important because I am using this on element level without giving it as argument to element loop
    for (unsigned int d=0; d<dim; ++d)
    {
      solution_n[d].update_ghost_values();
      solution_n[d+dim+1].update_ghost_values();
    }
  /***************** STEP 0: xwall update **********************************/
    {
      xwall.UpdateTauW(solution_n);
      xwallstatevec[1]=*xwall.ReturnTauW();
      xwallstatevec[0].update_ghost_values();
      xwallstatevec[1].update_ghost_values();
#ifdef XWALL
      precompute_inverse_mass_matrix();
#endif
    }
  /*************************************************************************/

  /***************** STEP 1: convective (nonlinear) term ********************/
    rhs_convection(solution_n,rhs_convection_n);
    compute_rhs(f);
    for (unsigned int d=0; d<dim; ++d)
    {
      velocity_temp[d].equ(beta[0],rhs_convection_n[d],beta[1],rhs_convection_nm[d],beta[2],rhs_convection_nm2[d]); // Stokes problem: velocity_temp[d] = f[d];
      velocity_temp[d].sadd(1.,f[d]);
      velocity_temp[d].sadd(time_step,alpha[0],solution_n[d],alpha[1],solution_nm[d],alpha[2],solution_nm2[d]);
      //xwall
      velocity_temp[d+dim].equ(beta[0],rhs_convection_n[d+dim],beta[1],rhs_convection_nm[d+dim],beta[2],rhs_convection_nm2[d+dim]); // Stokes problem: velocity_temp[d] = f[d];
      velocity_temp[d+dim].sadd(1.,f[d+dim]);
      velocity_temp[d+dim].sadd(time_step,alpha[0],solution_n[d+1+dim],alpha[1],solution_nm[d+1+dim],alpha[2],solution_nm2[d+1+dim]);
    }
    rhs_convection_nm2.swap(rhs_convection_nm);
    rhs_convection_nm.swap(rhs_convection_n);

    computing_times[0] += timer.wall_time();
  /*************************************************************************/

  /************ STEP 2: solve poisson equation for pressure ****************/
    timer.restart();

    rhs_pressure(velocity_temp,solution_np);

  // set maximum number of iterations, tolerance
    ReductionControl solver_control (1e3, 1.e-12, 1.e-8); //1.e-5
//  SolverControl solver_control (1e3, 1.e-6);
  SolverCG<parallel::distributed::Vector<double> > solver (solver_control);

//  Timer cg_timer;
//  cg_timer.restart();

  // start CG-iterations with pressure solution at time t_n
  parallel::distributed::Vector<value_type> solution(solution_n[dim]);

  // CG-Solver without preconditioning
//    solver.solve (mg_matrices_pressure[mg_matrices_pressure.max_level()], solution, solution_np[dim], PreconditionIdentity());

//    times_cg_pressure[0] += cg_timer.wall_time();
//    iterations_cg_pressure[0] += solver_control.last_step();
//    cg_timer.restart();
//    solution = solution_n[dim];

    // PCG-Solver with GMG + Chebyshev smoother as a preconditioner

  mg::Matrix<parallel::distributed::Vector<double> > mgmatrix_pressure(mg_matrices_pressure);
  Multigrid<parallel::distributed::Vector<double> > mg_pressure(data.back().get_dof_handler(1),
                             mgmatrix_pressure,
                               mg_coarse_pressure,
                               mg_transfer_pressure,
                               mg_smoother_pressure,
                               mg_smoother_pressure);
  PreconditionMG<dim, parallel::distributed::Vector<double>, MGTransferMF<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > >
  preconditioner_pressure(data.back().get_dof_handler(1), mg_pressure, mg_transfer_pressure);
  try
  {
    solver.solve (mg_matrices_pressure[mg_matrices_pressure.max_level()], solution, solution_np[dim], PreconditionIdentity());//preconditioner_pressure);
  }
  catch (SolverControl::NoConvergence test)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::cout<<"Pressure Poisson solver reached maximum iterations. The residual at the last iteration was: " << test.last_residual << std::endl;
    }
    if(test.last_residual>1.0)
      Assert(false,ExcInternalError());
  }

//    times_cg_pressure[1] += cg_timer.wall_time();
//    iterations_cg_pressure[1] += solver_control.last_step();

//    if(time_step_number%10 == 0)
//    std::cout << std::endl << "Solve pressure Poisson equation: Number of timesteps: " << time_step_number << std::endl
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

//  if(pure_dirichlet_bc)
//  {
//    shift_pressure(solution);
//  }
  solution_np[dim] = solution;

    if(time_step_number%output_solver_info_every_timesteps == 0)
  {
      std::cout << std::endl << "Number of timesteps: " << time_step_number << std::endl;
      std::cout << "Solve Poisson equation for p: PCG iterations: " << std::setw(3) << solver_control.last_step() << "  Wall time: " << timer.wall_time() << std::endl;
  }

  computing_times[1] += timer.wall_time();
  /*************************************************************************/

  /********************** STEP 3: projection *******************************/
    timer.restart();
    DistributeConstraintP(solution_np[dim]);

  apply_projection(solution_np,velocity_temp2);
  for (unsigned int d=0; d<2*dim; ++d)
  {
    velocity_temp2[d].sadd(time_step,1.0,velocity_temp[d]);
  }
  computing_times[2] += timer.wall_time();
  /*************************************************************************/

  /************************ STEP 4: viscous term ***************************/
    timer.restart();

    rhs_viscous(velocity_temp2,solution_np);

  // set maximum number of iterations, tolerance
  ReductionControl solver_control_velocity (1e3, 1.e-12, 1.e-8);//1.e-5
//  SolverCG<parallel::distributed::BlockVector<double> > solver_velocity (solver_control_velocity);
  SolverGMRES<parallel::distributed::BlockVector<double> > solver_velocity (solver_control_velocity);
  NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> ns_viscous_matrix;
  ns_viscous_matrix.initialize(*this);

//  for (unsigned int d=0;d<dim;++d)
  {
    double wall_time_temp = timer.wall_time();

//    Timer cg_timer_viscous;
//    cg_timer_viscous.restart();

    // start CG-iterations with solution_n
    parallel::distributed::BlockVector<value_type> tmp_solution(6);
    tmp_solution.block(0) = solution_n[0];
    tmp_solution.block(1) = solution_n[1];
    tmp_solution.block(2) = solution_n[2];
    tmp_solution.block(3) = solution_n[0+dim+1];
    tmp_solution.block(4) = solution_n[1+dim+1];
    tmp_solution.block(5) = solution_n[2+dim+1];
    tmp_solution.collect_sizes();
    parallel::distributed::BlockVector<value_type> tmp_solution_np(6);
    tmp_solution_np.block(0) = solution_np[0];
    tmp_solution_np.block(1) = solution_np[1];
    tmp_solution_np.block(2) = solution_np[2];
    tmp_solution_np.block(3) = solution_np[0+dim+1];
    tmp_solution_np.block(4) = solution_np[1+dim+1];
    tmp_solution_np.block(5) = solution_np[2+dim+1];
    tmp_solution_np.collect_sizes();

    // CG-Solver without preconditioning
    //solver_velocity.solve (ns_viscous_matrix, solution, solution_np[d], PreconditionIdentity());
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], PreconditionIdentity());

//    times_cg_velo[0] += cg_timer_viscous.wall_time();
//    iterations_cg_velo[0] += solver_control_velocity.last_step();
//    cg_timer_viscous.restart();
//    solution = solution_n[d];

    // PCG-Solver with inverse mass matrix as a preconditioner
    // solver_velocity.solve (ns_viscous_matrix, solution, solution_np[d], preconditioner);


    PreconditionerInverseMassMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> preconditioner(*this);

    try
    {
      solver_velocity.solve (ns_viscous_matrix, tmp_solution, tmp_solution_np, preconditioner);//PreconditionIdentity());
    }
    catch (SolverControl::NoConvergence)
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        std::cout<<"Viscous solver failed to solve to given convergence." << std::endl;
    }


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
//    PreconditionMG<dim, parallel::distributed::Vector<double>, MGTransferPrebuilt<parallel::distributed::Vector<double> > >
//    preconditioner_viscous(data.back().get_dof_handler(0), mg_viscous, mg_transfer_viscous);
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], preconditioner_viscous);

    // PCG-Solver with Chebyshev preconditioner
//    PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,parallel::distributed::Vector<value_type> > precondition_chebyshev;
//    typename PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>,parallel::distributed::Vector<value_type> >::AdditionalData smoother_data;
//    smoother_data.smoothing_range = 30;
//    smoother_data.degree = 5;
//    smoother_data.eig_cg_n_iterations = 30;
//    precondition_chebyshev.initialize(mg_matrices_viscous[mg_matrices_viscous.max_level()], smoother_data);
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], precondition_chebyshev);

//    times_cg_velo[2] += cg_timer_viscous.wall_time();
//    iterations_cg_velo[2] += solver_control_velocity.last_step();
    solution_np[0] = tmp_solution.block(0);
    solution_np[1] = tmp_solution.block(1);
    solution_np[2] = tmp_solution.block(2);
    solution_np[0+dim+1] = tmp_solution.block(3);
    solution_np[1+dim+1] = tmp_solution.block(4);
    solution_np[2+dim+1] = tmp_solution.block(5);

//    if(time_step_number%output_solver_info_every_timesteps == 0)
//    {
//    std::cout << "Solve viscous step for u" << d+1 <<":    PCG iterations: " << std::setw(3) << solver_control_velocity.last_step() << "  Wall time: " << timer.wall_time()-wall_time_temp << std::endl;
//    }
  }
//  if(time_step_number%10 == 0)
//    std::cout << "Solve viscous step for u: Number of timesteps: " << time_step_number << std::endl
//          << "CG (no preconditioning):  wall time: " << times_cg_velo[0]/time_step_number << " Iterations: " << (double)iterations_cg_velo[0]/time_step_number/dim << std::endl
//          << "PCG (inv mass precond.):  wall time: " << times_cg_velo[1]/time_step_number << " Iterations: " << (double)iterations_cg_velo[1]/time_step_number/dim << std::endl
//          << "PCG (GMG with Chebyshev): wall time: " << times_cg_velo[2]/time_step_number << " Iterations: " << (double)iterations_cg_velo[2]/time_step_number/dim << std::endl
//          << std::endl;

  computing_times[3] += timer.wall_time();
  /*************************************************************************/

  // solution at t_n -> solution at t_n-1    and    solution at t_n+1 -> solution at t_n
  solution_nm2.swap(solution_nm);
  solution_nm.swap(solution_n);
  solution_n.swap(solution_np);

  vorticity_nm2.swap(vorticity_nm);
  vorticity_nm.swap(vorticity_n);

  compute_vorticity(solution_n,vorticity_n);

//  compute_lift_and_drag();
//  compute_pressure_difference();
//  compute_vorticity(solution_n,vorticity_n);
  if(time_step_number == 1)
    update_time_integrator();
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  update_time_integrator ()
  {
//    BDF2
//    gamma0 = 3.0/2.0;
//    alpha[0] = 2.0;
//    alpha[1] = -0.5;
//    beta[0] = 2.0;
//    beta[1] = -1.0;
//    BDF3
    gamma0 = 11./6.;
    alpha[0] = 3.;
    alpha[1] = -1.5;
    alpha[2] = 1./3.;
    beta[0] = 3.0;
    beta[1] = -3.0;
    beta[2] = 1.0;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  check_time_integrator()
  {
    if (std::abs(gamma0-1.0)>1.e-12 || std::abs(alpha[0]-1.0)>1.e-12 || std::abs(alpha[1]-0.0)>1.e-12 || std::abs(beta[0]-1.0)>1.e-12 || std::abs(beta[1]-0.0)>1.e-12)
    {
      std::cout<< "Time integrator constants invalid!" << std::endl;
      std::abort();
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  analyse_computing_times()
  {
  double time=0.0;
  for(unsigned int i=0;i<4;++i)
    time+=computing_times[i];
  std::cout<<std::endl<<"Computing times:"
       <<std::endl<<"Step 1: Convection:\t"<<computing_times[0]/time
       <<std::endl<<"Step 2: Pressure:\t"<<computing_times[1]/time
       <<std::endl<<"Step 3: Projection:\t"<<computing_times[2]/time
       <<std::endl<<"Step 4: Viscous:\t"<<computing_times[3]/time
       <<std::endl<<"Time (Step 1-4):\t"<<time<<std::endl;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  calculate_penalty_parameter(double &factor) const
  {
    // triangular/tetrahedral elements: penalty parameter = stab_factor*(p+1)(p+d)/dim * surface/volume
//  factor = stab_factor * (fe_degree +1.0) * (fe_degree + dim) / dim;

    // quadrilateral/hexahedral elements: penalty parameter = stab_factor*(p+1)(p+1) * surface/volume
    factor = stab_factor * (fe_degree +1.0) * (fe_degree + 1.0);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  calculate_penalty_parameter_pressure(double &factor) const
  {
    // triangular/tetrahedral elements: penalty parameter = stab_factor*(p+1)(p+d)/dim * surface/volume
//  factor = stab_factor * (fe_degree_p +1.0) * (fe_degree_p + dim) / dim;

    // quadrilateral/hexahedral elements: penalty parameter = stab_factor*(p+1)(p+1) * surface/volume
    factor = stab_factor * (fe_degree_p +1.0) * (fe_degree_p + 1.0);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_lift_and_drag()
  {
  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval_velocity(data.back(),true,0,0);
  FEFaceEvaluation<dim,fe_degree_p,fe_degree+1,1,value_type> fe_eval_pressure(data.back(),true,1,0);

  Tensor<1,dim,value_type> Force;
  for(unsigned int d=0;d<dim;++d)
    Force[d] = 0.0;

  for(unsigned int face=data.back().n_macro_inner_faces(); face<(data.back().n_macro_inner_faces()+data.back().n_macro_boundary_faces()); face++)
  {
    fe_eval_velocity.reinit (face);
    fe_eval_velocity.read_dof_values(solution_n,0);
    fe_eval_velocity.evaluate(false,true);

    fe_eval_pressure.reinit (face);
    fe_eval_pressure.read_dof_values(solution_n,dim);
    fe_eval_pressure.evaluate(true,false);

    if (data.back().get_boundary_indicator(face) == 2)
    {
      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<value_type> pressure = fe_eval_pressure.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
        Tensor<2,dim,VectorizedArray<value_type> > velocity_gradient = fe_eval_velocity.get_gradient(q);
        fe_eval_velocity.submit_value(pressure*normal -  make_vectorized_array<value_type>(viscosity)*
            (velocity_gradient+transpose(velocity_gradient))*normal,q);
      }
      Tensor<1,dim,VectorizedArray<value_type> > Force_local = fe_eval_velocity.integrate_value();

      // sum over all entries of VectorizedArray
      for (unsigned int d=0; d<dim;++d)
        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          Force[d] += Force_local[d][n];
    }
  }
  Force = Utilities::MPI::sum(Force,MPI_COMM_WORLD);

//  // compute lift and drag coefficients (c = (F/rho)/(1/2 U² D)
//  const double U = Um * (dim==2 ? 2./3. : 4./9.);
//  const double H = 0.41;
//  if(dim == 2)
//    Force *= 2.0/pow(U,2.0)/D;
//  else if(dim == 3)
//    Force *= 2.0/pow(U,2.0)/D/H;
//
//  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
//  {
//    std::string filename_drag, filename_lift;
//    filename_drag = "output/drag_refine" + Utilities::int_to_string(data.back().get_dof_handler(1).get_tria().n_levels()-1) + "_fedegree" + Utilities::int_to_string(fe_degree) + ".txt"; //filename_drag = "drag.txt";
//    filename_lift = "output/lift_refine" + Utilities::int_to_string(data.back().get_dof_handler(1).get_tria().n_levels()-1) + "_fedegree" + Utilities::int_to_string(fe_degree) + ".txt"; //filename_lift = "lift.txt";
//
//    std::ofstream f_drag,f_lift;
//    if(clear_files)
//    {
//      f_drag.open(filename_drag.c_str(),std::ios::trunc);
//      f_lift.open(filename_lift.c_str(),std::ios::trunc);
//    }
//    else
//    {
//      f_drag.open(filename_drag.c_str(),std::ios::app);
//      f_lift.open(filename_lift.c_str(),std::ios::app);
//    }
//    f_drag<<std::scientific<<std::setprecision(6)<<time+time_step<<"\t"<<Force[0]<<std::endl;
//    f_drag.close();
//    f_lift<<std::scientific<<std::setprecision(6)<<time+time_step<<"\t"<<Force[1]<<std::endl;
//    f_lift.close();
//  }
  }


//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  compute_eddy_viscosity(const std::vector<parallel::distributed::Vector<value_type> >     &src)
//  {
//
//    eddy_viscosity = 0;
//    data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_eddy_viscosity,this, eddy_viscosity, src);
//
//    const double mean = eddy_viscosity.mean_value();
//    eddy_viscosity.update_ghost_values();
//    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
//      std::cout << "new viscosity:   " << mean << "/" << viscosity << std::endl;
//  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_compute_eddy_viscosity(const MatrixFree<dim,value_type>                  &data,
//                parallel::distributed::Vector<value_type>      &dst,
//                const std::vector<parallel::distributed::Vector<value_type> >  &src,
//                const std::pair<unsigned int,unsigned int>            &cell_range) const
//  {
//    const VectorizedArray<value_type> Cs = make_vectorized_array(CS);
//    VectorizedArray<value_type> hfac = make_vectorized_array(1.0/(double)fe_degree);
//
//  //Warning: eddy viscosity is only interpolated using the polynomial space
//
//  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> velocity_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
//  FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi(data,0,0);
//  FEEvaluation<dim,1,fe_degree+1,1,double> fe_wdist(data,2,0);
//  FEEvaluation<dim,1,fe_degree+1,1,double> fe_tauw(data,2,0);
//  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);
//  const unsigned int dofs_per_cell = phi.dofs_per_cell;
//  AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
//
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//  {
//    phi.reinit(cell);
//    {
//      VectorizedArray<value_type> volume = make_vectorized_array(0.);
//      {
//        AlignedVector<VectorizedArray<value_type> > JxW_values;
//        JxW_values.resize(phi.n_q_points);
//        phi.fill_JxW_values(JxW_values);
//        for (unsigned int q=0; q<phi.n_q_points; ++q)
//          volume += JxW_values[q];
//      }
//      velocity_xwall.reinit(cell);
//      velocity_xwall.read_dof_values(src,0,src,dim+1);
//      velocity_xwall.evaluate (false,true,false);
//      fe_wdist.reinit(cell);
//      fe_wdist.read_dof_values(xwallstatevec[0]);
//      fe_wdist.evaluate(true,false,false);
//      fe_tauw.reinit(cell);
//      fe_tauw.read_dof_values(xwallstatevec[1]);
//      fe_tauw.evaluate(true,false,false);
//      for (unsigned int q=0; q<phi.n_q_points; ++q)
//      {
//        Tensor<2,dim,VectorizedArray<value_type> > s = velocity_xwall.get_gradient(q);
//
//        VectorizedArray<value_type> snorm = make_vectorized_array(0.);
//        for (unsigned int i = 0; i<dim ; i++)
//          for (unsigned int j = 0; j<dim ; j++)
//            snorm += make_vectorized_array(0.5)*(s[i][j]+s[j][i])*(s[i][j]+s[j][i]);
//        //simple wall correction
//        VectorizedArray<value_type> fmu = (1.-std::exp(-fe_wdist.get_value(q)/viscosity*std::sqrt(fe_tauw.get_value(q))/25.));
//        VectorizedArray<value_type> lm = Cs*std::pow(volume,1./3.)*hfac*fmu;
//        phi.submit_value (make_vectorized_array(viscosity) + std::pow(lm,2.)*std::sqrt(make_vectorized_array(2.)*snorm), q);
//      }
//      phi.integrate (true,false);
//
//      inverse.fill_inverse_JxW_values(coefficients);
//      inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());
//
//      phi.set_dof_values(dst);
//    }
//  }
//
//  }
  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_pressure_difference()
  {
  double pressure_1 = 0.0, pressure_2 = 0.0;
  unsigned int counter_1 = 0, counter_2 = 0;

  Point<dim> point_1, point_2;
  if(dim == 2)
  {
    Point<dim> point_1_2D(0.45,0.2), point_2_2D(0.55,0.2);
    point_1 = point_1_2D;
    point_2 = point_2_2D;
  }
  else if(dim == 3)
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
//  const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
//  cell_point_1 = GridTools::find_active_cell_around_point (mapping,data.back().get_dof_handler(1), point_1);
//  if(cell_point_1.first->is_locally_owned())
//  {
//    counter_1 = 1;
//    //std::cout<< "Point 1 found on Processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
//
//    Vector<double> value(1);
//    my_point_value(mapping,data.back().get_dof_handler(1),solution_n[dim],cell_point_1,value);
//    pressure_1 = value(0);
//  }
//  counter_1 = Utilities::MPI::sum(counter_1,MPI_COMM_WORLD);
//  pressure_1 = Utilities::MPI::sum(pressure_1,MPI_COMM_WORLD);
//  pressure_1 = pressure_1/counter_1;
//
//  const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
//  cell_point_2 = GridTools::find_active_cell_around_point (mapping,data.back().get_dof_handler(1), point_2);
//  if(cell_point_2.first->is_locally_owned())
//  {
//    counter_2 = 1;
//    //std::cout<< "Point 2 found on Processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
//
//    Vector<double> value(1);
//    my_point_value(mapping,data.back().get_dof_handler(1),solution_n[dim],cell_point_2,value);
//    pressure_2 = value(0);
//  }
//  counter_2 = Utilities::MPI::sum(counter_2,MPI_COMM_WORLD);
//  pressure_2 = Utilities::MPI::sum(pressure_2,MPI_COMM_WORLD);
//  pressure_2 = pressure_2/counter_2;
//
//  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
//  {
//    std::string filename = "output/pressure_difference_refine" + Utilities::int_to_string(data.back().get_dof_handler(1).get_tria().n_levels()-1) + "_fedegree" + Utilities::int_to_string(fe_degree) + ".txt"; // filename = "pressure_difference.txt";
//
//    std::ofstream f;
//    if(clear_files)
//    {
//      f.open(filename.c_str(),std::ios::trunc);
//    }
//    else
//    {
//      f.open(filename.c_str(),std::ios::app);
//    }
//    f << std::scientific << std::setprecision(6) << time+time_step << "\t" << pressure_1-pressure_2 << std::endl;
//    f.close();
//  }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal) const
  {
    parallel::distributed::Vector<value_type> src;
    data.back().loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_laplace_diagonal,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_laplace_diagonal_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_laplace_diagonal_boundary_face,
              this, laplace_diagonal, src);
    for (unsigned int i=0; i<data.back().get_constrained_dofs(1).size(); ++i)
      laplace_diagonal.local_element(data.back().get_constrained_dofs(1)[i]) += 1.0;
    if(pure_dirichlet_bc)
    {
		  parallel::distributed::Vector<value_type> vec1(laplace_diagonal);
		  for(unsigned int i=0;i<vec1.local_size();++i)
			  vec1.local_element(i) = 1.;
      parallel::distributed::Vector<value_type> d;
		  d.reinit(laplace_diagonal);
      apply_pressure(vec1,d);
      double length = vec1*vec1;
      double factor = vec1*d;
      laplace_diagonal.add(-2./length,d,factor/pow(length,2.),vec1);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal, unsigned int level) const
  {
    parallel::distributed::Vector<value_type> src;
    data[level].loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_laplace_diagonal,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_laplace_diagonal_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_laplace_diagonal_boundary_face,
              this, laplace_diagonal, src);
    for (unsigned int i=0; i<data[level].get_constrained_dofs(1).size(); ++i)
      laplace_diagonal.local_element(data[level].get_constrained_dofs(1)[i]) += 1.0;
    if(pure_dirichlet_bc)
    {
		  parallel::distributed::Vector<value_type> vec1(laplace_diagonal);
		  for(unsigned int i=0;i<vec1.local_size();++i)
			  vec1.local_element(i) = 1.;
      parallel::distributed::Vector<value_type> d;
		  d.reinit(laplace_diagonal);
      apply_pressure(vec1,d,level);
      double length = vec1*vec1;
      double factor = vec1*d;
      laplace_diagonal.add(-2./length,d,factor/pow(length,2.),vec1);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_laplace_diagonal (const MatrixFree<dim,value_type>        &data,
            parallel::distributed::Vector<double>      &dst,
            const parallel::distributed::Vector<double>    &,
            const std::pair<unsigned int,unsigned int>     &cell_range) const
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

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_laplace_diagonal_face (const MatrixFree<dim,value_type>       &data,
                  parallel::distributed::Vector<double>    &dst,
                  const parallel::distributed::Vector<double>  &,
                  const std::pair<unsigned int,unsigned int>  &face_range) const
  {
//  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);
//  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval_neighbor(data,false,1,1);
//  const unsigned int level = data.get_cell_iterator(0,0)->level();
//
//  for(unsigned int face=face_range.first; face<face_range.second; face++)
//  {
//    fe_eval.reinit (face);
//    fe_eval_neighbor.reinit (face);
//
//    /*VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
//             std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
//      (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5   *stab_factor; */
//
//      double factor = 1.;
//      calculate_penalty_parameter_pressure(factor);
//      //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
//      VectorizedArray<value_type> sigmaF = std::max(fe_eval.read_cell_data(array_penalty_parameter[level]),fe_eval_neighbor.read_cell_data(array_penalty_parameter[level])) * (value_type)factor;
//
//    // element-
//    VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
//    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
//    {
//      for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
//        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
//      for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
//        fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);
//
//      fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);
//
//      fe_eval.evaluate(true,true);
//      fe_eval_neighbor.evaluate(true,true);
//
//      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
//      {
//        VectorizedArray<value_type> valueM = fe_eval.get_value(q);
//        VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);
//
//        VectorizedArray<value_type> jump_value = valueM - valueP;
//        VectorizedArray<value_type> average_gradient =
//            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
//        average_gradient = average_gradient - jump_value * sigmaF;
//
//        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
//        fe_eval.submit_value(-average_gradient,q);
//      }
//      fe_eval.integrate(true,true);
//      local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
//    }
//    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
//      fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
//    fe_eval.distribute_local_to_global(dst);
//
//    // neighbor (element+)
//    VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
//    for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
//    {
//      for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
//        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
//      for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
//        fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);
//
//      fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array(1.);
//
//      fe_eval.evaluate(true,true);
//      fe_eval_neighbor.evaluate(true,true);
//
//      for(unsigned int q=0;q<fe_eval_neighbor.n_q_points;++q)
//      {
//        VectorizedArray<value_type> valueM = fe_eval.get_value(q);
//        VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);
//
//        VectorizedArray<value_type> jump_value = valueM - valueP;
//        VectorizedArray<value_type> average_gradient =
//            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
//        average_gradient = average_gradient - jump_value * sigmaF;
//
//        fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
//        fe_eval_neighbor.submit_value(average_gradient,q);
//      }
//      fe_eval_neighbor.integrate(true,true);
//      local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
//    }
//    for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
//      fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];
//    fe_eval_neighbor.distribute_local_to_global(dst);
//  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_laplace_diagonal_boundary_face (const MatrixFree<dim,value_type>         &data,
                        parallel::distributed::Vector<double>      &dst,
                        const parallel::distributed::Vector<double>    &,
                        const std::pair<unsigned int,unsigned int>    &face_range) const
  {
//    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);
//    const unsigned int level = data.get_cell_iterator(0,0)->level();
//
//    for(unsigned int face=face_range.first; face<face_range.second; face++)
//    {
//      fe_eval.reinit (face);
//
//      //VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
//      //  (value_type)(fe_degree * (fe_degree + 1.0))  *stab_factor;
//
//      double factor = 1.;
//      calculate_penalty_parameter_pressure(factor);
//      //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
//      VectorizedArray<value_type> sigmaF = fe_eval.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;
//
//    VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
//    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
//    {
//      for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
//      {
//        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
//      }
//      fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);
//      fe_eval.evaluate(true,true);
//
//      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
//      {
//        if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
//        {
//          //set pressure gradient in normal direction to zero, i.e. pressure+ = pressure-, grad+ = -grad-
//          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
//          VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
//          average_gradient = average_gradient - jump_value * sigmaF;
//
//          fe_eval.submit_normal_gradient(-0.5*jump_value,q);
//          fe_eval.submit_value(-average_gradient,q);
//        }
//        else if (data.get_boundary_indicator(face) == 1) // outflow boundaries
//        {
//          //set pressure to zero, i.e. pressure+ = - pressure- , grad+ = grad-
//          VectorizedArray<value_type> valueM = fe_eval.get_value(q);
//
//          VectorizedArray<value_type> jump_value = 2.0*valueM;
//          VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
//          average_gradient = average_gradient - jump_value * sigmaF;
//
//          fe_eval.submit_normal_gradient(-0.5*jump_value,q);
//          fe_eval.submit_value(-average_gradient,q);
//        }
//      }
//      fe_eval.integrate(true,true);
//      local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
//      }
//    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
//      fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
//      fe_eval.distribute_local_to_global(dst);
//    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  calculate_diagonal_viscous(std::vector<parallel::distributed::Vector<value_type> > &diagonal,
 unsigned int level) const
  {

    std::vector<parallel::distributed::Vector<double> >  src_tmp;
    //not implemented with symmetric formulation
    Assert(false,ExcInternalError());
    data[level].loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_diagonal_viscous,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_diagonal_viscous_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_diagonal_viscous_boundary_face,
              this, diagonal, src_tmp);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_diagonal_viscous (const MatrixFree<dim,value_type>        &data,
               std::vector<parallel::distributed::Vector<double> >    &dst,
               const std::vector<parallel::distributed::Vector<double> >  &src,
               const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
#ifdef XWALL
    Assert(false,ExcInternalError());
#endif
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif

   for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
   {
     fe_eval_xwall.reinit (cell);
     fe_eval_xwall.evaluate_eddy_viscosity(solution_n,cell);
     fe_eval_xwall.reinit (cell);

    VectorizedArray<value_type> local_diagonal_vector[fe_eval_xwall.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
        fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
      fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));
      fe_eval_xwall.evaluate (true,true,false);
      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        fe_eval_xwall.submit_value (gamma0/time_step*fe_eval_xwall.get_value(q), q);
        fe_eval_xwall.submit_gradient (make_vectorized_array<value_type>(fe_eval_xwall.eddyvisc[q])*fe_eval_xwall.get_gradient(q), q);
      }
      fe_eval_xwall.integrate (true,true);
      local_diagonal_vector[j] = fe_eval_xwall.read_cellwise_dof_value(j);
    }
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
      fe_eval_xwall.write_cellwise_dof_value(j,local_diagonal_vector[j]);
    fe_eval_xwall.distribute_local_to_global (dst.at(0),dst.at(1));
   }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_diagonal_viscous_face (const MatrixFree<dim,value_type>       &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                   const std::pair<unsigned int,unsigned int>  &face_range) const
  {
#ifdef XWALL
    Assert(false,ExcInternalError());
#endif
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,1,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,2);
#endif

    const unsigned int level = data.get_cell_iterator(0,0)->level();

     for(unsigned int face=face_range.first; face<face_range.second; face++)
     {
       fe_eval_xwall.reinit (face);
       fe_eval_xwall_neighbor.reinit (face);
       fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));
       fe_eval_xwall_neighbor.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall_neighbor.read_cell_data(element_volume));
       fe_eval_xwall.reinit (face);
       fe_eval_xwall_neighbor.reinit (face);
       double factor = 1.;
       calculate_penalty_parameter(factor);
       //VectorizedArray<value_type> sigmaF = std::abs(fe_eval_xwall.get_normal_volume_fraction()) * (value_type)factor;
      VectorizedArray<value_type> sigmaF = std::max(fe_eval_xwall.read_cell_data(array_penalty_parameter[level]),fe_eval_xwall_neighbor.read_cell_data(array_penalty_parameter[level])) * (value_type)factor;

       // element-
       VectorizedArray<value_type> local_diagonal_vector[fe_eval_xwall.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
        fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
      for (unsigned int i=0; i<fe_eval_xwall_neighbor.dofs_per_cell; ++i)
        fe_eval_xwall_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));

      fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));

      fe_eval_xwall.evaluate(true,true);
      fe_eval_xwall_neighbor.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        VectorizedArray<value_type> uM = fe_eval_xwall.get_value(q);
        VectorizedArray<value_type> uP = fe_eval_xwall_neighbor.get_value(q);

        VectorizedArray<value_type> jump_value = uM - uP;
        VectorizedArray<value_type> average_gradient =
            ( fe_eval_xwall.get_normal_gradient(q,true) + fe_eval_xwall_neighbor.get_normal_gradient(q,true) ) * make_vectorized_array<value_type>(0.5);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval_xwall.submit_normal_gradient(-0.5*fe_eval_xwall.eddyvisc[q]*jump_value,q);
        fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);
      }
      fe_eval_xwall.integrate(true,true);
      local_diagonal_vector[j] = fe_eval_xwall.read_cellwise_dof_value(j);
       }
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
      fe_eval_xwall.write_cellwise_dof_value(j, local_diagonal_vector[j]);
    fe_eval_xwall.distribute_local_to_global(dst.at(0),dst.at(1));

       // neighbor (element+)
    VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_xwall_neighbor.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval_xwall_neighbor.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
        fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
      for (unsigned int i=0; i<fe_eval_xwall_neighbor.dofs_per_cell; ++i)
        fe_eval_xwall_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));

      fe_eval_xwall_neighbor.write_cellwise_dof_value(j,make_vectorized_array(1.));

      fe_eval_xwall.evaluate(true,true);
      fe_eval_xwall_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
        {
          VectorizedArray<value_type> uM = fe_eval_xwall.get_value(q);
          VectorizedArray<value_type> uP = fe_eval_xwall_neighbor.get_value(q);

          VectorizedArray<value_type> jump_value = uM - uP;
          VectorizedArray<value_type> average_gradient =
              ( fe_eval_xwall.get_normal_gradient(q,true) + fe_eval_xwall_neighbor.get_normal_gradient(q,true) ) * make_vectorized_array<value_type>(0.5);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval_xwall_neighbor.submit_normal_gradient(-0.5*fe_eval_xwall_neighbor.eddyvisc[q]*jump_value,q);
          fe_eval_xwall_neighbor.submit_value(fe_eval_xwall_neighbor.eddyvisc[q]*average_gradient,q);
        }
      fe_eval_xwall_neighbor.integrate(true,true);
      local_diagonal_vector_neighbor[j] = fe_eval_xwall_neighbor.read_cellwise_dof_value(j);
    }
    for (unsigned int j=0; j<fe_eval_xwall_neighbor.dofs_per_cell; ++j)
      fe_eval_xwall_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);
    fe_eval_xwall_neighbor.distribute_local_to_global(dst.at(0),dst.at(1));
     }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_diagonal_viscous_boundary_face (const MatrixFree<dim,value_type>       &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                       const std::pair<unsigned int,unsigned int>  &face_range) const
  {
#ifdef XWALL
    Assert(false,ExcInternalError());
#endif
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
#endif

    const unsigned int level = data.get_cell_iterator(0,0)->level();

     for(unsigned int face=face_range.first; face<face_range.second; face++)
     {
       fe_eval_xwall.reinit (face);
       fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));
       fe_eval_xwall.reinit (face);
       double factor = 1.;
       calculate_penalty_parameter(factor);
       //VectorizedArray<value_type> sigmaF = std::abs(fe_eval_xwall.get_normal_volume_fraction()) * (value_type)factor;
      VectorizedArray<value_type> sigmaF = fe_eval_xwall.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

       VectorizedArray<value_type> local_diagonal_vector[fe_eval_xwall.tensor_dofs_per_cell];
       for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
       {
         for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
      {
           fe_eval_xwall.write_cellwise_dof_value(i, make_vectorized_array(0.));
      }
         fe_eval_xwall.write_cellwise_dof_value(j, make_vectorized_array(1.));
      fe_eval_xwall.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          VectorizedArray<value_type> uM = fe_eval_xwall.get_value(q);
          VectorizedArray<value_type> uP = -uM;

          VectorizedArray<value_type> jump_value = uM - uP;
          VectorizedArray<value_type> average_gradient = fe_eval_xwall.get_normal_gradient(q,true);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval_xwall.submit_normal_gradient(-0.5*fe_eval_xwall.eddyvisc[q]*jump_value,q);
          fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);
        }
        else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
          fe_eval_xwall.submit_normal_gradient(-0.5*fe_eval_xwall.eddyvisc[q]*jump_value,q);
          fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);
        }
      }
      fe_eval_xwall.integrate(true,true);
      local_diagonal_vector[j] = fe_eval_xwall.read_cellwise_dof_value(j);
       }
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
      fe_eval_xwall.write_cellwise_dof_value(j, local_diagonal_vector[j]);
    fe_eval_xwall.distribute_local_to_global(dst.at(0),dst.at(1));
     }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_convection (const std::vector<parallel::distributed::Vector<value_type> >   &src,
            std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<2*dim;++d)
    dst[d] = 0;

  // data.loop
  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection_boundary_face,
            this, dst, src);

  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,
                             this, dst, dst);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_rhs (std::vector<parallel::distributed::Vector<value_type> >  &dst)
  {
  for(unsigned int d=0;d<2*dim;++d)
    dst[d] = 0;
  // data.loop
  data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_rhs,this, dst, dst);

  // data.cell_loop
  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,
                             this, dst, dst);

  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_viscous (const parallel::distributed::BlockVector<value_type>   &src,
              parallel::distributed::BlockVector<value_type>      &dst) const
  {
    for(unsigned int d=0;d<2*dim;++d)
      dst.block(d)=0;
  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous_boundary_face,
            this, dst, src);
  }
//
//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  apply_viscous (const parallel::distributed::BlockVector<value_type>   &src,
//              parallel::distributed::BlockVector<value_type>      &dst,
//            const unsigned int                &level) const
//  {
//    Assert(false,ExcInternalError());
//    std::vector<parallel::distributed::Vector<value_type> > src_tmp;
//    std::vector<parallel::distributed::Vector<value_type> > dst_tmp;
//    dst_tmp.resize(2);
//    src_tmp.push_back(src.block(0));
//    src_tmp.push_back(src.block(1));
//  data[level].loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous,
//            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous_face,
//            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous_boundary_face,
//            this, dst, src);
//  dst.block(0)=dst_tmp.at(0);
//  dst.block(1)=dst_tmp.at(1);
//  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_viscous (const std::vector<parallel::distributed::Vector<value_type> >   &src,
            std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<dim;++d)
    dst[d] = 0;
  for(unsigned int d=0;d<dim;++d)
    dst[d+1+dim] = 0;

  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous_boundary_face,
            this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection (const MatrixFree<dim,value_type>              &data,
            std::vector<parallel::distributed::Vector<double> >      &dst,
            const std::vector<parallel::distributed::Vector<double> >  &src,
            const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
  // inexact integration  (data,0,0) : second argument: which dof-handler, third argument: which quadrature
//  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif
  // exact integration of convective term
//  FEEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> velocity (data,0,2);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_xwall.reinit(cell);
  //    velocity.reinit (cell);
      fe_eval_xwall.read_dof_values(src,0, src, dim+1);
      fe_eval_xwall.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_xwall.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > F;
        outer_product(F,u,u);
        fe_eval_xwall.submit_gradient (F, q);
      }
      fe_eval_xwall.integrate (false,true);
      fe_eval_xwall.distribute_local_to_global (dst,0, dst, dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection_face (const MatrixFree<dim,value_type>               &data,
              std::vector<parallel::distributed::Vector<double> >      &dst,
              const std::vector<parallel::distributed::Vector<double> >  &src,
              const std::pair<unsigned int,unsigned int>          &face_range) const
  {
  // inexact integration
//  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);
//  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval_neighbor(data,false,0,0);

#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,2);
#endif
  // exact integration
//  FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval(data,true,0,2);
//  FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_neighbor(data,false,0,2);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {

    fe_eval_xwall.reinit(face);
    fe_eval_xwall_neighbor.reinit (face);

    fe_eval_xwall.read_dof_values(src, 0, src, dim+1);
//    fe_eval.read_dof_values(src,0);
    fe_eval_xwall.evaluate(true, false);
//    fe_eval.evaluate(true,false);
    fe_eval_xwall_neighbor.read_dof_values(src,0,src,dim+1);
    fe_eval_xwall_neighbor.evaluate(true,false);

  /*  VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);
    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
      VectorizedArray<value_type> uM_n = uM*normal;
      VectorizedArray<value_type> uP_n = uP*normal;
      VectorizedArray<value_type> lambda_qpoint = std::max(std::abs(uM_n), std::abs(uP_n));
      lambda = std::max(lambda_qpoint,lambda);
    } */

    for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_xwall_neighbor.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_xwall.get_normal_vector(q);
      VectorizedArray<value_type> uM_n = uM*normal;
      VectorizedArray<value_type> uP_n = uP*normal;
      VectorizedArray<value_type> lambda;

      // calculation of lambda according to Hesthaven & Warburton
//      for(unsigned int k=0;k<lambda.n_array_elements;++k)
//        lambda[k] = std::abs(uM_n[k]) > std::abs(uP_n[k]) ? std::abs(uM_n[k]) : std::abs(uP_n[k]);//lambda = std::max(std::abs(uM_n), std::abs(uP_n));
      // calculation of lambda according to Hesthaven & Warburton

      // calculation of lambda according to Shahbazi et al.
      Tensor<2,dim,VectorizedArray<value_type> > unity_tensor;
      for(unsigned int d=0;d<dim;++d)
        unity_tensor[d][d] = 1.0;
      Tensor<2,dim,VectorizedArray<value_type> > flux_jacobian_M, flux_jacobian_P;
      outer_product(flux_jacobian_M,uM,normal);
      outer_product(flux_jacobian_P,uP,normal);
      flux_jacobian_M += uM_n*unity_tensor;
      flux_jacobian_P += uP_n*unity_tensor;

      // calculate maximum absolute eigenvalue of flux_jacobian_M: max |lambda(flux_jacobian_M)|
      VectorizedArray<value_type> lambda_max_m = make_vectorized_array<value_type>(0.0);
      for(unsigned int n=0;n<lambda_max_m.n_array_elements;++n)
      {
        LAPACKFullMatrix<value_type> FluxJacobianM(dim);
        for(unsigned int i=0;i<dim;++i)
          for(unsigned int j=0;j<dim;++j)
            FluxJacobianM(i,j) = flux_jacobian_M[i][j][n];
        FluxJacobianM.compute_eigenvalues();
        for(unsigned int d=0;d<dim;++d)
          lambda_max_m[n] = std::max(lambda_max_m[n],std::abs(FluxJacobianM.eigenvalue(d)));
      }

      // calculate maximum absolute eigenvalue of flux_jacobian_P: max |lambda(flux_jacobian_P)|
      VectorizedArray<value_type> lambda_max_p = make_vectorized_array<value_type>(0.0);
      for(unsigned int n=0;n<lambda_max_p.n_array_elements;++n)
      {
        LAPACKFullMatrix<value_type> FluxJacobianP(dim);
        for(unsigned int i=0;i<dim;++i)
          for(unsigned int j=0;j<dim;++j)
            FluxJacobianP(i,j) = flux_jacobian_P[i][j][n];
        FluxJacobianP.compute_eigenvalues();
        for(unsigned int d=0;d<dim;++d)
          lambda_max_p[n] = std::max(lambda_max_p[n],std::abs(FluxJacobianP.eigenvalue(d)));
      }
      // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
      lambda = std::max(lambda_max_m, lambda_max_p);
//      Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
//      Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_xwall_neighbor.get_value(q);
//      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_xwall.get_normal_vector(q);
//      VectorizedArray<value_type> uM_n = uM*normal;
//      VectorizedArray<value_type> uP_n = uP*normal;
//      VectorizedArray<value_type> lambda; //lambda = std::max(std::abs(uM_n), std::abs(uP_n));
//      for(unsigned int k=0;k<lambda.n_array_elements;++k)
//        lambda[k] = std::abs(uM_n[k]) > std::abs(uP_n[k]) ? std::abs(uM_n[k]) : std::abs(uP_n[k]);

      Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
      Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
      Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

      fe_eval_xwall.submit_value(-lf_flux,q);
      fe_eval_xwall_neighbor.submit_value(lf_flux,q);
    }
    fe_eval_xwall.integrate(true,false);
    fe_eval_xwall.distribute_local_to_global(dst,0, dst, dim);
    fe_eval_xwall_neighbor.integrate(true,false);
    fe_eval_xwall_neighbor.distribute_local_to_global(dst,0,dst,dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection_boundary_face (const MatrixFree<dim,value_type>             &data,
                       std::vector<parallel::distributed::Vector<double> >    &dst,
                       const std::vector<parallel::distributed::Vector<double> >  &src,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
  {
  // inexact integration
//    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);

#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
#endif

    for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_xwall.reinit (face);
    fe_eval_xwall.read_dof_values(src,0,src,dim+1);
    fe_eval_xwall.evaluate(true,false);

  /*  VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);
    if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
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

    for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);

        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
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
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_xwall.get_normal_vector(q);
        VectorizedArray<value_type> uM_n = uM*normal;
        VectorizedArray<value_type> uP_n = uP*normal;
        VectorizedArray<value_type> lambda;
        // calculation of lambda according to Hesthaven & Warburton
//        for(unsigned int k=0;k<lambda.n_array_elements;++k)
//          lambda[k] = std::abs(uM_n[k]) > std::abs(uP_n[k]) ? std::abs(uM_n[k]) : std::abs(uP_n[k]);
        // calculation of lambda according to Hesthaven & Warburton

        // calculation of lambda according to Shahbazi et al.
        Tensor<2,dim,VectorizedArray<value_type> > unity_tensor;
        for(unsigned int d=0;d<dim;++d)
          unity_tensor[d][d] = 1.0;
        Tensor<2,dim,VectorizedArray<value_type> > flux_jacobian_M, flux_jacobian_P;
        outer_product(flux_jacobian_M,uM,normal);
        outer_product(flux_jacobian_P,uP,normal);
        flux_jacobian_M += uM_n*unity_tensor;
        flux_jacobian_P += uP_n*unity_tensor;

        // calculate maximum absolute eigenvalue of flux_jacobian_M: max |lambda(flux_jacobian_M)|
        VectorizedArray<value_type> lambda_max_m = make_vectorized_array<value_type>(0.0);
        for(unsigned int n=0;n<lambda_max_m.n_array_elements;++n)
        {
          LAPACKFullMatrix<value_type> FluxJacobianM(dim);
          for(unsigned int i=0;i<dim;++i)
            for(unsigned int j=0;j<dim;++j)
              FluxJacobianM(i,j) = flux_jacobian_M[i][j][n];
          FluxJacobianM.compute_eigenvalues();
          for(unsigned int d=0;d<dim;++d)
            lambda_max_m[n] = std::max(lambda_max_m[n],std::abs(FluxJacobianM.eigenvalue(d)));
        }

        // calculate maximum absolute eigenvalue of flux_jacobian_P: max |lambda(flux_jacobian_P)|
        VectorizedArray<value_type> lambda_max_p = make_vectorized_array<value_type>(0.0);
        for(unsigned int n=0;n<lambda_max_p.n_array_elements;++n)
        {
          LAPACKFullMatrix<value_type> FluxJacobianP(dim);
          for(unsigned int i=0;i<dim;++i)
            for(unsigned int j=0;j<dim;++j)
              FluxJacobianP(i,j) = flux_jacobian_P[i][j][n];
          FluxJacobianP.compute_eigenvalues();
          for(unsigned int d=0;d<dim;++d)
            lambda_max_p[n] = std::max(lambda_max_p[n],std::abs(FluxJacobianP.eigenvalue(d)));
        }
        // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
        lambda = std::max(lambda_max_m, lambda_max_p);
        // calculation of lambda according to Shahbazi et al.
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval_xwall.submit_value(-lf_flux,q);
      }
      else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
      {
        // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_xwall.get_normal_vector(q);
        VectorizedArray<value_type> uM_n = uM*normal;
        VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);

        Tensor<1,dim,VectorizedArray<value_type> > jump_value;
        for(unsigned d=0;d<dim;++d)
          jump_value[d] = 0.0;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = uM*uM_n;
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval_xwall.submit_value(-lf_flux,q);
      }
    }

    fe_eval_xwall.integrate(true,false);
    fe_eval_xwall.distribute_local_to_global(dst,0, dst, dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_rhs (const MatrixFree<dim,value_type>              &data,
          std::vector<parallel::distributed::Vector<double> >      &dst,
          const std::vector<parallel::distributed::Vector<double> >  &,
          const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    // (data,0,0) : second argument: which dof-handler, third argument: which quadrature
//  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);
#ifdef XWALL
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_xwall.reinit (cell);

    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
      Tensor<1,dim,VectorizedArray<value_type> > rhs;
      for(unsigned int d=0;d<dim;++d)
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
      fe_eval_xwall.submit_value (rhs, q);
    }
    fe_eval_xwall.integrate (true,false);
    fe_eval_xwall.distribute_local_to_global (dst,0, dst, dim);
  }
  }

//  template <int dim, int model>
//  class Evaluator
//  {
//    void evaluate()
//    if (model == 0)
//      ...
//    else
//      ...
//
//  };
//
//  template <int dim>
//  class Evaluator<dim,0>
//  {
//    void evaluate()
//    {
//      ...;
//    }
//  }
//
//  template <int dim>
//  class Evaluator<dim,0>
//  {
//    void evaluate()
//    {
//      ...;
//    }
//  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_viscous (const MatrixFree<dim,value_type>        &data,
            parallel::distributed::BlockVector<double>      &dst,
            const parallel::distributed::BlockVector<double>  &src,
            const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_xwall.evaluate_eddy_viscosity(solution_n,cell);
    fe_eval_xwall.reinit (cell);
    fe_eval_xwall.read_dof_values(src,0,src,dim);
    fe_eval_xwall.evaluate (true,true);


    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      fe_eval_xwall.submit_value (gamma0/time_step * fe_eval_xwall.get_value(q), q);
#ifdef SYMMETRIC
      fe_eval_xwall.submit_gradient (fe_eval_xwall.eddyvisc[q]*fe_eval_xwall.get_symmetric_gradient(q), q);
#else
      fe_eval_xwall.submit_gradient (fe_eval_xwall.eddyvisc[q]*fe_eval_xwall.get_gradient(q), q);
#endif
    }
    fe_eval_xwall.integrate (true,true);
    fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_viscous_face (const MatrixFree<dim,value_type>       &data,
                parallel::distributed::BlockVector<double>      &dst,
                const parallel::distributed::BlockVector<double>  &src,
                const std::pair<unsigned int,unsigned int>  &face_range) const
  {
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,0);
#endif

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_xwall.reinit (face);
      fe_eval_xwall_neighbor.reinit (face);
      fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));
      fe_eval_xwall_neighbor.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall_neighbor.read_cell_data(element_volume));
      fe_eval_xwall.reinit (face);
      fe_eval_xwall_neighbor.reinit (face);

      fe_eval_xwall.read_dof_values(src,0,src,dim);
      fe_eval_xwall.evaluate(true,true);
      fe_eval_xwall_neighbor.read_dof_values(src,0,src,dim);
      fe_eval_xwall_neighbor.evaluate(true,true);

//      VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
//               std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
//        (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5    *stab_factor;

      double factor = 1.;
      calculate_penalty_parameter(factor);
      //VectorizedArray<value_type> sigmaF = std::abs(fe_eval_xwall.get_normal_volume_fraction()) * (value_type)factor;
      VectorizedArray<value_type> sigmaF = std::max(fe_eval_xwall.read_cell_data(array_penalty_parameter[level]),fe_eval_xwall_neighbor.read_cell_data(array_penalty_parameter[level])) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {

        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_xwall_neighbor.get_value(q);
        VectorizedArray<value_type> average_viscosity = 0.5*(fe_eval_xwall.eddyvisc[q] + fe_eval_xwall_neighbor.eddyvisc[q]);
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
#ifdef SYMMETRIC
        Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor =
            ( fe_eval_xwall.get_symmetric_gradient(q) + fe_eval_xwall_neighbor.get_symmetric_gradient(q)) * make_vectorized_array<value_type>(0.5);
#else
        Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor =
                    ( fe_eval_xwall.get_gradient(q) + fe_eval_xwall_neighbor.get_gradient(q)) * make_vectorized_array<value_type>(0.5);
#endif
        Tensor<2,dim,VectorizedArray<value_type> > jump_tensor;
        outer_product(jump_tensor,jump_value,fe_eval_xwall.get_normal_vector(q));


        //we do not want to symmetrize the penalty part
        average_gradient_tensor = average_gradient_tensor*average_viscosity - std::max(fe_eval_xwall.eddyvisc[q], fe_eval_xwall_neighbor.eddyvisc[q])*jump_tensor * sigmaF;
//#ifdef SYMMETRIC
//        jump_tensor = fe_eval_xwall.make_symmetric(jump_tensor);
//#endif

        Tensor<1,dim,VectorizedArray<value_type> > average_gradient;
        for (unsigned int comp=0; comp<dim; comp++)
          {
          average_gradient[comp] = average_gradient_tensor[comp][0] *
              fe_eval_xwall.get_normal_vector(q)[0];
            for (unsigned int d=1; d<dim; ++d)
              average_gradient[comp] += average_gradient_tensor[comp][d] *
                fe_eval_xwall.get_normal_vector(q)[d];
          }
#ifdef SYMMETRIC
      fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.make_symmetric(average_viscosity*jump_tensor),q);
      fe_eval_xwall_neighbor.submit_gradient(0.5*fe_eval_xwall.make_symmetric(average_viscosity*jump_tensor),q);
#else
      fe_eval_xwall.submit_gradient(0.5*average_viscosity*jump_tensor,q);
      fe_eval_xwall_neighbor.submit_gradient(0.5*average_viscosity*jump_tensor,q);
#endif
        fe_eval_xwall.submit_value(-average_gradient,q);
        fe_eval_xwall_neighbor.submit_value(average_gradient,q);

      }
      fe_eval_xwall.integrate(true,true);
      fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim);
      fe_eval_xwall_neighbor.integrate(true,true);
      fe_eval_xwall_neighbor.distribute_local_to_global(dst,0,dst,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_viscous_boundary_face (const MatrixFree<dim,value_type>       &data,
                    parallel::distributed::BlockVector<double>      &dst,
                    const parallel::distributed::BlockVector<double>  &src,
                    const std::pair<unsigned int,unsigned int>  &face_range) const
  {
//    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
#endif
    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_xwall.reinit (face);
      fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));

      fe_eval_xwall.reinit (face);

      fe_eval_xwall.read_dof_values(src,0,src,dim);
      fe_eval_xwall.evaluate(true,true);

//    VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
//      (value_type)(fe_degree * (fe_degree + 1.0))   *stab_factor;

      double factor = 1.;
      calculate_penalty_parameter(factor);
      //VectorizedArray<value_type> sigmaF = std::abs(fe_eval_xwall.get_normal_volume_fraction()) * (value_type)factor;
      VectorizedArray<value_type> sigmaF = fe_eval_xwall.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = -uM;
          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
  #ifdef SYMMETRIC
          Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor =
              fe_eval_xwall.get_symmetric_gradient(q);
  #else
          Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor =
                      fe_eval_xwall.get_gradient(q);
  #endif
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor;
          outer_product(jump_tensor,jump_value,fe_eval_xwall.get_normal_vector(q));


          //we do not want to symmetrize the penalty part
          average_gradient_tensor = average_gradient_tensor - jump_tensor * sigmaF;
//  #ifdef SYMMETRIC
//          jump_tensor = fe_eval_xwall.make_symmetric(jump_tensor);
//  #endif

          Tensor<1,dim,VectorizedArray<value_type> > average_gradient;
          for (unsigned int comp=0; comp<dim; comp++)
            {
            average_gradient[comp] = average_gradient_tensor[comp][0] *
                fe_eval_xwall.get_normal_vector(q)[0];
              for (unsigned int d=1; d<dim; ++d)
                average_gradient[comp] += average_gradient_tensor[comp][d] *
                  fe_eval_xwall.get_normal_vector(q)[d];
            }
#ifdef SYMMETRIC
          fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.make_symmetric(fe_eval_xwall.eddyvisc[q]*jump_tensor),q);
#else
          fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.eddyvisc[q]*jump_tensor,q);
#endif
          fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);

        }
        else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
          Tensor<1,dim,VectorizedArray<value_type> > jump_value;
          Tensor<1,dim,VectorizedArray<value_type> > average_gradient;// = make_vectorized_array<value_type>(0.0);
          for(unsigned int i=0;i<dim;i++)
          {
            average_gradient[i] = make_vectorized_array(0.);
            jump_value[i] = make_vectorized_array(0.);
          }
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor;

          outer_product(jump_tensor,jump_value,fe_eval_xwall.get_normal_vector(q));
#ifdef SYMMETRIC
          fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.make_symmetric(fe_eval_xwall.eddyvisc[q]*jump_tensor),q);
#else
          fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.eddyvisc[q]*jump_tensor,q);
#endif
          fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);

        }
      }
      fe_eval_xwall.integrate(true,true);
      fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous (const MatrixFree<dim,value_type>                &data,
              std::vector<parallel::distributed::Vector<double> >      &dst,
              const std::vector<parallel::distributed::Vector<double> >  &src,
              const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
      // (data,0,0) : second argument: which dof-handler, third argument: which quadrature
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_xwall.reinit (cell);
      fe_eval_xwall.read_dof_values(src,0,src,dim);
      fe_eval_xwall.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
      Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_xwall.get_value(q);
      fe_eval_xwall.submit_value (make_vectorized_array<value_type>(1.0/time_step)*u, q);
      }
      fe_eval_xwall.integrate (true,false);
      fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim+1);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous_face (const MatrixFree<dim,value_type>                 &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous_boundary_face (const MatrixFree<dim,value_type>             &data,
                         std::vector<parallel::distributed::Vector<double> >    &dst,
                         const std::vector<parallel::distributed::Vector<double> >  &src,
                         const std::pair<unsigned int,unsigned int>          &face_range) const
  {

#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
#endif

    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_xwall.reinit (face);
      fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));
      fe_eval_xwall.reinit (face);

      double factor = 1.;
      calculate_penalty_parameter(factor);

      VectorizedArray<value_type> sigmaF = fe_eval_xwall.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > g_np;
          for(unsigned int d=0;d<dim;++d)
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

          g_np *= fe_eval_xwall.eddyvisc[q];;
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor;
          outer_product(jump_tensor,g_np,fe_eval_xwall.get_normal_vector(q));

#ifdef SYMMETRIC
          fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.make_symmetric(2.*jump_tensor),q);
#else
          fe_eval_xwall.submit_gradient(jump_tensor,q);
#endif
          fe_eval_xwall.submit_value(2.0*sigmaF*g_np,q);

        }
        else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > h;
          for(unsigned int d=0;d<dim;++d)
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
          for(unsigned d=0;d<dim;++d)
            jump_value[d] = 0.0;

          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor;
          outer_product(jump_tensor,jump_value,fe_eval_xwall.get_normal_vector(q));
//#ifdef SYMMETRIC
//          jump_tensor = fe_eval_xwall.make_symmetric(jump_tensor);
//#endif
#ifdef SYMMETRIC
          fe_eval_xwall.submit_gradient(2.*jump_tensor,q);
#else
          fe_eval_xwall.submit_gradient(jump_tensor,q);
#endif
          fe_eval_xwall.submit_value(fe_eval_xwall.eddyvisc[q]*h,q);
        }
      }

      fe_eval_xwall.integrate(true,true);
      fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim+1);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  precompute_inverse_mass_matrix ()
  {
   std::vector<parallel::distributed::Vector<value_type> > dummy;
  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_precompute_mass_matrix,
                   this, dummy, dummy);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_precompute_mass_matrix (const MatrixFree<dim,value_type>        &data,
      std::vector<parallel::distributed::Vector<value_type> >    &,
      const std::vector<parallel::distributed::Vector<value_type> >  &,
               const std::pair<unsigned int,unsigned int>   &cell_range)
  {

    //initialize routine for non-enriched elements
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> phi(data,0,0);
//#ifdef XWALL
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
//#endif
//no XWALL but with XWALL routine
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall (data,src.at(dim+1),src.at(dim+2),0,0);

/*
   std::vector<Table<2,VectorizedArray<value_type>>> matrices(matrix_free.n_macro_cells());
   for (unsigned int cell=0; cell<matrix.n_macro_cells(); ++cell)
   {
     if (enriched)
     {
       matrices[cell].reinit(fe_eval_xwall.dofs_per_cell, fe_eval_xwall.dofs_per_cell);
       ... fuelle wie unten
     }
   }

   for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
   {
     phi.reinit(cell);
     fe_eval_xwall.read_dof_values(...);
     if (enriched)
     for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; i++)
     {
       VectorizedArray<value_type> sum = VectorizedArray<value_type>();
       for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; i++)
         sum += matrices[cell](i,j) * fe_eval_xwall.begin_dof_values()[j];
     }
   }
*/

   //   FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_xwall (data,0,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit(cell);

    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall.reinit (cell);
      std::vector<FullMatrix<value_type> > matrix;
      {
        FullMatrix<value_type> onematrix(fe_eval_xwall.tensor_dofs_per_cell);
        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
          matrix.push_back(onematrix);
      }
      for (unsigned int j=0; j<fe_eval_xwall.tensor_dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval_xwall.evaluate (true,false,false);
        for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
        {
  //        std::cout << fe_eval_xwall.get_value(q)[0] << std::endl;
          fe_eval_xwall.submit_value (fe_eval_xwall.get_value(q), q);
        }
        fe_eval_xwall.integrate (true,false);

        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
            if(fe_eval_xwall.component_enriched(v))
              (matrix[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
            else//this is a non-enriched element
            {
              if(i<phi.dofs_per_cell && j<phi.dofs_per_cell)
                (matrix[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
              else if(i == j)//diagonal
                (matrix[v])(i,j) = 1.0;
              else
                (matrix[v])(i,j) = 0.0;
            }
      }
//      for (unsigned int i=0; i<10; ++i)
//        std::cout << std::endl;
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        matrix[v].print(std::cout,14,8);

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
        (matrix[v]).gauss_jordan();
      }
      matrices[cell].reinit(fe_eval_xwall.dofs_per_cell, fe_eval_xwall.dofs_per_cell);
      //now apply vectors to inverse matrix
      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
        for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
        {
          VectorizedArray<value_type> value;
          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
            value[v] = (matrix[v])(i,j);
          matrices[cell](i,j) = value;
        }
    }

  //


  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_inverse_mass_matrix (const parallel::distributed::BlockVector<value_type>  &src,
      parallel::distributed::BlockVector<value_type>      &dst) const
  {
    for (unsigned int i = 0; i<2*dim; i++)
      dst.block(i)=0;

  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,
                   this, dst, src);

  for (unsigned int i = 0; i<2*dim; i++)
    dst.block(i)*= time_step/gamma0;

  }
  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_mass_matrix (const MatrixFree<dim,value_type>        &data,
                parallel::distributed::BlockVector<value_type>    &dst,
                const parallel::distributed::BlockVector<value_type>  &src,
                const std::pair<unsigned int,unsigned int>   &cell_range) const
  {

    //initialize routine for non-enriched elements
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> phi(data,0,0);
//    VectorizedArray<value_type> coefficients[FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type>::tensor_dofs_per_cell]
    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, dim, value_type> inverse(phi);
#ifdef XWALL
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#endif

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
#ifdef XWALL
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall.reinit (cell);
    if(fe_eval_xwall.enriched)
    {
      //now apply vectors to inverse matrix
      for (unsigned int idim = 0; idim < dim; ++idim)
      {
        fe_eval_xwall.read_dof_values(src.block(idim),src.block(idim+dim));
        AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
            vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
        fe_eval_xwall.distribute_local_to_global (dst.block(idim),dst.block(idim+dim));
      }
    }
    else
#endif
    {
      phi.reinit(cell);
      phi.read_dof_values(src,0);

      inverse.fill_inverse_JxW_values(coefficients);
      inverse.apply(coefficients,dim,phi.begin_dof_values(),phi.begin_dof_values());

      phi.set_dof_values(dst,0);
    }
  }
  //

  }
  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_mass_matrix (const MatrixFree<dim,value_type>        &data,
      std::vector<parallel::distributed::Vector<value_type> >    &dst,
      const std::vector<parallel::distributed::Vector<value_type> >  &src,
               const std::pair<unsigned int,unsigned int>   &cell_range) const
  {

    if(dst.size()>dim)
    {
    //initialize routine for non-enriched elements
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> phi(data,0,0);
//    VectorizedArray<value_type> coefficients[FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type>::tensor_dofs_per_cell]
    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, dim, value_type> inverse(phi);
#ifdef XWALL
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#endif
//no XWALL but with XWALL routine
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall (data,src.at(dim+1),src.at(dim+2),0,0);

/*
   std::vector<Table<2,VectorizedArray<value_type>>> matrices(matrix_free.n_macro_cells());
   for (unsigned int cell=0; cell<matrix.n_macro_cells(); ++cell)
   {
     if (enriched)
     {
       matrices[cell].reinit(fe_eval_xwall.dofs_per_cell, fe_eval_xwall.dofs_per_cell);
       ... fuelle wie unten
     }
   }

   for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
   {
     phi.reinit(cell);
     fe_eval_xwall.read_dof_values(...);
     if (enriched)
     for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; i++)
     {
       VectorizedArray<value_type> sum = VectorizedArray<value_type>();
       for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; i++)
         sum += matrices[cell](i,j) * fe_eval_xwall.begin_dof_values()[j];
     }
   }
*/

   //   FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_xwall (data,0,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
#ifdef XWALL
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall.reinit (cell);
    if(fe_eval_xwall.enriched)
    {
      //now apply vectors to inverse matrix
      for (unsigned int idim = 0; idim < dim; ++idim)
      {
        fe_eval_xwall.read_dof_values(src.at(idim),src.at(idim+dim));
        AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
            vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
        fe_eval_xwall.distribute_local_to_global (dst.at(idim),dst.at(idim+dim));
      }
    }
    else
#endif
    {
      phi.reinit(cell);
      phi.read_dof_values(src,0);

      inverse.fill_inverse_JxW_values(coefficients);
      inverse.apply(coefficients,dim,phi.begin_dof_values(),phi.begin_dof_values());

      phi.set_dof_values(dst,0);
    }
  }
  //
    }
    else
    {
      FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi (data,0,0);

      AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);
  #ifdef XWALL
     FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#endif
  //no XWALL but with XWALL routine
  //   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall (data,src.at(dim+1),src.at(dim+2),0,0);



     //   FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_xwall (data,0,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
#ifdef XWALL
      //first, check if we have an enriched element
      //if so, perform the routine for the enriched elements
      fe_eval_xwall.reinit (cell);
      if(fe_eval_xwall.enriched)
      {
        //now apply vectors to inverse matrix
          fe_eval_xwall.read_dof_values(src.at(0),src.at(1));
          AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
          for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
            for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
              vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
          for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
            fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
          fe_eval_xwall.distribute_local_to_global (dst.at(0),dst.at(1));
      }
      else
  #endif
      {
        phi.reinit(cell);
        phi.read_dof_values(src.at(0));

        inverse.fill_inverse_JxW_values(coefficients);
        inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());

        phi.set_dof_values(dst.at(0));
      }
    }
    }
  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_apply_mass_matrix(const MatrixFree<dim,value_type>                &data,
//                std::vector<parallel::distributed::Vector<value_type> >    &dst,
//                const std::vector<parallel::distributed::Vector<value_type> >  &src,
//                const std::pair<unsigned int,unsigned int>          &cell_range) const
//  {
//    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> phi(data,0,0);
//
//    const unsigned int dofs_per_cell = phi.dofs_per_cell;
//
//    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
//    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, dim, value_type> inverse(phi);
//
//    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//    {
//    phi.reinit(cell);
//    phi.read_dof_values(src,0);
//
//    inverse.fill_inverse_JxW_values(coefficients);
//    inverse.apply(coefficients,dim,phi.begin_dof_values(),phi.begin_dof_values());
//
//    phi.set_dof_values(dst,0);
//    }
//  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_mass_matrix(const MatrixFree<dim,value_type>          &data,
                parallel::distributed::Vector<value_type>      &dst,
                const std::vector<parallel::distributed::Vector<value_type> >  &src,
                const std::pair<unsigned int,unsigned int>    &cell_range) const
  {
    ;
//    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi(data,0,0);
//
//    const unsigned int dofs_per_cell = phi.dofs_per_cell;
//
//    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
//    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);
//
//    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//    {
//    phi.reinit(cell);
//    phi.read_dof_values(src);
//
//    inverse.fill_inverse_JxW_values(coefficients);
//    inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());
//
//    phi.set_dof_values(dst);
//    }
//    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi (data,0,0);
//
//    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
//    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);
//#ifdef XWALL
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,src.at(dim+1),src.at(dim+2),0,3);
////no XWALL but with XWALL routine
////   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall (data,src.at(dim+1),src.at(dim+2),0,0);
//
//
//
//   //   FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_xwall (data,0,0);
//
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//  {
//    //first, check if we have an enriched element
//    //if so, perform the routine for the enriched elements
//    fe_eval_xwall.reinit (cell);
//    phi.reinit(cell);
//    if(fe_eval_xwall.enriched)
//    {
//      std::vector<FullMatrix<value_type> > matrices;
//      {
//        FullMatrix<value_type> matrix(fe_eval_xwall.tensor_dofs_per_cell);
//        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//          matrices.push_back(matrix);
//      }
//      for (unsigned int j=0; j<fe_eval_xwall.tensor_dofs_per_cell; ++j)
//      {
//        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//          fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
//        fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));
//
//        fe_eval_xwall.evaluate (true,false,false);
//        for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
//        {
//  //        std::cout << fe_eval_xwall.get_value(q)[0] << std::endl;
//          fe_eval_xwall.submit_value (fe_eval_xwall.get_value(q), q);
//        }
//        fe_eval_xwall.integrate (true,false);
//
//        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//            if(fe_eval_xwall.component_enriched(v))
//              (matrices[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
//            else//this is a non-enriched element
//            {
//              if(i<phi.tensor_dofs_per_cell && j<phi.tensor_dofs_per_cell)
//                (matrices[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
//              else if(i == j)//diagonal
//                (matrices[v])(i,j) = 1.0;
//              else
//                (matrices[v])(i,j) = 0.0;
//            }
//      }
////      for (unsigned int i=0; i<10; ++i)
////        std::cout << std::endl;
////      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
////        matrices[v].print(std::cout,14,8);
//
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//      {
//        (matrices[v]).gauss_jordan();
//      }
//
//      //now apply vectors to inverse matrix
//
//        fe_eval_xwall.read_dof_values(src.at(0),src.at(0));
//
//        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        {
//          Vector<value_type> vector_input(fe_eval_xwall.tensor_dofs_per_cell);
//          for (unsigned int j=0; j<fe_eval_xwall.tensor_dofs_per_cell; ++j)
//            vector_input(j)=(fe_eval_xwall.read_cellwise_dof_value(j))[v];
//          Vector<value_type> vector_result(fe_eval_xwall.tensor_dofs_per_cell);
//          (matrices[v]).vmult(vector_result,vector_input);
//          for (unsigned int j=0; j<fe_eval_xwall.tensor_dofs_per_cell; ++j)
//            fe_eval_xwall.write_cellwise_dof_value(j,vector_result(j),v);
//        }
//        fe_eval_xwall.distribute_local_to_global (dst,dst);
//
//    }
//    else
//#endif
//    {
//      phi.read_dof_values(src.at(0));
//
//      inverse.fill_inverse_JxW_values(coefficients);
//      inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());
//
//      phi.set_dof_values(dst);
//    }
//  }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_vorticity (const std::vector<parallel::distributed::Vector<value_type> >   &src,
              std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<2*number_vorticity_components;++d)
    dst[d] = 0;
  // data.loop
  data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_vorticity,this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_vorticity(const MatrixFree<dim,value_type>                  &data,
                std::vector<parallel::distributed::Vector<value_type> >      &dst,
                const std::vector<parallel::distributed::Vector<value_type> >  &src,
                const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
//    //TODO Benjamin the vorticity lives only on the standard space
////#ifdef XWALL
////    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,3);
////    FEEvaluation<dim,fe_degree,n_q_points_1d_xwall,number_vorticity_components,value_type> phi(data,0,3);
////#else
////    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
//    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,0);
////    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,number_vorticity_components,value_type> fe_eval_xwall_phi(data,src.at(dim),src.at(dim+1),0,0);
//    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);

//
#ifdef XWALL
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> velocity_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
   FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
   FEEvaluation<dim,fe_degree,fe_degree+1,number_vorticity_components,value_type> phi(data,0,0);
#else
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,number_vorticity_components,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,0);
   FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
   FEEvaluation<dim,fe_degree,fe_degree+1,number_vorticity_components,value_type> phi(data,0,0);
#endif
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);
  const unsigned int dofs_per_cell = phi.dofs_per_cell;
  AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
//    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);

//no XWALL but with XWALL routine
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall (data,src.at(dim+1),src.at(dim+2),0,0);



   //   FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_xwall (data,0,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit(cell);
#ifdef XWALL
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall.reinit (cell);
    if(fe_eval_xwall.enriched)
    {
      const unsigned int total_dofs_per_cell = fe_eval_xwall.dofs_per_cell * number_vorticity_components;
      velocity_xwall.reinit(cell);
      velocity_xwall.read_dof_values(src,0,src,dim+1);
      velocity_xwall.evaluate (false,true,false);
      std::vector<FullMatrix<value_type> > matrices;
      {
        FullMatrix<value_type> matrix(total_dofs_per_cell);
        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
          matrices.push_back(matrix);
      }
      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval_xwall.evaluate (true,false,false);
        for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
        {
  //        std::cout << fe_eval_xwall.get_value(q)[0] << std::endl;
          fe_eval_xwall.submit_value (fe_eval_xwall.get_value(q), q);
        }
        fe_eval_xwall.integrate (true,false);

        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
            if(fe_eval_xwall.component_enriched(v))
              (matrices[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
            else//this is a non-enriched element
            {
              if(i<phi.dofs_per_cell*number_vorticity_components && j<phi.dofs_per_cell*number_vorticity_components)
                (matrices[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
              else if(i == j)//diagonal
                (matrices[v])(i,j) = 1.0;
              else
                (matrices[v])(i,j) = 0.0;
            }
      }
//      for (unsigned int i=0; i<10; ++i)
//        std::cout << std::endl;
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        matrices[v].print(std::cout,14,8);

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
        (matrices[v]).gauss_jordan();
      }
      //initialize again to get a clean version
      fe_eval_xwall.reinit (cell);
      //now apply vectors to inverse matrix
      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        fe_eval_xwall.submit_value (velocity_xwall.get_curl(q), q);
//        std::cout << velocity_xwall.get_curl(q)[2][0] << "   "  << velocity_xwall.get_curl(q)[2][1] << std::endl;
      }
      fe_eval_xwall.integrate (true,false);


      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
//        std::cout << fe_eval_xwall.dofs_per_cell << std::endl;
//        std::cout << fe_eval_xwall.tensor_dofs_per_cell << std::endl;
        Vector<value_type> vector_input(total_dofs_per_cell);
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          vector_input(j)=(fe_eval_xwall.read_cellwise_dof_value(j))[v];
//        vector_input.print(std::cout);
        Vector<value_type> vector_result(total_dofs_per_cell);
        (matrices[v]).vmult(vector_result,vector_input);
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          fe_eval_xwall.write_cellwise_dof_value(j,vector_result(j),v);
      }
      fe_eval_xwall.distribute_local_to_global (dst,0,dst,number_vorticity_components);

    }
    else
#endif
    {
      velocity.reinit(cell);
      velocity.read_dof_values(src,0);
      velocity.evaluate (false,true,false);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
      Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
//      std::cout << omega[2][0] << "    " << omega[2][1] << std::endl;
        phi.submit_value (omega, q);
      }
      phi.integrate (true,false);

      inverse.fill_inverse_JxW_values(coefficients);
      inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());

      phi.set_dof_values(dst,0);
    }
  }

//    else

//    {
//      phi.read_dof_values(src,0);
//
//      inverse.fill_inverse_JxW_values(coefficients);
//      inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());
//
//      phi.set_dof_values(dst,0);
//    }
//  }

  //


  }

  template <int dim, typename FEEval>
  struct CurlCompute
  {
    static
    Tensor<1,dim,VectorizedArray<typename FEEval::number_type> >
    compute(FEEval     &fe_eval,
        const unsigned int   q_point)
    {
    return fe_eval.get_curl(q_point);
    }
  };

  template <typename FEEval>
  struct CurlCompute<2,FEEval>
  {
  static
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> >
    compute(FEEval     &fe_eval,
        const unsigned int   q_point)
    {
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> > rot;
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> > temp = fe_eval.get_gradient(q_point);
    rot[0] = temp[1];
    rot[1] = - temp[0];
    return rot;
    }
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_P (parallel::distributed::Vector<value_type> &vector) const
  {
    parallel::distributed::Vector<value_type> vec1(vector);
    for(unsigned int i=0;i<vec1.local_size();++i)
      vec1.local_element(i) = 1.;
    vec1.update_ghost_values();
    double scalar = vec1*vector;
    double length = vec1*vec1;
    vector.add(-scalar/length,vec1);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  shift_pressure (parallel::distributed::Vector<value_type>  &pressure)
  {
    parallel::distributed::Vector<value_type> vec1(pressure);
    for(unsigned int i=0;i<vec1.local_size();++i)
      vec1.local_element(i) = 1.;
    AnalyticalSolution<dim> analytical_solution(dim,time+time_step);
    double exact = analytical_solution.value(first_point);
    double current = 0.;
    if (pressure.locally_owned_elements().is_element(dof_index_first_point))
      current = pressure(dof_index_first_point);
    current = Utilities::MPI::sum(current, MPI_COMM_WORLD);
    pressure.add(exact-current,vec1);
  }


  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_pressure (const parallel::distributed::Vector<value_type>    &src,
                  parallel::distributed::Vector<value_type>      &dst) const
  {
  dst = 0;

  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_pressure,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_pressure_face,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_pressure_boundary_face,
        this, dst, src);
  for (unsigned int i=0; i<data.back().get_constrained_dofs(1).size(); ++i)
    dst.local_element(data.back().get_constrained_dofs(1)[i]) +=
        src.local_element(data.back().get_constrained_dofs(1)[i]);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_pressure (const parallel::distributed::Vector<value_type>    &src,
                  parallel::distributed::Vector<value_type>      &dst,
                  const unsigned int                 &level) const
  {
//  dst = 0;
  data[level].loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_pressure,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_pressure_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_pressure_boundary_face,
            this, dst, src);
  for (unsigned int i=0; i<data[level].get_constrained_dofs(1).size(); ++i)
    dst.local_element(data[level].get_constrained_dofs(1)[i]) +=
        src.local_element(data[level].get_constrained_dofs(1)[i]);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_pressure (const MatrixFree<dim,value_type>        &data,
            parallel::distributed::Vector<double>      &dst,
            const parallel::distributed::Vector<double>    &src,
            const std::pair<unsigned int,unsigned int>     &cell_range) const
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

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_pressure_face (const MatrixFree<dim,value_type>       &data,
                parallel::distributed::Vector<double>    &dst,
                const parallel::distributed::Vector<double>  &src,
                const std::pair<unsigned int,unsigned int>  &face_range) const
  {
//    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);
//    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval_neighbor(data,false,1,1);
//
//    const unsigned int level = data.get_cell_iterator(0,0)->level();
//
//    for(unsigned int face=face_range.first; face<face_range.second; face++)
//    {
//      fe_eval.reinit (face);
//      fe_eval_neighbor.reinit (face);
//
//      fe_eval.read_dof_values(src);
//      fe_eval.evaluate(true,true);
//      fe_eval_neighbor.read_dof_values(src);
//      fe_eval_neighbor.evaluate(true,true);
////      VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
////               std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
////        (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5;//   *stab_factor;
//
//      double factor = 1.;
//      calculate_penalty_parameter_pressure(factor);
//      //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
//      VectorizedArray<value_type> sigmaF = std::max(fe_eval.read_cell_data(array_penalty_parameter[level]),fe_eval_neighbor.read_cell_data(array_penalty_parameter[level])) * (value_type)factor;
//
//      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
//      {
//        VectorizedArray<value_type> valueM = fe_eval.get_value(q);
//        VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);
//
//        VectorizedArray<value_type> jump_value = valueM - valueP;
//        VectorizedArray<value_type> average_gradient =
//            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
//        average_gradient = average_gradient - jump_value * sigmaF;
//
//        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
//        fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
//        fe_eval.submit_value(-average_gradient,q);
//        fe_eval_neighbor.submit_value(average_gradient,q);
//      }
//      fe_eval.integrate(true,true);
//      fe_eval.distribute_local_to_global(dst);
//      fe_eval_neighbor.integrate(true,true);
//      fe_eval_neighbor.distribute_local_to_global(dst);
//    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_pressure_boundary_face (const MatrixFree<dim,value_type>           &data,
                      parallel::distributed::Vector<double>      &dst,
                      const parallel::distributed::Vector<double>    &src,
                      const std::pair<unsigned int,unsigned int>    &face_range) const
  {
//  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);
//
//  const unsigned int level = data.get_cell_iterator(0,0)->level();
//
//  for(unsigned int face=face_range.first; face<face_range.second; face++)
//  {
//    fe_eval.reinit (face);
//
//    fe_eval.read_dof_values(src);
//    fe_eval.evaluate(true,true);
//
////    VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
////      (value_type)(fe_degree * (fe_degree + 1.0));//  *stab_factor;
//
//      double factor = 1.;
//      calculate_penalty_parameter_pressure(factor);
//      //VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;
//      VectorizedArray<value_type> sigmaF = fe_eval.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;
//
//    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
//    {
//      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
//      {
//        //set pressure gradient in normal direction to zero, i.e. pressure+ = pressure-, grad+ = -grad-
//        VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
//        VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
//        average_gradient = average_gradient - jump_value * sigmaF;
//
//        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
//        fe_eval.submit_value(-average_gradient,q);
//      }
//      else if (data.get_boundary_indicator(face) == 1) // outflow boundaries
//      {
//        //set pressure to zero, i.e. pressure+ = - pressure- , grad+ = grad-
//        VectorizedArray<value_type> valueM = fe_eval.get_value(q);
//
//        VectorizedArray<value_type> jump_value = 2.0*valueM;
//        VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
//        average_gradient = average_gradient - jump_value * sigmaF;
//
//        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
//        fe_eval.submit_value(-average_gradient,q);
//      }
//    }
//    fe_eval.integrate(true,true);
//    fe_eval.distribute_local_to_global(dst);
//  }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_pressure (const std::vector<parallel::distributed::Vector<value_type> >     &src,
             std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {

  dst[dim] = 0;
  // data.loop
  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_face,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_boundary_face,
        this, dst, src);

  if(pure_dirichlet_bc)
  {  apply_P(dst[dim]);  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure (const MatrixFree<dim,value_type>                &data,
            std::vector<parallel::distributed::Vector<double> >      &dst,
            const std::vector<parallel::distributed::Vector<double> >  &src,
            const std::pair<unsigned int,unsigned int>           &cell_range) const
  {

#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
    FEEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,1,3);
#else
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree_p+1,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,1);
  FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);
#endif

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_xwall.reinit (cell);
    pressure.reinit (cell);
    fe_eval_xwall.read_dof_values(src,0,src,dim);
    fe_eval_xwall.evaluate (false,true,false);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
    VectorizedArray<value_type> divergence = fe_eval_xwall.get_divergence(q);
    pressure.submit_value (-divergence/time_step, q);
    }
    pressure.integrate (true,false);
    pressure.distribute_local_to_global (dst,dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_face (const MatrixFree<dim,value_type>               &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_boundary_face (const MatrixFree<dim,value_type>               &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const
  {

#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_n (data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_nm (data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_nm2 (data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> omega_n(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> omega_nm(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> omega_nm2(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,true,1,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall_n (data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall_nm (data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall_nm2 (data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> omega_n(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> omega_nm(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> omega_nm2(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree+(fe_degree+2)/2,1,value_type> pressure (data,true,1,2);
#endif

    const unsigned int level = data.get_cell_iterator(0,0)->level();

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_xwall_n.reinit(face);
    fe_eval_xwall_n.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall_n.read_cell_data(element_volume));
    pressure.reinit (face);
    fe_eval_xwall_n.reinit (face);
    fe_eval_xwall_n.read_dof_values(solution_n,0,solution_n,dim+1);
    fe_eval_xwall_n.evaluate (true,true);
    fe_eval_xwall_nm.reinit (face);
    fe_eval_xwall_nm.read_dof_values(solution_nm,0,solution_nm,dim+1);
    fe_eval_xwall_nm.evaluate (true,true);
    fe_eval_xwall_nm2.reinit (face);
    fe_eval_xwall_nm2.read_dof_values(solution_nm2,0,solution_nm2,dim+1);
    fe_eval_xwall_nm2.evaluate (true,true);

    omega_n.reinit (face);
    omega_n.read_dof_values(vorticity_n,0,vorticity_n,number_vorticity_components);
    omega_n.evaluate (false,true);
    omega_nm.reinit (face);
    omega_nm.read_dof_values(vorticity_nm,0,vorticity_nm,number_vorticity_components);
    omega_nm.evaluate (false,true);
    omega_nm2.reinit (face);
    omega_nm2.read_dof_values(vorticity_nm2,0,vorticity_nm2,number_vorticity_components);
    omega_nm2.evaluate (false,true);
    //VectorizedArray<value_type> sigmaF = (std::abs( pressure.get_normal_volume_fraction()) ) *
    //  (value_type)(fe_degree * (fe_degree + 1.0)) *stab_factor;

    double factor = 1.;
    calculate_penalty_parameter_pressure(factor);
    //VectorizedArray<value_type> sigmaF = std::abs(pressure.get_normal_volume_fraction()) * (value_type)factor;
    VectorizedArray<value_type> sigmaF = fe_eval_xwall_n.read_cell_data(array_penalty_parameter[level]) * (value_type)factor;

    for(unsigned int q=0;q<pressure.n_q_points;++q)
    {
      if (data.get_boundary_indicator(face) == 0) // Inflow and wall boundaries
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
          for(unsigned int d=0;d<dim;++d)
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
          Tensor<1,dim,VectorizedArray<value_type> > u_n = fe_eval_xwall_n.get_value(q);
          Tensor<2,dim,VectorizedArray<value_type> > grad_u_n = fe_eval_xwall_n.get_gradient(q);
          Tensor<1,dim,VectorizedArray<value_type> > conv_n = grad_u_n * u_n;
          Tensor<1,dim,VectorizedArray<value_type> > u_nm = fe_eval_xwall_nm.get_value(q);
          Tensor<2,dim,VectorizedArray<value_type> > grad_u_nm = fe_eval_xwall_nm.get_gradient(q);
          Tensor<1,dim,VectorizedArray<value_type> > u_nm2 = fe_eval_xwall_nm2.get_value(q);
          Tensor<2,dim,VectorizedArray<value_type> > grad_u_nm2 = fe_eval_xwall_nm2.get_gradient(q);
          Tensor<1,dim,VectorizedArray<value_type> > conv_nm = grad_u_nm * u_nm;
          Tensor<1,dim,VectorizedArray<value_type> > conv_nm2 = grad_u_nm2 * u_nm2;
//          Tensor<1,dim,VectorizedArray<value_type> > rot_n = CurlCompute<dim,decltype(omega_n)>::compute(omega_n,q);
//          Tensor<1,dim,VectorizedArray<value_type> > rot_nm = CurlCompute<dim,decltype(omega_nm)>::compute(omega_nm,q);

          // kaiser cluster: decltype() is unknown
#ifdef XWALL
          Tensor<1,dim,VectorizedArray<value_type> > rot_n = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> >::compute(omega_n,q);
          Tensor<1,dim,VectorizedArray<value_type> > rot_nm = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> >::compute(omega_nm,q);
          Tensor<1,dim,VectorizedArray<value_type> > rot_nm2 = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> >::compute(omega_nm2,q);
#else
          Tensor<1,dim,VectorizedArray<value_type> > rot_n = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> >::compute(omega_n,q);
          Tensor<1,dim,VectorizedArray<value_type> > rot_nm = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> >::compute(omega_nm,q);
          Tensor<1,dim,VectorizedArray<value_type> > rot_nm2 = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> >::compute(omega_nm2,q);
#endif
          // 2nd order extrapolation
//        h = - normal * (make_vectorized_array<value_type>(beta[0])*(dudt_n + conv_n + make_vectorized_array<value_type>(viscosity)*rot_n - rhs_n)
//                + make_vectorized_array<value_type>(beta[1])*(dudt_nm + conv_nm + make_vectorized_array<value_type>(viscosity)*rot_nm - rhs_nm));

        h = - normal * (dudt_np - rhs_np + make_vectorized_array<value_type>(beta[0])*(conv_n + fe_eval_xwall_n.eddyvisc[q]*rot_n)
                + make_vectorized_array<value_type>(beta[1])*(conv_nm + fe_eval_xwall_n.eddyvisc[q]*rot_nm)
                + make_vectorized_array<value_type>(beta[2])*(conv_nm2 + fe_eval_xwall_n.eddyvisc[q]*rot_nm2));
        // Stokes
//        h = - normal * (dudt_np - rhs_np + make_vectorized_array<value_type>(beta[0])*(make_vectorized_array<value_type>(viscosity)*rot_n)
//                        + make_vectorized_array<value_type>(beta[1])*(make_vectorized_array<value_type>(viscosity)*rot_nm));
        // 1st order extrapolation
//        h = - normal * (dudt_np - rhs_np + conv_n + make_vectorized_array<value_type>(viscosity)*rot_n);

        pressure.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
        pressure.submit_value(h,q);
      }
      else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
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

        pressure.submit_normal_gradient(-g,q);
        pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
      }
    }
    pressure.integrate(true,true);
    pressure.distribute_local_to_global(dst,dim);
  }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                  std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<2*dim;++d)
    dst[d] = 0;
  // data.cell_loop
  data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_projection,this, dst, src);
  // data.cell_loop

  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,
                   this, dst, dst);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_projection (const MatrixFree<dim,value_type>              &data,
          std::vector<parallel::distributed::Vector<double> >      &dst,
          const std::vector<parallel::distributed::Vector<double> >  &src,
          const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
    FEEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,1,3);
#else
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree_p+1,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,1);
  FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);
#endif
//  FEEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity (data,0,1);
//  FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_xwall.reinit (cell);
    pressure.reinit (cell);
    pressure.read_dof_values(src,dim);
    pressure.evaluate (false,true,false);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > pressure_gradient = pressure.get_gradient(q);
      fe_eval_xwall.submit_value (-pressure_gradient, q);
    }
    fe_eval_xwall.integrate (true,false);
    fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim);
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
  typedef typename NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;
  NavierStokesProblem(const unsigned int n_refinements);
  void run();

  private:
//  Point<dim> grid_transform (const Point<dim> &in);
  void make_grid_and_dofs ();
  void write_output(std::vector<parallel::distributed::Vector<value_type>>   &solution_n,
             std::vector<parallel::distributed::Vector<value_type>>   &vorticity,
             XWall<dim,fe_degree,fe_degree_xwall>* xwall,
             const unsigned int                     timestep_number);
  void calculate_error(std::vector<parallel::distributed::Vector<value_type>> &solution_n, const double delta_t=0.0);
  void calculate_time_step();

  ConditionalOStream pcout;

  double time, time_step;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;
    FE_DGQArbitraryNodes<dim>  fe;
    FE_Q<dim>  fe_p;
    FE_DGQArbitraryNodes<dim>  fe_xwall;
    DoFHandler<dim>  dof_handler;
    DoFHandler<dim>  dof_handler_p;
    DoFHandler<dim>  dof_handler_xwall;

  const double cfl;
  const unsigned int n_refinements;
  const double output_interval_time;
  };

  template<int dim>
  NavierStokesProblem<dim>::NavierStokesProblem(const unsigned int refine_steps):
  pcout (std::cout,
         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  time(START_TIME),
  triangulation(MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe(QGaussLobatto<1>(fe_degree+1)),
  fe_p(QGaussLobatto<1>(fe_degree_p+1)),
  fe_xwall(QGaussLobatto<1>(fe_degree_xwall+1)),
  dof_handler(triangulation),
  dof_handler_p(triangulation),
  dof_handler_xwall(triangulation),
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

  template <int dim>
  Point<dim> grid_transform (const Point<dim> &in)
  {
    Point<dim> out = in;

    out[0] = in(0)-numbers::PI;
#ifdef XWALL    //wall-model
    out[1] =  2.*in(1)-1.;
#else    //no wall model
    out[1] =  std::tanh(GRID_STRETCH_FAC*(2.*in(1)-1.))/std::tanh(GRID_STRETCH_FAC);
#endif
    out[2] = in(2)-0.5*numbers::PI;
    return out;
  }

  template<int dim>
  void NavierStokesProblem<dim>::make_grid_and_dofs ()
  {
    /* --------------- Generate grid ------------------- */
    //turbulent channel flow
    Point<dim> coordinates;
    coordinates[0] = 2*numbers::PI;
    coordinates[1] = 1.;
    if (dim == 3)
      coordinates[2] = numbers::PI;
    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
//    const double left = -1.0, right = 1.0;
//    GridGenerator::hyper_cube(triangulation,left,right);
//    const unsigned int base_refinements = n_refinements;
    std::vector<unsigned int> refinements(dim, 1);
    //refinements[0] *= 3;
    GridGenerator::subdivided_hyper_rectangle (triangulation,
        refinements,Point<dim>(),
        coordinates);
    // set boundary indicator
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
    //periodicity in x- and z-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    triangulation.begin()->face(0)->set_all_boundary_ids(0+10);
    triangulation.begin()->face(1)->set_all_boundary_ids(1+10);
    //periodicity in z-direction, if dim==3
//    for (unsigned int face=4; face<GeometryInfo<dim>::faces_per_cell; ++face)
    triangulation.begin()->face(4)->set_all_boundary_ids(2+10);
    triangulation.begin()->face(5)->set_all_boundary_ids(3+10);

    GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 0, periodic_faces);
    GridTools::collect_periodic_faces(triangulation, 2+10, 3+10, 2, periodic_faces);
//    for (unsigned int d=2; d<dim; ++d)
//      GridTools::collect_periodic_faces(triangulation, 2*d+10, 2*d+1+10, d, periodic_faces);
    triangulation.add_periodicity(periodic_faces);
    triangulation.refine_global(n_refinements);

    GridTools::transform (&grid_transform<dim>, triangulation);
    // vortex problem
//    const double left = -0.5, right = 0.5;
//    GridGenerator::subdivided_hyper_cube(triangulation,2,left,right);
//
//    triangulation.refine_global(n_refinements);
//
//    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
//    for(;cell!=endc;++cell)
//    {
//    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
//    {
//     if (((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12) && (cell->face(face_number)->center()(1)<0))||
//         ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12) && (cell->face(face_number)->center()(1)>0))||
//         ((std::fabs(cell->face(face_number)->center()(1) - left)< 1e-12) && (cell->face(face_number)->center()(0)<0))||
//         ((std::fabs(cell->face(face_number)->center()(1) - right)< 1e-12) && (cell->face(face_number)->center()(0)>0)))
//        cell->face(face_number)->set_boundary_indicator (1);
//    }
//    }
    // vortex problem

    pcout << std::endl << "Generating grid for " << dim << "-dimensional problem" << std::endl << std::endl
      << "  number of refinements:" << std::setw(10) << n_refinements << std::endl
      << "  number of cells:      " << std::setw(10) << triangulation.n_global_active_cells() << std::endl
      << "  number of faces:      " << std::setw(10) << triangulation.n_active_faces() << std::endl
      << "  number of vertices:   " << std::setw(10) << triangulation.n_vertices() << std::endl;

    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);
    dof_handler_p.distribute_dofs(fe_p);
    dof_handler_xwall.distribute_dofs(fe_xwall);
    dof_handler.distribute_mg_dofs(fe);
    dof_handler_p.distribute_mg_dofs(fe_p);
    dof_handler_xwall.distribute_mg_dofs(fe_xwall);

    float ndofs_per_cell_velocity = pow(float(fe_degree+1),dim)*dim;
    float ndofs_per_cell_pressure = pow(float(fe_degree_p+1),dim);
    float ndofs_per_cell_xwall    = pow(float(fe_degree_xwall+1),dim)*dim;
    pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
      << "Velocity:" << std::endl
      << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree << std::endl
      << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_velocity << std::endl
      << "  number of dofs (velocity):\t" << std::setw(10) << dof_handler.n_dofs()*dim << std::endl
      << "Pressure:" << std::endl
      << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree_p << std::endl
      << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_pressure << std::endl
      << "  number of dofs (pressure):\t" << std::setw(10) << dof_handler_p.n_dofs() << std::endl
      << "Enrichment:" << std::endl
      << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree_xwall << std::endl
      << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_xwall << std::endl
      << "  number of dofs (xwall):\t" << std::setw(10) << dof_handler_xwall.n_dofs()*dim << std::endl;
  }



  template <int dim>
  class Postprocessor : public DataPostprocessor<dim>
  {
  public:
    Postprocessor (const unsigned int partition)
      :
      partition (partition)
    {}

    virtual
    std::vector<std::string>
    get_names() const
    {
      // must be kept in sync with get_data_component_interpretation and
      // compute_derived_quantities_vector
      std::vector<std::string> solution_names (dim, "velocity");
      solution_names.push_back ("p");
      for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back ("velocity_xwall");
      for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back ("vorticity");
      solution_names.push_back ("owner");
      return solution_names;
    }

    virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(3*dim+2, DataComponentInterpretation::component_is_part_of_vector);
      // pressure
      interpretation[dim] = DataComponentInterpretation::component_is_scalar;
      // owner
      interpretation.back() = DataComponentInterpretation::component_is_scalar;
      return interpretation;
    }

    virtual
    UpdateFlags
    get_needed_update_flags () const
    {
      return update_values | update_quadrature_points;
    }

    virtual void
    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                       const std::vector<std::vector<Tensor<1,dim> > > &/*duh*/,
                                       const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                       const std::vector<Point<dim> >                  &/*normals*/,
                                       const std::vector<Point<dim> >                  &evaluation_points,
                                       std::vector<Vector<double> >                    &computed_quantities) const
    {
      const unsigned int n_quadrature_points = uh.size();
      Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
      Assert (uh[0].size() == 4*dim+1,                            ExcInternalError());

      for (unsigned int q=0; q<n_quadrature_points; ++q)
        {
          // TODO: fill in wall distance function
          double wdist = 0.0;
          if(evaluation_points[q][1]<0.0)
            wdist = 1.0+evaluation_points[q][1];
          else
            wdist = 1.0-evaluation_points[q][1];
          //todo: add correct utau
          const double enrichment_func = SimpleSpaldingsLaw::SpaldingsLaw(wdist,1.0);
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](d)
              = (uh[q](d) + uh[q](dim+1+d) * enrichment_func);

          // pressure
          computed_quantities[q](dim) = uh[q](dim);

          // velocity_xwall
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](dim+1+d) = uh[q](dim+1+d);

          // vorticity
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](2*dim+1+d) = uh[q](2*dim+1+d)+uh[q](3*dim+1+d)*enrichment_func;

          // owner
          computed_quantities[q](3*dim+1) = partition;
        }
    }

  private:
    const unsigned int partition;
  };


  template<int dim>
  void NavierStokesProblem<dim>::
  write_output(std::vector<parallel::distributed::Vector<value_type>>   &solution_n,
          std::vector<parallel::distributed::Vector<value_type>>   &vorticity,
          XWall<dim,fe_degree,fe_degree_xwall>* xwall,
          const unsigned int                     output_number)
  {

    // velocity + xwall dofs
    const FESystem<dim> joint_fe (fe, dim,
                                  fe_p, 1,
                                  fe_xwall, dim,
                                  fe, dim,
                                  fe_xwall, dim);

    DoFHandler<dim> joint_dof_handler (dof_handler.get_tria());
    joint_dof_handler.distribute_dofs (joint_fe);
    parallel::distributed::Vector<double>
      joint_solution (joint_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
      loc_vel_dof_indices (fe.dofs_per_cell), loc_pre_dof_indices(fe_p.dofs_per_cell),
      loc_vel_xwall_dof_indices(fe_xwall.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
      joint_cell = joint_dof_handler.begin_active(),
      joint_endc = joint_dof_handler.end(),
      vel_cell = dof_handler.begin_active(),
      pre_cell = dof_handler_p.begin_active(),
      vel_cell_xwall = dof_handler_xwall.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++ pre_cell, ++vel_cell_xwall)
      if (joint_cell->is_locally_owned())
      {
        joint_cell->get_dof_indices (loc_joint_dof_indices);
        vel_cell->get_dof_indices (loc_vel_dof_indices);
        pre_cell->get_dof_indices (loc_pre_dof_indices);
        vel_cell_xwall->get_dof_indices (loc_vel_xwall_dof_indices);
        for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
          switch (joint_fe.system_to_base_index(i).first.first)
            {
            case 0:
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                solution_n[ joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 1:
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                solution_n[ dim ]
                (loc_pre_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 2:
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                solution_n[ dim+1+joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_xwall_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 3:
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                vorticity[ joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 4:
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                vorticity[ dim + joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_xwall_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            default:
              Assert (false, ExcInternalError());
              break;
            }
      }

  Postprocessor<dim> postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

  DataOut<dim> data_out;
  data_out.attach_dof_handler(joint_dof_handler);
  data_out.add_data_vector(joint_solution, postprocessor);

  (*(*xwall).ReturnWDist()).update_ghost_values();
  (*(*xwall).ReturnTauW()).update_ghost_values();
  data_out.add_data_vector (*(*xwall).ReturnDofHandlerWallDistance(),(*(*xwall).ReturnWDist()), "wdist");
  data_out.add_data_vector (*(*xwall).ReturnDofHandlerWallDistance(),(*(*xwall).ReturnTauW()), "tauw");

    std::ostringstream filename;
    filename << "output/"
             << output_prefix
             << "_Proc"
             << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
             << "_"
             << output_number
             << ".vtu";

    data_out.build_patches (3);

    std::ofstream output (filename.str().c_str());

    data_out.write_vtu (output);

  if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {

    std::vector<std::string> filenames;
    for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
    {
      std::ostringstream filename;
      filename << "output/"
               << output_prefix
               << "_Proc"
               << i
               << "_"
               << output_number
               << ".vtu";

        filenames.push_back(filename.str().c_str());
    }
    std::string master_name = output_prefix + "_" + Utilities::int_to_string(output_number) + ".pvtu";
    std::ofstream master_output (master_name.c_str());
    data_out.write_pvtu_record (master_output, filenames);
  }
  }

  template<int dim>
  void NavierStokesProblem<dim>::
  calculate_error(std::vector<parallel::distributed::Vector<value_type>>   &solution_n,
              const double                         delta_t)
  {
  for(unsigned int d=0;d<dim;++d)
  {
    Vector<double> norm_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
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
  VectorTools::integrate_difference (dof_handler_p,
                     solution_n[dim],
                     AnalyticalSolution<dim>(dim,time+delta_t),
                     norm_per_cell,
                     QGauss<dim>(fe.degree+2),
                     VectorTools::L2_norm);
  double solution_norm =
    std::sqrt(Utilities::MPI::sum (norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
  pcout << "error (L2-norm) pressure p:"
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

    // cfl = a * time_step / d_min
    //time_step = cfl * global_min_cell_diameter / global_max_cell_a;
    time_step = cfl * global_min_cell_diameter / MAX_VELOCITY;

    // decrease time_step in order to exactly hit END_TIME
    time_step = (END_TIME-START_TIME)/(1+int((END_TIME-START_TIME)/time_step));

//    time_step = 2.e-4;// 0.1/pow(2.0,8);

    pcout << std::endl << "time step size:\t" << std::setw(10) << time_step << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::run()
  {
  make_grid_and_dofs();

  calculate_time_step();

  NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>  navier_stokes_operation(dof_handler, dof_handler_p, dof_handler_xwall, time_step, periodic_faces);

  // prescribe initial conditions
  for(unsigned int d=0;d<dim;++d)
    VectorTools::interpolate(dof_handler, AnalyticalSolution<dim>(d,time), navier_stokes_operation.solution_n[d]);
  VectorTools::interpolate(dof_handler_p, AnalyticalSolution<dim>(dim,time), navier_stokes_operation.solution_n[dim]);
  for(unsigned int d=0;d<dim;++d)
    VectorTools::interpolate(dof_handler_xwall, AnalyticalSolution<dim>(d+dim+1,time), navier_stokes_operation.solution_n[d+dim+1]);
  navier_stokes_operation.solution_nm = navier_stokes_operation.solution_n;
  navier_stokes_operation.solution_nm2 = navier_stokes_operation.solution_n;

  // compute vorticity from initial data at time t = START_TIME
  {
    navier_stokes_operation.compute_vorticity(navier_stokes_operation.solution_n,navier_stokes_operation.vorticity_n);
//    navier_stokes_operation.compute_eddy_viscosity(navier_stokes_operation.solution_n);
  }
  navier_stokes_operation.vorticity_nm = navier_stokes_operation.vorticity_n;

  unsigned int output_number = 0;
  write_output(navier_stokes_operation.solution_n,
          navier_stokes_operation.vorticity_n,
          navier_stokes_operation.ReturnXWall(),
          output_number++);
    pcout << std::endl << "Write output at START_TIME t = " << START_TIME << std::endl;
//  calculate_error(navier_stokes_operation.solution_n);

  const double EPSILON = 1.0e-10;
  unsigned int time_step_number = 1;

  for(;time<(END_TIME-EPSILON);time+=time_step,++time_step_number)
  {
    navier_stokes_operation.do_timestep(time,time_step,time_step_number);
    pcout << "Step = " << time_step_number << "  t = " << time << std::endl;
    if( (time+time_step) > (output_number*output_interval_time-EPSILON) )
    {
      //distribute constraints in case of periodic BC with continuous pressure
      navier_stokes_operation.DistributeConstraintP(navier_stokes_operation.solution_n[ dim ]);
      write_output(navier_stokes_operation.solution_n,
            navier_stokes_operation.vorticity_n,
            navier_stokes_operation.ReturnXWall(),
            output_number++);
      pcout << std::endl << "Write output at TIME t = " << time+time_step << std::endl;
//      calculate_error(navier_stokes_operation.solution_n,time_step);
    }
  }
  navier_stokes_operation.analyse_computing_times();
  }
}

int main (int argc, char** argv)
{
  try
    {
      using namespace DG_NavierStokes;
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

      deallog.depth_console(0);

      for(unsigned int refine_steps = refine_steps_min;refine_steps <= refine_steps_max;++refine_steps)
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
