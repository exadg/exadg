/*
 * orr_sommerfeld_equation.h
 *
 *  Created on: Aug 31, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ORR_SOMMERFELD_EQUATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ORR_SOMMERFELD_EQUATION_H_


#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <complex>

extern "C" void zggev_(char*,char*,int*,
                       std::complex<double>*,int*,                  /*matrix A*/
                       std::complex<double>*,int*,                  /*matrix B*/
                       std::complex<double>*,std::complex<double>*, /*eigenvalues*/
                       std::complex<double>*,int*,                  /*left eigenvectors*/
                       std::complex<double>*,int*,                  /*right eigenvectors*/
                       std::complex<double>*,int*,double*,int*);

/*
extern "C" void zgetrf_(int*,int*,std::complex<double>*,int*,int*,int*);
*/

using namespace dealii;

/*
 *  Orr-Sommerfeld equation:
 *
 *   i alpha * ((psi'' - alpha² psi)*(U-c) - U'' psi) = 1/Re * (psi'''' - 2 alpha² psi'' + alpha⁴ psi)
 *
 *   where U = (1-y^2) -> U'' = -2
 *   and   c = omega/alpha
 *
 *   Define lambda = - i omega:
 *
 *   1/Re * (psi'''' - 2 alpha² psi'' + alpha⁴ psi)
 *   - 2*i*alpha psi - i*alpha (1-y²) (psi'' - alpha² psi) = - i*omega (psi'' - alpha² psi)
 *                                                         =    lambda (psi'' - alpha² psi)
 *
 *   Discretize this equation to arrive at the generalized eigenvalue problem
 *
 *    A * EV = lambda B * EV
 *
 */
template<int dim=1>
void compute_eigenvector(std::vector<std::complex<double> > &eigenvector,
                         std::complex<double>               &omega,
                         double const                       &Reynolds_number,
                         double const                       &wave_number,
                         FE_DGQ<dim>                        &fe)
{
  std::cout << std::endl
            << "Solution of Orr-Sommerfeld equation for Poiseuille flow"
            << std::endl << std::endl;

  const unsigned int fe_degree = fe.get_degree();
  const double alpha = wave_number;
  const double Re = Reynolds_number;

  AssertThrow(dim==1, ExcNotImplemented());
  AssertThrow(fe_degree>3, ExcNotImplemented());

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -1, 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  Table<2,std::complex<double> > A(dof_handler.n_dofs(),
                                   dof_handler.n_dofs());
  Table<2,std::complex<double> > B(dof_handler.n_dofs(),
                                   dof_handler.n_dofs());

  // ensures correct integration of (1-y^2) terms
  const unsigned int n_q_points = fe_degree+2;
  QGauss<dim> quadrature(n_q_points);
  FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients |
                          update_JxW_values | update_quadrature_points | update_hessians);
  std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);

  fe_values.reinit(dof_handler.begin_active());
  dof_handler.begin_active()->get_dof_indices(dof_indices);

  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
  {
    for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
    {
      double sum_a = 0, sum_b = 0;
      double sum_a_imag = 0;
      for (unsigned int q=0; q<n_q_points; ++q)
      {
        const double phi = fe_values.shape_value(i,q);
        const double psi = fe_values.shape_value(j,q);
        const double dphi = fe_values.shape_grad(i,q)[0];
        const double dpsi = fe_values.shape_grad(j,q)[0];
        const double ddphi = fe_values.shape_hessian(i,q)[0][0];
        const double ddpsi = fe_values.shape_hessian(j,q)[0][0];

        // 1/Re * [(phi'', psi'') + 2*alpha^2*(phi',psi') + alpha^4*(phi,psi)]
        sum_a += (ddphi * ddpsi + 2.*alpha*alpha*dphi*dpsi +
                  alpha*alpha*alpha*alpha*phi*psi) * fe_values.JxW(q)/Re;

        // alpha * (phi,-2*psi-(1-y^2)*(psi''-alpha^2*psi))
        const double y=fe_values.quadrature_point(q)[0];
        sum_a_imag -= (phi*alpha*(2*psi+(1-y*y)*(ddpsi-alpha*alpha*psi)))*fe_values.JxW(q);

        // alpha * (phi,psi''-alpha^2*psi)
        sum_b += phi*(ddpsi-alpha*alpha*psi)*fe_values.JxW(q);
      }
      const types::global_dof_index ii = dof_indices[i];
      const types::global_dof_index jj = dof_indices[j];
      A(jj,ii) = std::complex<double>(sum_a,sum_a_imag);
      B(jj,ii) = std::complex<double>(sum_b,0);
    }
  }

  // apply Dirichlet boundary conditions. the conditions psi(-1) = psi(1) = 0
  // are easy as they simply correspond to deleting the first and last row and
  // colum, but psi'(-1) = psi'(1) = 0 is more involved because we need to
  // build a constraint that eliminates the second and second to last row by
  // an expression of all other rows (except the first and last one).
  //
  // the algorithm to compute the constraints is described e.g. in
  // M.S. Shephard (1984), LINEAR MULTIPOINT CONSTRAINTS APPLIED VIA
  // TRANSFORMATION AS PART OF A DIRECT STIFFNESS ASSEMBLY PROCESS, IJNME 20.
  // we reformulate eq (2) by inverting the matrix G_i and add that into the
  // matrix G_j as in eq (3)
  const int m = A.n_rows();
  LAPACKFullMatrix<double> constraints(4, m-4);
  for (int i=0; i<m-4; ++i)
    constraints(2,i) = fe.shape_grad(i+2,Point<dim>(0.))[0];
  for (int i=0; i<m-4; ++i)
    constraints(3,i) = fe.shape_grad(i+2,Point<dim>(1.))[0];
  LAPACKFullMatrix<double> Gmat(4,4);
  Gmat(0,0) = -1;
  Gmat(1,1) = -1;
  Gmat(2,0) = -fe.shape_grad(0,Point<dim>(0.))[0];
  Gmat(2,1) = -fe.shape_grad(m-1,Point<dim>(0.))[0];
  Gmat(2,2) = -fe.shape_grad(1,Point<dim>(0.))[0];
  Gmat(2,3) = -fe.shape_grad(m-2,Point<dim>(0.))[0];
  Gmat(3,0) = -fe.shape_grad(0,Point<dim>(1.))[0];
  Gmat(3,1) = -fe.shape_grad(m-1,Point<dim>(1.))[0];
  Gmat(3,2) = -fe.shape_grad(1,Point<dim>(1.))[0];
  Gmat(3,3) = -fe.shape_grad(m-2,Point<dim>(1.))[0];
  Gmat.compute_lu_factorization();
  Gmat.apply_lu_factorization(constraints, false);

  /*
  for (unsigned int i=0; i<4; ++i)
    {
      for (int j=0; j<m-4; ++j)
        std::cout << constraints(i,j) << " ";
      std::cout << std::endl;
    }
  */

  Table<2,std::complex<double> > AA(m-4, m-4);
  Table<2,std::complex<double> > BB(m-4, m-4);

  // create a lambda that eliminates the columns
  auto eliminate_constraints = [](const Table<2,std::complex<double>> &M,
                                  const unsigned int                  m,
                                  const LAPACKFullMatrix<double>      &constraints,
                                  Table<2,std::complex<double>>       &MM) -> void
  {
    for (unsigned int i=0; i<m-4; ++i)
    {
      for (unsigned int j=0; j<m-4; ++j)
      {
        MM(i,j) = M(i+2,j+2) +
        constraints(2,j)*M(i+2,1) + constraints(3,j)*M(i+2,m-2) +
        constraints(2,i)*M(1,j+2) + constraints(3,i)*M(m-2,j+2) +
        constraints(2,i)*M(1,1)*constraints(2,j) + constraints(2,i)*M(1,m-2)*constraints(3,j)+
        constraints(3,i)*M(m-2,1)*constraints(2,j) + constraints(3,i)*M(m-2,m-2)*constraints(3,j);
      }
    }
  };
  eliminate_constraints(A,m,constraints,AA);
  eliminate_constraints(B,m,constraints,BB);

  /*
  std::cout << "A:" << std::endl;
  for (unsigned int i=0; i<AA.size(0); ++i)
    {
      for (unsigned int j=0; j<AA.size(1); ++j)
        std::cout << AA(i,j) << " ";
      std::cout << std::endl;
    }
  std::cout << "B:" << std::endl;
  for (unsigned int i=0; i<AA.size(0); ++i)
    {
      for (unsigned int j=0; j<AA.size(1); ++j)
        std::cout << BB(i,j) << " ";
      std::cout << std::endl;
    }
  */

  int n=AA.n_rows();
  char left='N';
  char right='V';
  std::vector<std::complex<double> > eigval_alpha(AA.n_rows());
  std::vector<std::complex<double> > eigval_beta(AA.n_rows());
  Table<2,std::complex<double> > eigvec(AA.n_rows(), AA.n_rows());
  std::vector<std::complex<double> > work(1);
  int lwork=-1;
  std::vector<double> dwork(8*AA.n_rows());
  int info = 0;
  zggev_(&left, &right, &n, &AA(0,0), &n, &BB(0,0), &n, eigval_alpha.data(),
         eigval_beta.data(), &eigvec(0,0), &n, &eigvec(0,0), &n,
         work.data(), &lwork, dwork.data(), &info);
  lwork=work[0].real();
  work.resize(lwork);
  std::cout << "Size of work array: " << lwork << std::endl;
  zggev_(&left, &right, &n, &AA(0,0), &n, &BB(0,0), &n, eigval_alpha.data(),
         eigval_beta.data(), &eigvec(0,0), &n, &eigvec(0,0), &n,
         work.data(), &lwork, dwork.data(), &info);
  std::cout << "Success of eigenvalue computation: " << info << std::endl;

  // lambda = - i*omega = omega_imag - i*omega_real
  // instability means omega_imag > 0 --> lambda_real > 0
  unsigned int const precision = 6;
  std::cout << std::endl << "Positive eigenvalues: " << std::endl;
  int positive_i = -1;
  for (unsigned int i=0; i<eigval_alpha.size(); ++i)
  {
    std::complex<double> eigval = eigval_alpha[i]/eigval_beta[i];
    if (eigval.real() > 0)
    {
      positive_i = i;

      // output
      //std::cout << eigval <<  "  [" << eigval_alpha[i] << " " << eigval_beta[i] << "]   " << std::endl;
      std::cout << std::scientific << std::setprecision(precision) << eigval << std::endl;

      // calculate omega
      std::complex<double> i(0.0,1.0);
      omega = i * eigval;

      //std::cout<< "Omega = " << omega<<std::endl;
    }
  }

  // compute eigenvector
  if (positive_i != -1)
  {
    eigenvector.resize(m);

    // find max value
    std::complex<double> max_value(0.0,0.0);
    for(int i=0; i<n; ++i)
    {
      if(std::norm(eigvec(positive_i,i)) > std::norm(max_value))
        max_value = eigvec(positive_i,i);
    }

    // normalize eigenvector so that the maximum value is 1
    if(std::norm(max_value) > 1.e-12)
      max_value = 1./max_value;
    else
      max_value = 1.;
    for (int i=0; i<m-4; ++i)
      eigenvector[i+2] = eigvec(positive_i,i)*max_value;

    for (int i=0; i<m-4; ++i)
    {
      // eigenvector[0] = 0 (contraint)
      eigenvector[1] += constraints(2,i) * eigenvector[i+2];
      eigenvector[m-2] += constraints(3,i) * eigenvector[i+2];
      // eigenvector[m-1] = 0 (constraint)
    }
 
    /*
    // print eigenvector
    std::cout << std::endl << "Eigenvector on uniform grid: " << std::endl;
    const int n_output = fe_degree + fe_degree%2;
    for (int i=0; i<=n_output; ++i)
    {
      const double y_unit = (double)i/n_output;
      std::complex<double> evaluated = 0;
      for (int j=0; j<m; ++j)
        evaluated += eigenvector[j] * fe.shape_value(j,Point<dim>(y_unit));

      std::cout << std::scientific << std::setprecision(precision)
                << std::setw(precision+8) << std::right << (-1+2.*y_unit)
                << std::setw(precision+8) << std::right << evaluated.real()
                << std::setw(precision+8) << std::right << evaluated.imag()
                << std::endl;
    }
    std::cout << std::endl;
    */

    /*
    // ensure correctness of constraint psi'(-1)=0 on the left
    std::complex<double> derivative = 0;
    for (int i=0; i<m; ++i)
      derivative += eigenvector[i] * fe.shape_grad(i,Point<dim>(0.))[0];
    std::cout << "Correctness of constraint: " << derivative << std::endl;
    */
  }
}


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ORR_SOMMERFELD_EQUATION_H_ */
