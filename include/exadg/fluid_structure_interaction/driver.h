/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_

// application
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>

// utilities
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/timer_tree.h>

// grid
#include <exadg/grid/mapping_degree.h>
#include <exadg/grid/moving_mesh_elasticity.h>
#include <exadg/grid/moving_mesh_poisson.h>
#include <exadg/poisson/spatial_discretization/operator.h>

// IncNS
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>

// Structure
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>

namespace ExaDG
{
namespace FSI
{
using namespace dealii;

/*
 * Own implementation of matrix class.
 */
template<typename Number>
class Matrix
{
public:
  // Constructor.
  Matrix(unsigned int const size) : M(size)
  {
    data.resize(M * M);

    init();
  }

  void
  init()
  {
    for(unsigned int i = 0; i < M; ++i)
      for(unsigned int j = 0; j < M; ++j)
        data[i * M + j] = Number(0.0);
  }

  Number
  get(unsigned int const i, unsigned int const j) const
  {
    AssertThrow(i < M && j < M, ExcMessage("Index exceeds matrix dimensions."));

    return data[i * M + j];
  }

  void
  set(Number const value, unsigned int const i, unsigned int const j)
  {
    AssertThrow(i < M && j < M, ExcMessage("Index exceeds matrix dimensions."));

    data[i * M + j] = value;
  }

private:
  // number of rows and columns of matrix
  unsigned int const  M;
  std::vector<Number> data;
};

template<typename VectorType, typename Number>
void
compute_QR_decomposition(std::vector<VectorType> & Q, Matrix<Number> & R, Number const eps = 1.e-2)
{
  for(unsigned int i = 0; i < Q.size(); ++i)
  {
    Number const norm_initial = Number(Q[i].l2_norm());

    // orthogonalize
    for(unsigned int j = 0; j < i; ++j)
    {
      Number r_ji = Q[j] * Q[i];
      R.set(r_ji, j, i);
      Q[i].add(-r_ji, Q[j]);
    }

    // normalize or drop if linear dependent
    Number r_ii = Number(Q[i].l2_norm());
    if(r_ii < eps * norm_initial)
    {
      Q[i] = 0.0;
      for(unsigned int j = 0; j < i; ++j)
        R.set(0.0, j, i);
      R.set(1.0, i, i);
    }
    else
    {
      R.set(r_ii, i, i);
      Q[i] *= 1. / r_ii;
    }
  }
}

/*
 *  Matrix has to be upper triangular with d_ii != 0 for all 0 <= i < n
 */
template<typename Number>
void
backward_substitution(Matrix<Number> const &      matrix,
                      std::vector<Number> &       dst,
                      std::vector<Number> const & rhs)
{
  int const n = dst.size();

  for(int i = n - 1; i >= 0; --i)
  {
    double value = rhs[i];
    for(int j = i + 1; j < n; ++j)
    {
      value -= matrix.get(i, j) * dst[j];
    }

    dst[i] = value / matrix.get(i, i);
  }
}

template<typename Number, typename VectorType>
void
backward_substitution_multiple_rhs(Matrix<Number> const &          matrix,
                                   std::vector<VectorType> &       dst,
                                   std::vector<VectorType> const & rhs)
{
  int const n = dst.size();

  for(int i = n - 1; i >= 0; --i)
  {
    VectorType value = rhs[i];
    for(int j = i + 1; j < n; ++j)
    {
      value.add(-matrix.get(i, j), dst[j]);
    }

    dst[i].equ(1.0 / matrix.get(i, i), value);
  }
}

template<typename VectorType>
void
inv_jacobian_times_residual(VectorType &                                                  b,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & D_history,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & R_history,
                            std::vector<std::shared_ptr<std::vector<VectorType>>> const & Z_history,
                            VectorType const &                                            residual)
{
  VectorType a = residual;

  // reset
  b = 0.0;

  for(int idx = Z_history.size() - 1; idx >= 0; --idx)
  {
    std::shared_ptr<std::vector<VectorType>> D = D_history[idx];
    std::shared_ptr<std::vector<VectorType>> R = R_history[idx];
    std::shared_ptr<std::vector<VectorType>> Z = Z_history[idx];

    int const           k = Z->size();
    std::vector<double> Z_times_a(k, 0.0);
    for(int i = 0; i < k; ++i)
      Z_times_a[i] = (*Z)[i] * a;

    // add to b
    for(int i = 0; i < k; ++i)
      b.add(Z_times_a[i], (*D)[i]);

    // add to a
    for(int i = 0; i < k; ++i)
      a.add(-Z_times_a[i], (*R)[i]);
  }
}

struct PartitionedData
{
  PartitionedData()
    : method("Aitken"),
      abs_tol(1.e-12),
      rel_tol(1.e-3),
      omega_init(0.1),
      reused_time_steps(0),
      partitioned_iter_max(100),
      geometric_tolerance(1.e-10)
  {
  }

  std::string  method;
  double       abs_tol;
  double       rel_tol;
  double       omega_init;
  unsigned int reused_time_steps;
  unsigned int partitioned_iter_max;

  // tolerance used to locate points at the fluid-structure interface
  double geometric_tolerance;
};

template<int dim, typename Number>
class Driver
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  Driver(std::string const & input_file, MPI_Comm const & comm, bool const is_test);

  static void
  add_parameters(dealii::ParameterHandler & prm, PartitionedData & fsi_data);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree_fluid,
        unsigned int const                            degree_structure,
        unsigned int const                            refine_space_fluid,
        unsigned int const                            refine_space_structure);

  void
  solve() const;

  void
  print_performance_results(double const total_time) const;

private:
  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  unsigned int
  solve_partitioned_problem() const;

  void
  coupling_structure_to_ale(VectorType const & displacement_structure) const;

  void
  solve_ale() const;

  void
  coupling_structure_to_fluid(bool const extrapolate) const;

  void
  coupling_fluid_to_structure() const;

  void
  apply_dirichlet_neumann_scheme(VectorType &       d_tilde,
                                 VectorType const & d,
                                 unsigned int       iteration) const;

  bool
  check_convergence(VectorType const & residual) const;

  void
  print_solver_info_header(unsigned int const i) const;

  void
  print_solver_info_converged(unsigned int const i) const;

  void
  print_partitioned_iterations() const;

  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  /**************************************** STRUCTURE *****************************************/

  // input parameters
  Structure::InputParameters structure_param;

  // triangulation
  std::shared_ptr<Triangulation<dim>> structure_triangulation;

  // mapping
  std::shared_ptr<Mapping<dim>> structure_mapping;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    structure_periodic_faces;

  // material descriptor
  std::shared_ptr<Structure::MaterialDescriptor> structure_material_descriptor;

  // boundary conditions
  std::shared_ptr<Structure::BoundaryDescriptor<dim>> structure_boundary_descriptor;

  // field functions
  std::shared_ptr<Structure::FieldFunctions<dim>> structure_field_functions;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> structure_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     structure_matrix_free;

  // spatial discretization
  std::shared_ptr<Structure::Operator<dim, Number>> structure_operator;

  // temporal discretization
  std::shared_ptr<Structure::TimeIntGenAlpha<dim, Number>> structure_time_integrator;

  // postprocessor
  std::shared_ptr<Structure::PostProcessor<dim, Number>> structure_postprocessor;

  /**************************************** STRUCTURE *****************************************/


  /****************************************** FLUID *******************************************/

  // triangulation
  std::shared_ptr<Triangulation<dim>> fluid_triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    fluid_periodic_faces;

  // mapping
  std::shared_ptr<Mapping<dim>> fluid_static_mapping;

  // moving mapping (ALE)
  std::shared_ptr<MovingMeshBase<dim, Number>> fluid_moving_mapping;

  // mapping (static or moving)
  std::shared_ptr<Mapping<dim>> fluid_mapping;

  // parameters
  IncNS::InputParameters fluid_param;

  std::shared_ptr<IncNS::FieldFunctions<dim>>      fluid_field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> fluid_boundary_descriptor_velocity;
  std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> fluid_boundary_descriptor_pressure;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> fluid_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     fluid_matrix_free;

  // spatial discretization
  std::shared_ptr<IncNS::SpatialOperatorBase<dim, Number>> fluid_operator;

  // temporal discretization
  std::shared_ptr<IncNS::TimeIntBDF<dim, Number>> fluid_time_integrator;

  // Postprocessor
  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> fluid_postprocessor;

  /****************************************** FLUID *******************************************/


  /************************************ ALE - MOVING MESH *************************************/

  // use a PDE solver for moving mesh problem
  std::shared_ptr<MatrixFreeData<dim, Number>> ale_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     ale_matrix_free;

  // Poisson-type mesh smoothing
  Poisson::InputParameters ale_poisson_param;

  std::shared_ptr<Poisson::FieldFunctions<dim>>        ale_poisson_field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> ale_poisson_boundary_descriptor;

  std::shared_ptr<Poisson::Operator<dim, Number, dim>> ale_poisson_operator;

  // elasticity-type mesh smoothing
  Structure::InputParameters ale_elasticity_param;

  std::shared_ptr<Structure::FieldFunctions<dim>>     ale_elasticity_field_functions;
  std::shared_ptr<Structure::BoundaryDescriptor<dim>> ale_elasticity_boundary_descriptor;
  std::shared_ptr<Structure::MaterialDescriptor>      ale_elasticity_material_descriptor;

  std::shared_ptr<Structure::Operator<dim, Number>> ale_elasticity_operator;

  /************************************ ALE - MOVING MESH *************************************/


  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/

  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> structure_to_fluid;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> structure_to_ale;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> fluid_to_structure;

  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/

  /*
   *  Fixed-point iteration.
   */
  PartitionedData fsi_data;

  // required for quasi-Newton methods
  mutable std::vector<std::shared_ptr<std::vector<VectorType>>> D_history, R_history, Z_history;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;

  mutable std::pair<unsigned int, unsigned long long> partitioned_iterations;
};

} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_ */
