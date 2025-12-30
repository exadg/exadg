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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/compressible_navier_stokes/spatial_discretization/operator.h>
#include <exadg/functions_and_boundary_conditions/interpolate.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/grid_related_time_step_restrictions.h>
#include <exadg/operators/quadrature.h>
#include <exadg/operators/solution_projection_between_triangulations.h>
#include <exadg/time_integration/restart.h>

namespace ExaDG
{
namespace CompNS
{
template<int dim, typename Number>
Operator<dim, Number>::Operator(
  std::shared_ptr<Grid<dim> const>               grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>    mapping_in,
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>     field_functions_in,
  Parameters const &                             param_in,
  std::string const &                            field_in,
  MPI_Comm const &                               mpi_comm_in)
  : dealii::EnableObserverPointer(),
    grid(grid_in),
    mapping(mapping_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    param(param_in),
    field(field_in),
    dof_handler(*grid_in->triangulation),
    dof_handler_vector(*grid_in->triangulation),
    dof_handler_scalar(*grid_in->triangulation),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_in) == 0),
    wall_time_operator_evaluation(0.0)
{
  pcout << std::endl << "Construct compressible Navier-Stokes DG operator ..." << std::endl;

  initialize_dof_handler_and_constraints();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_handler_and_constraints()
{
  fe        = create_finite_element<dim>(ElementType::Hypercube, true, dim + 2, param.degree);
  fe_vector = create_finite_element<dim>(ElementType::Hypercube, true, dim, param.degree);
  fe_scalar = create_finite_element<dim>(ElementType::Hypercube, true, 1, param.degree);

  // enumerate degrees of freedom
  dof_handler.distribute_dofs(*fe);
  dof_handler_vector.distribute_dofs(*fe_vector);
  dof_handler_scalar.distribute_dofs(*fe_scalar);

  if((param.restart_data.consider_mapping_write and param.restart_data.write_restart) or
     (param.restart_data.consider_mapping_read_source and param.restarted_simulation))
  {
    fe_mapping =
      create_finite_element<dim>(ElementType::Hypercube, true, dim, param.mapping_degree);
    dof_handler_mapping = std::make_shared<dealii::DoFHandler<dim>>(*grid->triangulation);
    dof_handler_mapping->distribute_dofs(*fe_mapping);
  }

  constraint.close();

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", fe->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
}

template<int dim, typename Number>
void
Operator<dim, Number>::fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const
{
  // append mapping flags
  MappingFlags mapping_flags_operator;
  mapping_flags_operator.cells =
    (dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points |
     dealii::update_normal_vectors | dealii::update_values);
  mapping_flags_operator.inner_faces |= dealii::update_quadrature_points;
  mapping_flags_operator.boundary_faces |= dealii::update_quadrature_points;

  matrix_free_data.append_mapping_flags(mapping_flags_operator);

  // mapping flags required for CFL condition
  MappingFlags flags_cfl;
  flags_cfl.cells = dealii::update_quadrature_points;
  matrix_free_data.append_mapping_flags(flags_cfl);

  // dof handler
  matrix_free_data.insert_dof_handler(&dof_handler, field + dof_index_all);
  matrix_free_data.insert_dof_handler(&dof_handler_vector, field + dof_index_vector);
  matrix_free_data.insert_dof_handler(&dof_handler_scalar, field + dof_index_scalar);

  // constraints
  matrix_free_data.insert_constraint(&constraint, field + dof_index_all);
  matrix_free_data.insert_constraint(&constraint, field + dof_index_vector);
  matrix_free_data.insert_constraint(&constraint, field + dof_index_scalar);

  // Quadrature rule
  unsigned int n_q_points_conv;
  if(param.n_q_points_convective == QuadratureRule::Standard)
    n_q_points_conv = param.degree + 1;
  else if(param.n_q_points_convective == QuadratureRule::Overintegration32k)
    n_q_points_conv = param.degree + (param.degree + 2) / 2;
  else if(param.n_q_points_convective == QuadratureRule::Overintegration2k)
    n_q_points_conv = 2 * param.degree + 1;
  else
    AssertThrow(false, dealii::ExcMessage("Specified quadrature rule is not implemented."));

  unsigned int n_q_points_vis;
  if(param.n_q_points_viscous == QuadratureRule::Standard)
    n_q_points_vis = param.degree + 1;
  else if(param.n_q_points_viscous == QuadratureRule::Overintegration32k)
    n_q_points_vis = param.degree + (param.degree + 2) / 2;
  else if(param.n_q_points_viscous == QuadratureRule::Overintegration2k)
    n_q_points_vis = 2 * param.degree + 1;
  else
    AssertThrow(false, dealii::ExcMessage("Specified quadrature rule is not implemented."));

  pcout << std::endl << "Quadrature rules:" << std::endl << std::endl;
  print_parameter(pcout, "number of 1D q-points (std)", param.degree + 1);
  print_parameter(pcout, "number of 1D q-points (conv)", n_q_points_conv);
  print_parameter(pcout, "number of 1D q-points (vis)", n_q_points_vis);

  std::shared_ptr<dealii::Quadrature<dim>> quadrature_standard =
    create_quadrature<dim>(param.grid.element_type, param.degree + 1);
  matrix_free_data.insert_quadrature(*quadrature_standard, field + quad_index_standard);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_conv =
    create_quadrature<dim>(param.grid.element_type, n_q_points_conv);
  matrix_free_data.insert_quadrature(*quadrature_conv, field + quad_index_overintegration_conv);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_vis =
    create_quadrature<dim>(param.grid.element_type, n_q_points_vis);
  matrix_free_data.insert_quadrature(*quadrature_vis, field + quad_index_overintegration_vis);
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_operators()
{
  // mass operator
  MassOperatorData mass_operator_data;
  mass_operator_data.dof_index  = get_dof_index_all();
  mass_operator_data.quad_index = get_quad_index_standard();
  mass_operator.initialize(*matrix_free, mass_operator_data);

  // inverse mass operator
  InverseMassOperatorData<Number> inverse_mass_operator_data_all;
  inverse_mass_operator_data_all.dof_index  = get_dof_index_all();
  inverse_mass_operator_data_all.quad_index = get_quad_index_standard();
  inverse_mass_operator_data_all.parameters = param.inverse_mass_operator;
  inverse_mass_all.initialize(*matrix_free, inverse_mass_operator_data_all);

  InverseMassOperatorData<Number> inverse_mass_operator_data_vector;
  inverse_mass_operator_data_vector.dof_index  = get_dof_index_vector();
  inverse_mass_operator_data_vector.quad_index = get_quad_index_standard();
  inverse_mass_operator_data_vector.parameters = param.inverse_mass_operator;
  inverse_mass_vector.initialize(*matrix_free, inverse_mass_operator_data_vector);

  InverseMassOperatorData<Number> inverse_mass_operator_data_scalar;
  inverse_mass_operator_data_scalar.dof_index  = get_dof_index_scalar();
  inverse_mass_operator_data_scalar.quad_index = get_quad_index_standard();
  inverse_mass_operator_data_scalar.parameters = param.inverse_mass_operator;
  inverse_mass_scalar.initialize(*matrix_free, inverse_mass_operator_data_scalar);

  // body force operator
  BodyForceOperatorData<dim> body_force_operator_data;
  body_force_operator_data.dof_index  = get_dof_index_all();
  body_force_operator_data.quad_index = get_quad_index_standard();
  body_force_operator_data.rhs_rho    = field_functions->right_hand_side_density;
  body_force_operator_data.rhs_u      = field_functions->right_hand_side_velocity;
  body_force_operator_data.rhs_E      = field_functions->right_hand_side_energy;
  body_force_operator.initialize(*matrix_free, body_force_operator_data);

  // convective operator
  ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.dof_index             = get_dof_index_all();
  convective_operator_data.quad_index            = get_quad_index_overintegration_conv();
  convective_operator_data.bc                    = boundary_descriptor;
  convective_operator_data.heat_capacity_ratio   = param.heat_capacity_ratio;
  convective_operator_data.specific_gas_constant = param.specific_gas_constant;
  convective_operator.initialize(*matrix_free, convective_operator_data);

  // viscous operator
  ViscousOperatorData<dim> viscous_operator_data;
  viscous_operator_data.dof_index             = get_dof_index_all();
  viscous_operator_data.quad_index            = get_quad_index_overintegration_vis();
  viscous_operator_data.IP_factor             = param.IP_factor;
  viscous_operator_data.dynamic_viscosity     = param.dynamic_viscosity;
  viscous_operator_data.reference_density     = param.reference_density;
  viscous_operator_data.thermal_conductivity  = param.thermal_conductivity;
  viscous_operator_data.heat_capacity_ratio   = param.heat_capacity_ratio;
  viscous_operator_data.specific_gas_constant = param.specific_gas_constant;
  viscous_operator_data.bc                    = boundary_descriptor;
  viscous_operator.initialize(*matrix_free, viscous_operator_data);

  if(param.use_combined_operator == true)
  {
    AssertThrow(param.n_q_points_convective == param.n_q_points_viscous,
                dealii::ExcMessage("Use the same number of quadrature points for convective term "
                                   "and viscous term in case of combined operator."));

    CombinedOperatorData<dim> combined_operator_data;
    combined_operator_data.dof_index  = get_dof_index_all();
    combined_operator_data.quad_index = get_quad_index_overintegration_vis();
    combined_operator_data.bc         = boundary_descriptor;

    combined_operator.initialize(*matrix_free,
                                 combined_operator_data,
                                 convective_operator,
                                 viscous_operator);
  }

  // calculators
  p_u_T_calculator.initialize(*matrix_free,
                              get_dof_index_all(),
                              get_dof_index_vector(),
                              get_dof_index_scalar(),
                              get_quad_index_l2_projections(),
                              param.heat_capacity_ratio,
                              param.specific_gas_constant);

  vorticity_calculator.initialize(*matrix_free, get_dof_index_vector(), get_quad_index_standard());

  divergence_calculator.initialize(*matrix_free,
                                   get_dof_index_vector(),
                                   get_dof_index_scalar(),
                                   get_quad_index_standard());

  shear_rate_calculator.initialize(*matrix_free,
                                   get_dof_index_vector(),
                                   get_dof_index_scalar(),
                                   get_quad_index_standard());
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup()
{
  // initialize MatrixFree and MatrixFreeData
  std::shared_ptr<dealii::MatrixFree<dim, Number>> mf =
    std::make_shared<dealii::MatrixFree<dim, Number>>();
  std::shared_ptr<MatrixFreeData<dim, Number>> mf_data =
    std::make_shared<MatrixFreeData<dim, Number>>();

  fill_matrix_free_data(*mf_data);

  mf->reinit(get_mapping(),
             mf_data->get_dof_handler_vector(),
             mf_data->get_constraint_vector(),
             mf_data->get_quadrature_vector(),
             mf_data->data);

  // Subsequently, call the other setup function with MatrixFree/MatrixFreeData objects as
  // arguments.
  this->setup(mf, mf_data);
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
                             std::shared_ptr<MatrixFreeData<dim, Number> const> matrix_free_data_in)
{
  pcout << std::endl << "Setup compressible Navier-Stokes DG operator ..." << std::endl;

  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  // perform setup of data structures that depend on matrix-free object
  setup_operators();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
dealii::types::global_dof_index
Operator<dim, Number>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_all());
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector_scalar(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_scalar());
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector_dim_components(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_vector());
}

template<int dim, typename Number>
void
Operator<dim, Number>::prescribe_initial_conditions(VectorType & src, double const time) const
{
  Utilities::interpolate(
    *mapping, dof_handler, *(this->field_functions->initial_solution), src, time);
}

template<int dim, typename Number>
void
Operator<dim, Number>::serialize_vectors(std::vector<VectorType const *> const & vectors) const
{
  // Write deserialization parameters. These do not change during the simulation, but the data are
  // small and we want to make sure to overwrite them.
  DeserializationParameters deserialization_parameters;
  deserialization_parameters.degree                 = param.degree;
  deserialization_parameters.mapping_degree         = param.mapping_degree;
  deserialization_parameters.consider_mapping_write = param.restart_data.consider_mapping_write;
  deserialization_parameters.triangulation_type     = param.grid.triangulation_type;
  write_deserialization_parameters(mpi_comm, param.restart_data, deserialization_parameters);

  // Attach vectors to triangulation and serialize.
  std::vector<dealii::DoFHandler<dim> const *> dof_handlers{&dof_handler};
  std::vector<std::vector<VectorType const *>> vectors_per_dof_handler{vectors};
  if(param.restart_data.consider_mapping_write)
  {
    store_vectors_in_triangulation_and_serialize(param.restart_data,
                                                 dof_handlers,
                                                 vectors_per_dof_handler,
                                                 this->get_mapping(),
                                                 dof_handler_mapping.get(),
                                                 param.mapping_degree);
  }
  else
  {
    store_vectors_in_triangulation_and_serialize(param.restart_data,
                                                 dof_handlers,
                                                 vectors_per_dof_handler);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::deserialize_vectors(std::vector<VectorType *> const & vectors)
{
  // Store ghost state to recover after deserialization.
  std::vector<bool> const has_ghost_elements = get_ghost_state(vectors);

  // Load the deserialization parameters.
  DeserializationParameters const deserialization_parameters =
    read_deserialization_parameters(mpi_comm, param.restart_data);

  // Load potentially unfitting checkpoint triangulation of TriangulationType.
  std::shared_ptr<dealii::Triangulation<dim>> checkpoint_triangulation =
    deserialize_triangulation<dim>(param.restart_data,
                                   deserialization_parameters.triangulation_type,
                                   mpi_comm);

  // Set up DoFHandlers *as checkpointed*, sequence matches `this->serialize_vectors()`.
  dealii::DoFHandler<dim> checkpoint_dof_handler(*checkpoint_triangulation);

  ElementType const checkpoint_element_type = get_element_type(*checkpoint_triangulation);

  std::shared_ptr<dealii::FiniteElement<dim>> checkpoint_fe = create_finite_element<dim>(
    checkpoint_element_type, true, dim + 2, deserialization_parameters.degree);

  checkpoint_dof_handler.distribute_dofs(*checkpoint_fe);

  std::vector<dealii::DoFHandler<dim> const *> checkpoint_dof_handlers{&checkpoint_dof_handler};

  std::vector<VectorType>                checkpoint_vectors(vectors.size());
  std::vector<std::vector<VectorType *>> checkpoint_vectors_ptr(1);
  checkpoint_vectors_ptr[0].resize(vectors.size());
  for(unsigned int i = 0; i < vectors.size(); ++i)
  {
    checkpoint_vectors[i].reinit(checkpoint_dof_handler.locally_owned_dofs(), mpi_comm);
    checkpoint_vectors_ptr[0][i] = &checkpoint_vectors[i];
  }

  if(param.restart_data.discretization_identical)
  {
    // DoFHandlers need to be setup with `checkpoint_triangulation`, otherwise
    // they are identical. We can simply copy the vector contents.
    load_vectors(checkpoint_vectors_ptr, checkpoint_dof_handlers);
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      vectors[i]->copy_locally_owned_data_from(checkpoint_vectors[i]);
    }
  }
  else
  {
    // Perform projection in case of a non-matching discretization.
    std::vector<dealii::DoFHandler<dim> const *> dof_handlers{&dof_handler};
    std::vector<std::vector<VectorType *>>       vectors_per_dof_handler{vectors};

    // Deserialize mapping from vector or project on reference triangulations.
    check_mapping_deserialization(param.restart_data.consider_mapping_read_source,
                                  deserialization_parameters.consider_mapping_write);
    std::shared_ptr<dealii::Mapping<dim> const> checkpoint_mapping;
    std::shared_ptr<MappingDoFVector<dim, typename VectorType::value_type>>
      checkpoint_mapping_dof_vector;
    if(param.restart_data.consider_mapping_read_source)
    {
      dealii::DoFHandler<dim> checkpoint_dof_handler_mapping(*checkpoint_triangulation);
      std::shared_ptr<dealii::FiniteElement<dim>> checkpoint_fe_mapping =
        create_finite_element<dim>(checkpoint_element_type,
                                   true,
                                   dim,
                                   deserialization_parameters.mapping_degree);
      checkpoint_dof_handler_mapping.distribute_dofs(*checkpoint_fe_mapping);

      checkpoint_mapping_dof_vector = load_vectors(checkpoint_vectors_ptr,
                                                   checkpoint_dof_handlers,
                                                   &checkpoint_dof_handler_mapping,
                                                   deserialization_parameters.mapping_degree);

      checkpoint_mapping = checkpoint_mapping_dof_vector->get_mapping();
    }
    else
    {
      load_vectors(checkpoint_vectors_ptr, checkpoint_dof_handlers);

      // Create dummy linear mapping since we have no mapping serialized to restore.
      std::shared_ptr<dealii::Mapping<dim>> tmp;
      GridUtilities::create_mapping(tmp,
                                    get_element_type(*checkpoint_triangulation),
                                    1 /* mapping_degree */);
      checkpoint_mapping = std::const_pointer_cast<dealii::Mapping<dim> const>(tmp);
    }

    ExaDG::GridToGridProjection::GridToGridProjectionData<dim> data;
    data.rpe_data.rtree_level            = param.restart_data.rpe_rtree_level;
    data.rpe_data.tolerance              = param.restart_data.rpe_tolerance_unit_cell;
    data.rpe_data.enforce_unique_mapping = param.restart_data.rpe_enforce_unique_mapping;

    ExaDG::GridToGridProjection::do_grid_to_grid_projection<dim, Number, VectorType>(
      checkpoint_mapping,
      checkpoint_dof_handlers,
      checkpoint_vectors_ptr,
      dof_handlers,
      *matrix_free,
      vectors_per_dof_handler,
      data);
  }

  // Recover ghost vector state.
  set_ghost_state(vectors, has_ghost_elements);
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate(VectorType & dst, VectorType const & src, Number const time) const
{
  dealii::Timer timer;
  timer.restart();

  evaluate_convective_and_viscous(dst, src, time);

  // shift viscous and convective terms to the right-hand side of the equation
  dst *= -1.0;

  // body force term
  if(param.right_hand_side == true)
  {
    body_force_operator.evaluate_add(dst, src, time);
  }

  // apply inverse mass operator
  inverse_mass_all.apply(dst, dst);

  wall_time_operator_evaluation += timer.wall_time();
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_convective(VectorType &       dst,
                                           VectorType const & src,
                                           Number const       time) const
{
  if(param.equation_type == EquationType::Euler or
     param.equation_type == EquationType::NavierStokes)
  {
    convective_operator.evaluate(dst, src, time);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_viscous(VectorType &       dst,
                                        VectorType const & src,
                                        Number const       time) const
{
  if(param.equation_type == EquationType::NavierStokes)
  {
    viscous_operator.evaluate(dst, src, time);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_convective_and_viscous(VectorType &       dst,
                                                       VectorType const & src,
                                                       Number const       time) const
{
  if(param.use_combined_operator == true)
  {
    // viscous and convective terms
    combined_operator.evaluate(dst, src, time);
  }
  else // apply operators separately
  {
    // set dst to zero
    dst = 0.0;

    // viscous operator
    if(param.equation_type == EquationType::NavierStokes)
    {
      viscous_operator.evaluate_add(dst, src, time);
    }

    // convective operator
    if(param.equation_type == EquationType::Euler or
       param.equation_type == EquationType::NavierStokes)
    {
      convective_operator.evaluate_add(dst, src, time);
    }
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_inverse_mass(VectorType & dst, VectorType const & src) const
{
  // apply inverse mass operator
  inverse_mass_all.apply(dst, src);
}

template<int dim, typename Number>
dealii::MatrixFree<dim, Number> const &
Operator<dim, Number>::get_matrix_free() const
{
  return *matrix_free;
}

template<int dim, typename Number>
dealii::Mapping<dim> const &
Operator<dim, Number>::get_mapping() const
{
  return *mapping;
}

template<int dim, typename Number>
dealii::FiniteElement<dim> const &
Operator<dim, Number>::get_fe() const
{
  return *fe;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler_scalar() const
{
  return dof_handler_scalar;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler_vector() const
{
  return dof_handler_vector;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index_vector() const
{
  return matrix_free_data->get_dof_index(field + dof_index_vector);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index_scalar() const
{
  return matrix_free_data->get_dof_index(field + dof_index_scalar);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index_all() const
{
  return matrix_free_data->get_dof_index(field + dof_index_all);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index_standard() const
{
  return matrix_free_data->get_quad_index(field + quad_index_standard);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index_overintegration_conv() const
{
  return matrix_free_data->get_quad_index(field + quad_index_overintegration_conv);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index_overintegration_vis() const
{
  return matrix_free_data->get_quad_index(field + quad_index_overintegration_vis);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index_l2_projections() const
{
  return matrix_free_data->get_quad_index(field + quad_index_l2_projections);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_pressure(VectorType & dst, VectorType const & src) const
{
  p_u_T_calculator.compute_pressure(dst, src);
  inverse_mass_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_velocity(VectorType & dst, VectorType const & src) const
{
  p_u_T_calculator.compute_velocity(dst, src);
  inverse_mass_vector.apply(dst, dst);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_temperature(VectorType & dst, VectorType const & src) const
{
  p_u_T_calculator.compute_temperature(dst, src);
  inverse_mass_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_vorticity(VectorType & dst, VectorType const & src) const
{
  vorticity_calculator.compute_projection_rhs(dst, src);
  inverse_mass_vector.apply(dst, dst);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_divergence(VectorType & dst, VectorType const & src) const
{
  divergence_calculator.compute_projection_rhs(dst, src);
  inverse_mass_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
Operator<dim, Number>::compute_shear_rate(VectorType & dst, VectorType const & src) const
{
  shear_rate_calculator.compute_projection_rhs(dst, src);
  inverse_mass_scalar.apply(dst, dst);
}

template<int dim, typename Number>
double
Operator<dim, Number>::get_wall_time_operator_evaluation() const
{
  return wall_time_operator_evaluation;
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_time_step_cfl_global() const
{
  // speed of sound a = sqrt(gamma * R * T)
  double const speed_of_sound =
    sqrt(param.heat_capacity_ratio * param.specific_gas_constant * param.max_temperature);
  double const acoustic_wave_speed = param.max_velocity + speed_of_sound;

  std::shared_ptr<dealii::Function<dim>> const velocity_field =
    std::make_shared<dealii::Functions::ConstantFunction<dim>>(acoustic_wave_speed, dim);

  return calculate_time_step_cfl_local<dim, Number>(
    *matrix_free,
    get_dof_index_vector(),
    get_quad_index_standard(),
    velocity_field,
    param.start_time /* will not be used (ConstantFunction) */,
    param.degree,
    param.exponent_fe_degree_cfl,
    CFLConditionType::VelocityComponents,
    mpi_comm);
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_time_step_diffusion() const
{
  double const h_min =
    calculate_minimum_vertex_distance(dof_handler.get_triangulation(), get_mapping(), mpi_comm);

  return ExaDG::calculate_const_time_step_diff(param.dynamic_viscosity / param.reference_density,
                                               h_min,
                                               param.degree,
                                               param.exponent_fe_degree_viscous);
}

template class Operator<2, float>;
template class Operator<2, double>;

template class Operator<3, float>;
template class Operator<3, double>;

} // namespace CompNS
} // namespace ExaDG
