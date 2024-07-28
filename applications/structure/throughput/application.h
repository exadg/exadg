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

#ifndef STRUCTURE_THROUGHPUT
#define STRUCTURE_THROUGHPUT

#include <exadg/grid/deformed_cube_manifold.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
    {
      prm.add_parameter("SpatialIntegration",    spatial_integration,      "Use spatial integration.");
      prm.add_parameter("ForceMaterialResidual", force_material_residual,  "Use undeformed configuration to evaluate the residual.");
      prm.add_parameter("CacheLevel",            cache_level,              "Cache level: 0 none, 1 scalars, 2 tensors.");
      prm.add_parameter("CheckType",             check_type,               "Check type for deformation gradient.");
      prm.add_parameter("MappingStrength",       mapping_strength,         "Strength of the mapping applied.");
      prm.add_parameter("ProblemType",           problem_type,             "Problem type considered, QuasiStatic vs Unsteady vs. Steady");
      prm.add_parameter("MaterialType",          material_type,            "StVenantKirchhoff vs. IncompressibleNeoHookean");
      prm.add_parameter("WeakDamping",           weak_damping_coefficient, "Weak damping coefficient for unsteady problems.");
    }
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  set_parameters() final
  {
    this->param.problem_type            = problem_type;
    this->param.body_force              = true;
    this->param.large_deformation       = true;
    this->param.pull_back_body_force    = false;
    this->param.pull_back_traction      = false;
    this->param.spatial_integration     = spatial_integration;
    this->param.cache_level             = cache_level;
    this->param.check_type              = check_type;
    this->param.force_material_residual = force_material_residual;

    this->param.density = density;
    if(this->param.problem_type == ProblemType::Unsteady and weak_damping_coefficient > 0.0)
    {
      this->param.weak_damping_active      = true;
      this->param.weak_damping_coefficient = weak_damping_coefficient;
    }

    // Using a Lagrangian description, we can simplify the mapping for this box.
    if(spatial_integration or mapping_strength > 1e-12)
    {
      this->param.mapping_degree              = this->param.degree;
      this->param.mapping_degree_coarse_grids = this->param.degree;
    }
    else
    {
      this->param.mapping_degree              = 1;
      this->param.mapping_degree_coarse_grids = 1;
    }

    this->param.grid.element_type = ElementType::Hypercube; // Simplex;
    if(this->param.grid.element_type == ElementType::Simplex)
    {
      this->param.grid.triangulation_type           = TriangulationType::FullyDistributed;
      this->param.grid.create_coarse_triangulations = true;
    }
    else if(this->param.grid.element_type == ElementType::Hypercube)
    {
      this->param.grid.triangulation_type           = TriangulationType::Distributed;
      this->param.grid.create_coarse_triangulations = false; // can also be set to true if desired
    }

    // These should not be needed for the throughput applications.
    this->param.solver = Solver::FGMRES;

    this->param.use_matrix_based_implementation = false;
    this->param.sparse_matrix_type              = SparseMatrixType::Trilinos;

    // Output all the application parameters to screen.
    std::string problem_type_str = "Undefined";
    if(problem_type == ProblemType::Steady)
    {
      problem_type_str = "Steady";
    }
    else if(problem_type == ProblemType::Unsteady)
    {
      problem_type_str = "Unsteady";
    }
    else if(problem_type == ProblemType::QuasiStatic)
    {
      problem_type_str = "QuasiStatic";
    }

    std::string material_type_str = "Undefined";
    if(material_type == MaterialType::CompressibleNeoHookean)
    {
      material_type_str = "CompressibleNeoHookean";
    }
    else if(material_type == MaterialType::IncompressibleNeoHookean)
    {
      material_type_str = "IncompressibleNeoHookean";
    }
    else if(material_type == MaterialType::IncompressibleFibrousTissue)
    {
      material_type_str = "IncompressibleFibrousTissue";
    }
    else if(material_type == MaterialType::StVenantKirchhoff)
    {
      material_type_str = "StVenantKirchhoff";
    }

    this->pcout << "ProblemType           = " << problem_type_str << "\n"
                << "SpatialIntegration    = " << spatial_integration << "\n"
                << "ForceMaterialResidual = " << force_material_residual << "\n"
                << "CacheLevel            = " << cache_level << "\n"
                << "CheckType             = " << check_type << "\n"
                << "MappingStrength       = " << mapping_strength << "\n"
                << "MaterialType          = " << material_type_str << "\n";
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)periodic_face_pairs;
      (void)vector_local_refinements;

      AssertThrow(
        this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
        dealii::ExcMessage(
          "Manifolds might not be applied correctly for TriangulationType::FullyDistributed. "
          "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

      double constexpr left = -1.0, right = 1.0;

      dealii::GridGenerator::subdivided_hyper_cube(tria,
                                                   this->n_subdivisions_1d_hypercube,
                                                   left,
                                                   right);

      if(mapping_strength > 1e-12)
      {
        unsigned int constexpr frequency = 2;
        apply_deformed_cube_manifold(tria, left, right, mapping_strength, frequency);
      }

      tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                  pair;
    typedef typename std::pair<dealii::types::boundary_id, dealii::ComponentMask> pair_mask;

    // zero Dirichlet BCs
    std::vector<bool> mask(dim, true);
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));
    this->boundary_descriptor->dirichlet_bc_initial_acceleration.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(dim)));

    this->boundary_descriptor->dirichlet_bc_component_mask.insert(pair_mask(0, mask));
  }

  void
  set_material_descriptor() final
  {
    typedef std::pair<dealii::types::material_id, std::shared_ptr<MaterialData>> Pair;

    if(material_type == MaterialType::StVenantKirchhoff)
    {
      Type2D const two_dim_type = Type2D::PlaneStress;
      double const nu           = 0.3;
      this->material_descriptor->insert(
        Pair(0, new StVenantKirchhoffData<dim>(material_type, E_modul, nu, two_dim_type)));
    }
    else if(material_type == MaterialType::IncompressibleNeoHookean)
    {
      Type2D const two_dim_type  = Type2D::Undefined;
      double const shear_modulus = 1.0e2;
      double const nu            = 0.49;
      double const bulk_modulus  = shear_modulus * 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu));

      this->material_descriptor->insert(Pair(0,
                                             new IncompressibleNeoHookeanData<dim>(material_type,
                                                                                   shear_modulus,
                                                                                   bulk_modulus,
                                                                                   two_dim_type)));
    }
    else if(material_type == MaterialType::CompressibleNeoHookean)
    {
      Type2D const two_dim_type  = Type2D::Undefined;
      double const shear_modulus = 1.0e2;
      double const nu            = 0.3;
      double const lambda        = shear_modulus * 2.0 * nu / (1.0 - 2.0 * nu);

      this->material_descriptor->insert(Pair(
        0,
        new CompressibleNeoHookeanData<dim>(material_type, shear_modulus, lambda, two_dim_type)));
    }
    else if(material_type == MaterialType::IncompressibleFibrousTissue)
    {
      Type2D const two_dim_type  = Type2D::Undefined;
      double const shear_modulus = 1.0e2;
      double const nu            = 0.49;
      double const bulk_modulus  = shear_modulus * 2.0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu));

      // Parameters corresponding to aortic tissue might be found in
      // [Weisbecker et al., J Mech Behav Biomed Mater 12, 2012] or
      // [Rolf-Pissarczyk et al., Comput Methods Appl Mech Eng 373, 2021].
      // a = 3.62, b = 34.3 for medial tissue lead to the H_ii below,
      // while the k_1 coefficient is scaled relative to the shear modulus
      // (for medial tissue, e.g., 62.1 kPa) used in the other cases here.
      double const fiber_angle_phi_in_degree = 27.47;                          // [deg]
      double const fiber_H_11                = 0.9168;                         // [-]
      double const fiber_H_22                = 0.0759;                         // [-]
      double const fiber_H_33                = 0.0073;                         // [-]
      double const fiber_k_1                 = 1.4e3 / 62.1e3 * shear_modulus; // [Pa]
      double const fiber_k_2                 = 22.1;                           // [-]
      double const fiber_switch_limit        = 0.0;                            // [-]

      this->material_descriptor->insert(
        Pair(0,
             new IncompressibleFibrousTissueData<dim>(material_type,
                                                      shear_modulus,
                                                      bulk_modulus,
                                                      fiber_angle_phi_in_degree,
                                                      fiber_H_11,
                                                      fiber_H_22,
                                                      fiber_H_33,
                                                      fiber_k_1,
                                                      fiber_k_2,
                                                      fiber_switch_limit,
                                                      {},
                                                      {},
                                                      {},
                                                      0.0,
                                                      two_dim_type)));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Material type is not expected in application."));
    }
  }

  void
  set_field_functions() final
  {
    this->field_functions->right_hand_side.reset(
      new dealii::Functions::ConstantFunction<dim>(1.0, dim));
    this->field_functions->initial_displacement.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_velocity.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessor<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    std::shared_ptr<PostProcessor<dim, Number>> pp(
      new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  bool spatial_integration     = false;
  bool force_material_residual = false;

  unsigned int check_type  = 0;
  unsigned int cache_level = 0;

  ProblemType problem_type = ProblemType::Unsteady;

  double weak_damping_coefficient = 0.0;

  MaterialType material_type = MaterialType::Undefined;
  double const E_modul       = 200.0;

  double const density = 0.001;

  double mapping_strength = 0.0;
};

} // namespace Structure

} // namespace ExaDG

#include <exadg/structure/user_interface/implement_get_application.h>

#endif /* STRUCTURE_THROUGHPUT */
