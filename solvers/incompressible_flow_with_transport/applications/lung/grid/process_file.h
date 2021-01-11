#ifndef LUNG_PROCESS_FILE
#define LUNG_PROCESS_FILE

#include "lung_util.h"

namespace ExaDG
{
using namespace dealii;

void
load_files(std::vector<std::string>          files,
           std::vector<Point<3>> &           points,
           std::vector<CellData<1>> &        cells,
           std::vector<CellAdditionalInfo> & cells_additional_data)
{
  double const mm_to_m = 0.001;

  for(auto filename : files)
  {
    std::ifstream infile(filename);

    const unsigned int start_point = points.size();

    std::string line;

    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);

    while(std::getline(infile, line))
    {
      std::istringstream iss(line);
      std::string        a, b;
      int                vertex;
      double             x, y, z;
      if(!(iss >> a >> vertex >> b >> x >> y >> z))
      {
        break;
      } // error

      points.push_back({x * mm_to_m, y * mm_to_m, z * mm_to_m}); // in meter
    }

    while(std::getline(infile, line))
    {
      std::istringstream iss(line);
      std::string        a, b;
      int                node, dnode;
      if(!(iss >> a >> node >> b >> dnode))
      {
        break;
      } // error
    }

    while(std::getline(infile, line))
    {
      std::istringstream iss(line);

      int         element;               //			1
      std::string red;                   //			RED_AIRWAY
      std::string line;                  //			LINE2
      int         node1;                 //			1
      int         node2;                 //			2
      std::string mat;                   //			MAT
      int         matn;                  //			1
      std::string elementsolvingtype;    //	ElemSolvingType
      std::string linear;                //			Linear
      std::string type;                  //			TYPE
      std::string resistive;             //		Resistive
      std::string resistance;            //		Resistance
      std::string generation_dependent;  //	Generation_Dependent_Pedley
      std::string power;                 //			PowerOfVelocityProfile
      int         power_val;             //		2
      std::string wall_elasticity;       //		WallElasticity
      double      wall_elasticity_val;   //	0.0
      std::string poisson_ratio;         //		PoissonsRatio
      double      poisson_ratio_val;     //	0.0
      std::string viscousts;             //		ViscousTs
      double      viscousts_val;         //		0.0
      std::string viscousphaseshift;     //	ViscousPhaseShift
      double      viscousphaseshift_val; //	0.0
      std::string wallthickness;         //		WallThickness
      double      wallthickness_val;     //	0.0
      std::string area;                  //			Area
      double      area_val;              //		35.9601174709
      std::string generation;            //		Generation
      int         generation_val;        //		3

      if(!(iss >> element >> red >> line >> node1 >> node2 >> mat >> matn >> elementsolvingtype >>
           linear >> type >> resistive >> resistance >> generation_dependent >> power >>
           power_val >> wall_elasticity >> wall_elasticity_val >> poisson_ratio >>
           poisson_ratio_val >> viscousts >> viscousts_val >> viscousphaseshift >>
           viscousphaseshift_val >> wallthickness >> wallthickness_val >> area >> area_val >>
           generation >> generation_val))
      {
        break;
      } // error
      CellData<1> cell;
      cell.vertices[0] = start_point + node1 - 1;
      cell.vertices[1] = start_point + node2 - 1;
      cells.push_back(cell);

      CellAdditionalInfo cai;
      cai.generation = generation_val;
      cai.radius     = std::pow(area_val / numbers::PI, 0.5) * mm_to_m; // in meter
      cells_additional_data.push_back(cai);
    }
  }
}

void
load_new_files(const std::vector<std::string> &  files,
               std::vector<Point<3>> &           points,
               std::vector<CellData<1>> &        cells,
               std::vector<CellAdditionalInfo> & cells_additional_data,
               unsigned int                      generations)
{
  double const              mm_to_m = 0.001;
  std::vector<unsigned int> point_id;

  for(const auto & filename : files)
  {
    std::ifstream infile(filename);

    const int max_point_nr = points.size();

    std::string line;

    std::getline(infile, line);
    std::getline(infile, line);

    while(std::getline(infile, line))
    {
      std::istringstream iss(line);
      std::string        a, b;
      std::string        str_vertex;
      int                vertex;
      double             x, y, z;
      if(!(iss >> a >> str_vertex >> b >> x >> y >> z))
      {
        break;
      } // error

      vertex = std::stoi(str_vertex, nullptr, 2);

      point_id.push_back(vertex);

      if(vertex < max_point_nr)
        points[vertex] = {x * mm_to_m, y * mm_to_m, z * mm_to_m};

      // points.emplace_back(x * mm_to_m, y * mm_to_m, z * mm_to_m); // in meter
    }

    while(std::getline(infile, line))
    {
      std::istringstream iss(line);

      std::string element;        //	LINE
      std::string str_element_nr; //	1
      int         element_nr;
      std::string nodes;     //	NODES
      std::string str_node1; //	1
      int         node1;
      std::string str_node2; //	2
      int         node2;
      std::string Radius;     //	RADIUS
      double      radius;     //	1.0
      std::string Generation; //	GENERATION
      int         generation;


      if(!(iss >> element >> str_element_nr >> nodes >> str_node1 >> str_node2 >> Radius >>
           radius >> Generation >> generation))
      {
        break;
      } // error

      // binary to decimal
      element_nr = std::stoi(str_element_nr, nullptr, 2);
      node1      = std::stoi(str_node1, nullptr, 2);
      node2      = std::stoi(str_node2, nullptr, 2);

      if((unsigned int)generation <= generations)
      {
        CellData<1> cell;
        cell.vertices[0] = node1;
        cell.vertices[1] = node2;
        cells.push_back(cell);

        CellAdditionalInfo cai;
        cai.cell_id    = element_nr;
        cai.generation = generation;
        cai.radius     = radius * mm_to_m; // in meter
        cells_additional_data.push_back(cai);
      }
    }
  }
}

void
call_METIS_MeshToDual(int *  ne,
                      int *  nn,
                      int *  eptr,
                      int *  eind,
                      int *  ncommon,
                      int *  numflag,
                      int ** xadj,
                      int ** adjency)
{
#ifdef DEBUG_INFO
  printf("ne      = %4d\n", *ne);
  printf("nn      = %4d\n", *nn);
  printf("ncommon = %4d\n", *ncommon);
  printf("numflag = %4d\n", *numflag);

  printf("eind\n");
  for(int i = 0; i < *ne; i++)
  {
    for(int j = eptr[i]; j < eptr[i + 1]; j++)
      printf("%4d ", eind[j]);
    printf("\n");
  }
#endif

#ifdef DEAL_II_WITH_METIS
  METIS_MeshToDual(ne, nn, eptr, eind, ncommon, numflag, xadj, adjency);
#else
  (void)ne;
  (void)nn;
  (void)eptr;
  (void)eind;
  (void)ncommon;
  (void)numflag;
  (void)xadj;
  (void)adjency;
  AssertThrow(false, ExcMessage("Not comiled with METIS!"));
#endif

#ifdef DEBUG_INFO
  printf("xadj\n");
  for(int i = 0; i < *ne; i++)
  {
    for(int j = (*xadj)[i]; j < (*xadj)[i + 1]; j++)
      printf("%4d ", (*adjency)[j]);
    printf("\n");
  }
#endif
}

void create_dual_graph(std::vector<Point<3>> &    points,
                       std::vector<CellData<1>> & cells,
                       int *&                     xadj_vertex,
                       int *&                     adjncy_vertex)
{
  int ne      = cells.size();
  int nn      = points.size();
  int ncommon = 1;
  int numflag = 0;

  std::vector<int> eind;
  std::vector<int> eptr;

  eptr.push_back(0);
  for(auto cell : cells)
  {
    for(int i = 0; i < 2; i++)
      eind.push_back(cell.vertices[i]);
    eptr.push_back(eind.size());
  }

  call_METIS_MeshToDual(
    &ne, &nn, &eptr[0], &eind[0], &ncommon, &numflag, &xadj_vertex, &adjncy_vertex);
}

} // namespace ExaDG

#endif
