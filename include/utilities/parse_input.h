#ifndef STRUCTURAL_FUNCTIONALITIES_PARSE_INPUT
#define STRUCTURAL_FUNCTIONALITIES_PARSE_INPUT

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

inline void
parse_input(std::string const  parameter_file,
            ParameterHandler & prm,
            bool const         skip_undefined,
            bool const         assert_all_entries_are_found)
{
  // parse file
  std::filebuf fb;
  if(fb.open(parameter_file, std::ios::in))
  {
    std::istream is(&fb);
    std::string  file_ending = parameter_file.substr(parameter_file.find_last_of(".") + 1);
    if(file_ending == "prm")
      prm.parse_input(is, parameter_file, "", skip_undefined);
    else if(file_ending == "xml")
      prm.parse_input_from_xml(is, skip_undefined);
    else if(file_ending == "json")
      prm.parse_input_from_json(is, skip_undefined);
    else
      AssertThrow(false,
                  ExcMessage("Unknown input file. Supported types are .prm, .xml, and .json."));

    fb.close();

    // TODO
    (void)assert_all_entries_are_found;
//    if(assert_all_entries_are_found)
//      prm.check_all_found();
  }
}

#endif
