#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "miner.h"
#include "pattern.h"
#include "typing.h"

namespace py = pybind11;

PYBIND11_MODULE(petsc_miner, m) {
  py::class_<Pattern>(m, "Pattern")
      .def(py::init<std::vector<int>, Projection, Candidates, int>(),
           py::arg("pattern"), py::arg("projection"), py::arg("candidates"),
           py::arg("support"))
      .def_readwrite("pattern", &Pattern::pattern)
      .def_readwrite("projection", &Pattern::projection)
      .def_readwrite("candidates", &Pattern::candidates)
      .def_readwrite("support", &Pattern::support)
      .def_readwrite("coef", &Pattern::coef)
      .def("__repr__", [](Pattern &m) {
        std::string s = "Pattern([";
        for (size_t i = 0; i < m.pattern.size(); ++i) {
          s += std::to_string(m.pattern[i]);
          if (i < m.pattern.size() - 1) {
            s += ", ";
          }
        }
        s += "])";
        return s;
      });

  py::class_<PatternMiner>(m, "PatternMiner")
      .def(py::init<int, int, int, double, int, bool>(), py::arg("alpha"),
           py::arg("min_size"), py::arg("max_size"), py::arg("duration"),
           py::arg("k"), py::arg("sort_alpha"))
      .def("mine", &PatternMiner::mine)
      .def("project", &PatternMiner::project)
      .def("project_soft", &PatternMiner::project_soft);
}