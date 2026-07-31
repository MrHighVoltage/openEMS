/*
* Copyright (C) 2025 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdlib>
#include <iostream>
#include <string>

#include "option_parser.h"
#include "sar_calculation.h"

using namespace std;

int main(int argc, const char* argv[])
{
  cout << " ---------------------------------------------------------------------- " << endl;
  cout << " | SAR calculation for openEMS "                                          << endl;
  cout << " | (C) 2012-2026 Thorsten Liebig <thorsten.liebig@gmx.de>  GPL license"   << endl;
  cout << " ---------------------------------------------------------------------- " << endl;

  OptionDesc desc("Options");
  try {
    string ifile, ofile, method;
    double m_avg = 0;
    double auto_range = 0;
    bool help = false;
    bool debug = false;
    bool export_cube_stats = false;
    bool legacyHDF5 = false;
    bool progress = false;
    unsigned int numThreads = 0;

    desc.addBoolSwitch("help,h", [&](bool value) { help = value; },
                       "print usage message");
    desc.addStringOption("input,i", "", [&](const string& value) { ifile = value; },
                         "pathname to input hdf5 file");
    desc.addStringOption("output,o", "", [&](const string& value) { ofile = value; },
                         "pathname for output hdf5 file");
    desc.addStringOption("method", "SIMPLE", [&](const string& value) { method = value; },
                         "set SAR method: IEEE_C95_3, IEEE_62704, SIMPLE");
    desc.addStringOption("mass,m", "", [&](const string& value) {
      if (!value.empty()) m_avg = strtod(value.c_str(), nullptr);
    }, "averaging mass in g");
    desc.addStringOption("autorange,a", "", [&](const string& value) {
      if (!value.empty()) auto_range = strtod(value.c_str(), nullptr);
    }, "autorange, value limit in dB from max. (>0)");
    desc.addUintOption("numThreads,n", 0, -1,
                       [&](unsigned int value) { numThreads = value; },
                       "number of threads");
    desc.addBoolSwitch("verbose,v", [&](bool value) { debug = value; },
                       "verbose");
    desc.addBoolSwitch("progress,p", [&](bool value) { progress = value; },
                       "show progress");
    desc.addBoolSwitch("export_cube_stats,e",
                       [&](bool value) { export_cube_stats = value; },
                       "Export Cube Statistics");
    desc.addBoolSwitch("legacyHDF5Dumps",
                       [&](bool value) { legacyHDF5 = value; },
                       "Dumping using the legacy HDF5 file format as required for Octave/Matlab import");

    desc.parse(argc, argv);

    if (help) {
      desc.printUsage(cout);
      return 0;
    }

    if (ifile.empty() || ofile.empty()) {
      cerr << "Both --input and --output are required.\n";
      desc.printUsage(cerr);
      return 1;
    }

    SAR_Calculation sar_calc;
    sar_calc.SetDebugLevel(int(debug));
    sar_calc.EnableProgress(progress);
    sar_calc.SetAveragingMass(m_avg / 1000);
    sar_calc.EnableAutoRange(auto_range);
    if (export_cube_stats)
      sar_calc.EnableCubeStats();
    if (!sar_calc.SetAveragingMethod(method, !debug))
      return -1;
    return sar_calc.CalcFromHDF5(ifile, ofile, legacyHDF5, numThreads);
  }
  catch (const exception& e) {
    cerr << e.what() << "\n";
    return 1;
  }
}
