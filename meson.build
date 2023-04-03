project('linear_algebra', 'cpp')
if (get_option('eigen3') == true)
  message('Use Eigen3 for spline interpolation')
  add_global_arguments('-DUSE_EIGEN3', language : 'cpp')
  eigen_dep = dependency('eigen3', 'cmake')
endif
fmt_dep = dependency('fmt')
boost_po_dep = dependency('boost', modules : ['program_options'])
src_base = ['Matrix.cpp', 'Helper.cpp']

interpolation_src = [src_base, 'Spline.cpp']
interpolation_deps = [fmt_dep]
interpolation_includes = []
if (get_option('eigen3') == true)
  interpolation_deps += [eigen_dep]
  interpolation_includes += [include_directories('/usr/include/eigen3')]
endif

# tests
src_testCholesky = src_base + ['tests/testCholesky.cpp']
executable('testCholesky', src_testCholesky, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testDeterminant = src_base + ['tests/testDeterminant.cpp']
executable('testDeterminant', src_testDeterminant, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testEigensystem = src_base + ['tests/testEigensystem.cpp']
executable('testEigensystem', src_testEigensystem, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testGramSchmidt = src_base + ['tests/testGramSchmidt.cpp']
executable('testGramSchmidt', src_testGramSchmidt, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testHouseholderQR = src_base + ['tests/testHouseholderQR.cpp']
executable('testHouseholderQR', src_testHouseholderQR, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testInterpolation = interpolation_src + ['tests/testInterpolation.cpp']
executable('testInterpolation', src_testInterpolation, cpp_args : ['-std=c++17'], dependencies: [interpolation_deps], include_directories: interpolation_includes)
src_testInverse = src_base + ['tests/testInverse.cpp']
executable('testInverse', src_testInverse, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testLinearSolver = src_base + ['tests/testLinearSolver.cpp']
executable('testLinearSolver', src_testLinearSolver, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testLU = src_base + ['tests/testLU.cpp']
executable('testLU', src_testLU, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testReparametrization = interpolation_src + ['tests/testReparametrization.cpp', 'Reparametrization.cpp']
executable('testReparametrization', src_testReparametrization, cpp_args : ['-std=c++17'], dependencies: [interpolation_deps], include_directories: interpolation_includes)
src_testSVD = src_base + ['tests/testSVD.cpp']
executable('testSVD', src_testSVD, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testQREigen = src_base + ['tests/testQREigen.cpp']
executable('testQREigen', src_testQREigen, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])

# programs
src_reparam = interpolation_src + ['programs/reparametrization/main.cpp', 'Reparametrization.cpp']
executable('reparametrization', src_reparam, cpp_args : ['-std=c++17'], dependencies: [interpolation_deps, boost_po_dep], include_directories: interpolation_includes)
