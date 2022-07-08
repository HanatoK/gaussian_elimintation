project('linear_algebra', 'cpp')
fmt_dep = dependency('fmt')
boost_po_dep = dependency('boost', modules : ['program_options'])
src_base = ['Matrix.cpp', 'Helper.cpp', 'Reparametrization.cpp', 'Spline.cpp']

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
src_testInterpolation = src_base + ['tests/testInterpolation.cpp']
executable('testInterpolation', src_testInterpolation, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testInverse = src_base + ['tests/testInverse.cpp']
executable('testInverse', src_testInverse, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testLinearSolver = src_base + ['tests/testLinearSolver.cpp']
executable('testLinearSolver', src_testLinearSolver, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testLU = src_base + ['tests/testLU.cpp']
executable('testLU', src_testLU, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testReparametrization = src_base + ['tests/testReparametrization.cpp']
executable('testReparametrization', src_testReparametrization, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testSVD = src_base + ['tests/testSVD.cpp']
executable('testSVD', src_testSVD, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])
src_testQREigen = src_base + ['tests/testQREigen.cpp']
executable('testQREigen', src_testQREigen, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])

# programs
src_reparam = src_base + ['programs/reparametrization/main.cpp']
executable('reparametrization', src_reparam, cpp_args : ['-std=c++17'], dependencies: [fmt_dep, boost_po_dep])
