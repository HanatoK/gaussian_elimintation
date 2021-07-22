project('linear_algebra', 'cpp')
fmt_dep = dependency('fmt')
src = ['Matrix.cpp', 'Helper.cpp', 'Reparametrization.cpp', 'Spline.cpp', 'main.cpp']
executable('linear_algebra_tests', src, cpp_args : ['-std=c++17'], dependencies: [fmt_dep])

boost_po_dep = dependency('boost', modules : ['program_options'])
src_reparam = ['Matrix.cpp', 'Helper.cpp', 'Reparametrization.cpp', 'Spline.cpp', 'programs/reparametrization/main.cpp']
executable('reparametrization', src_reparam, cpp_args : ['-std=c++17'], dependencies: [fmt_dep, boost_po_dep])
