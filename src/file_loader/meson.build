project('ccpc_file_loader', 'c', version : '1.0.0', default_options : ['warning_level=3', 'c_std=c17'])

# check for clang
cc = meson.get_compiler('c')

if cc.get_id() != 'clang'
    error(
        'It is *strongly* recommended (2-5 times faster binaries on modern CPUs) to build with Clang (CC=clang ...). You have to delete the build directory when changing Compilers.'.format(cc.get_id())
    )
endif

# get dependencies
py = import('python').find_installation()
np_dep = dependency('numpy')

sources = files('src/py_file_loader.c', 'src/lib.c')

# define module
extension_module = py.extension_module(
    'ccpc_file_loader',
    sources,
    dependencies: [np_dep],
    install: true,
)

test_loader = executable('test_loader', ['src/test.c', 'src/lib.c'], install: true)
