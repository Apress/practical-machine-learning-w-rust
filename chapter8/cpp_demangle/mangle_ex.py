# import our rust library, no need for cffi
from cpp_demangle import demangle

# run the demangle function, prints 'mangled::foo(double)'
print(demangle("_ZN7mangled3fooEd"))
