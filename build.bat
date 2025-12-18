@echo off

cl.exe /nologo /c /LD /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include" foo.c
link.exe /nologo /DLL foo.obj /OUT:foo.dll /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" cuda.lib cudart.lib kernel32.lib
copy foo.dll z:\
