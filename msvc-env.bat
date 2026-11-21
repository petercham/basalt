@echo off
rem Loads the MSVC + CUDA 13 environment, then runs the command given to it.
set "PATH=C:\Work\cuda130\Library\bin;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
%*
