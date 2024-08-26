@echo off

set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing node...

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    "%python_exec%" setup.py"
) else (
    echo Installing with system Python
    setup.py"
)

pause