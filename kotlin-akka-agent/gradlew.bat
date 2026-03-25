@rem
@rem Gradle startup script for Windows
@rem

@if "%DEBUG%"=="" @echo off

set DIRNAME=%~dp0
set APP_HOME=%DIRNAME%

set DEFAULT_JVM_OPTS=

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if %ERRORLEVEL% equ 0 goto execute
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
goto fail
:execute
set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar
%JAVA_EXE% %DEFAULT_JVM_OPTS% -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
:end
exit /b %ERRORLEVEL%
:fail
exit /b 1
