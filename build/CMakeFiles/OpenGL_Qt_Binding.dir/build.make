# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/build

# Include any dependencies generated for this target.
include CMakeFiles/OpenGL_Qt_Binding.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/OpenGL_Qt_Binding.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/OpenGL_Qt_Binding.dir/flags.make

CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o: CMakeFiles/OpenGL_Qt_Binding.dir/flags.make
CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o: ../qt_opengl.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o -c /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/qt_opengl.cpp

CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/qt_opengl.cpp > CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.i

CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/qt_opengl.cpp -o CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.s

CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o.requires:
.PHONY : CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o.requires

CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o.provides: CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o.requires
	$(MAKE) -f CMakeFiles/OpenGL_Qt_Binding.dir/build.make CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o.provides.build
.PHONY : CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o.provides

CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o.provides.build: CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o

# Object files for target OpenGL_Qt_Binding
OpenGL_Qt_Binding_OBJECTS = \
"CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o"

# External object files for target OpenGL_Qt_Binding
OpenGL_Qt_Binding_EXTERNAL_OBJECTS =

OpenGL_Qt_Binding: CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o
OpenGL_Qt_Binding: CMakeFiles/OpenGL_Qt_Binding.dir/build.make
OpenGL_Qt_Binding: /usr/local/lib/libopencv_viz.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_videostab.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_video.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_ts.a
OpenGL_Qt_Binding: /usr/local/lib/libopencv_superres.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_stitching.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_photo.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_ocl.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_objdetect.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_nonfree.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_ml.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_legacy.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_imgproc.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_highgui.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_gpu.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_flann.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_features2d.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_core.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_contrib.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_calib3d.so.2.4.9
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libGLU.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libGL.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libSM.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libICE.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libX11.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libXext.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libGLU.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libGL.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libSM.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libICE.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libX11.so
OpenGL_Qt_Binding: /usr/lib/x86_64-linux-gnu/libXext.so
OpenGL_Qt_Binding: /usr/local/lib/libopencv_nonfree.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_ocl.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_gpu.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_photo.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_objdetect.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_legacy.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_video.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_ml.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_calib3d.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_features2d.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_highgui.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_imgproc.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_flann.so.2.4.9
OpenGL_Qt_Binding: /usr/local/lib/libopencv_core.so.2.4.9
OpenGL_Qt_Binding: CMakeFiles/OpenGL_Qt_Binding.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable OpenGL_Qt_Binding"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OpenGL_Qt_Binding.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/OpenGL_Qt_Binding.dir/build: OpenGL_Qt_Binding
.PHONY : CMakeFiles/OpenGL_Qt_Binding.dir/build

CMakeFiles/OpenGL_Qt_Binding.dir/requires: CMakeFiles/OpenGL_Qt_Binding.dir/qt_opengl.cpp.o.requires
.PHONY : CMakeFiles/OpenGL_Qt_Binding.dir/requires

CMakeFiles/OpenGL_Qt_Binding.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/OpenGL_Qt_Binding.dir/cmake_clean.cmake
.PHONY : CMakeFiles/OpenGL_Qt_Binding.dir/clean

CMakeFiles/OpenGL_Qt_Binding.dir/depend:
	cd /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/build /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/build /home/ryouma/opencv-2.4.9/samples/cpp/Qt_sample/build/CMakeFiles/OpenGL_Qt_Binding.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/OpenGL_Qt_Binding.dir/depend

