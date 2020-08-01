file(GLOB OpenGLWindow_HDRS "*.h")

file(GLOB OpenGLWindowMac_CPP "tiny_mac*.cpp")
file(GLOB OpenGLWindowMacObjC_CPP "tiny_mac*.m")
file(GLOB OpenGLWindowWin32_CPP "tiny_win32*.cpp")
file(GLOB OpenGLWindowLinux_CPP "tiny_x11*.cpp")

file(GLOB OpenGLWindowCommon_CPP "*.cpp")

list(REMOVE_ITEM OpenGLWindowCommon_CPP ${OpenGLWindowMac_CPP})
list(REMOVE_ITEM OpenGLWindowCommon_CPP ${OpenGLWindowWin32_CPP})
list(REMOVE_ITEM OpenGLWindowCommon_CPP ${OpenGLWindowLinux_CPP})
list(REMOVE_ITEM OpenGLWindowCommon_CPP tiny_x11_opengl_window.cpp)
list(REMOVE_ITEM OpenGLWindowCommon_CPP tiny_mac_opengl_window.cpp)

if(WIN32)
  set(OpenGLWindow_SRCS ./third_party/glad/gl.c ${OpenGLWindowWin32_CPP} ${OpenGLWindowCommon_CPP})
  add_definitions(-DGLEW_STATIC)
else()
  if(APPLE)
    message("APPLE!")
    set(OpenGLWindow_SRCS ${OpenGLWindowMac_CPP} ./third_party/glad/gl.c ${OpenGLWindowMacObjC_CPP} ${OpenGLWindowCommon_CPP})
  else()
    message("LINUX!")
    set(OpenGLWindow_SRCS ${OpenGLWindowLinux_CPP} ${OpenGLWindowCommon_CPP} ./third_party/glad/gl.c)
  endif()
endif(WIN32)

# No Linux detection?
if(NOT WIN32 AND NOT APPLE)
  include_directories(
    ./third_party/glad
    ./third_party/optionalX11
  )
  add_definitions("-DGLEW_STATIC")
  add_definitions("-DGLEW_INIT_OPENGL11_FUNCTIONS=1")
  add_definitions("-DGLEW_DYNAMIC_LOAD_ALL_GLX_FUNCTIONS=1")
  add_definitions("-DDYNAMIC_LOAD_X11_FUNCTIONS=1")

  set(
    OpenGLWindow_SRCS
    ${OpenGLWindowLinux_CPP}
    ${OpenGLWindowCommon_CPP}
    ./third_party/glad/gl.c
    ./third_party/glad/glx.c
  )
endif()

message(${OpenGLWindowCommon_CPP})

if(BUILD_EGL)
  set(OpenGLWindow_SRCS ${OpenGLWindow_SRCS} ./third_party/glad/egl.c)
endif(BUILD_EGL)

set(OpenGLWindow_SRCS ${OpenGLWindow_SRCS})

message("OpenGLWindow_SRCS=${OpenGLWindow_SRCS}")

add_library(opengl_window ${OpenGLWindow_SRCS} ${OpenGLWindow_HDRS})
target_include_directories(
  opengl_window
  PRIVATE
  ./third_party/glad
  ${Stb_INCLUDE_DIR}
)

if(UNIX AND NOT APPLE)
  target_link_libraries(opengl_window ${DL})
elseif(APPLE)
  find_library(COCOA NAMES Cocoa)
  message(${COCOA})
endif()

if(BUILD_SHARED_LIBS)
  if(WIN32 OR APPLE)
    target_link_libraries(opengl_window ${OPENGL_gl_LIBRARY})
  else()
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    find_package(Threads)
    target_link_libraries(opengl_window ${CMAKE_THREAD_LIBS_INIT})
    target_compile_definitions(opengl_window PUBLIC DYNAMIC_LOAD_X11_FUNCTIONS PRIVATE DYNAMIC_LOAD_X11_FUNCTIONS)

  endif()
endif()

if(APPLE)
  find_package(OpenGL)
  target_link_libraries(opengl_window ${COCOA} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
endif()

install(
  TARGETS opengl_window
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib${LIB_SUFFIX}
  ARCHIVE DESTINATION lib${LIB_SUFFIX}
)
