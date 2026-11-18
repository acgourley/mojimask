# The Qt host: shows QML, hosts the video item wgpu draws into, bridges the store, serves the developer hooks.
#   cmake -S native/app -B native/app/build -G Ninja -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build native/app/build
# Dev (default): the host dlopen()s native/target/debug/libmm_app.dylib at start (--lib to choose another), so a Rust
# change needs no C++ relink. -DMM_STATIC=ON links target/release/libmm_app.a instead (release).
cmake_minimum_required(VERSION 3.21)
project(Mojimask LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET 15.0)
set(QT_NO_PRIVATE_MODULE_WARNING ON)
option(MM_STATIC "link the Rust store statically (release) instead of loading the dylib at start" OFF)

find_package(Qt6 6.7 REQUIRED COMPONENTS Gui GuiPrivate Quick QuickControls2 Qml HttpServer Test)
qt_standard_project_setup(REQUIRES 6.7)

qt_add_executable(Mojimask MACOSX_BUNDLE
    src/main.cpp src/mmapi.cpp src/mmapi.h src/store.cpp src/store.h src/videoitem.cpp src/videoitem.h
    src/devtools.cpp src/devtools.h src/platform_mac.mm
    ../crates/mm-app/include/mm_app.h)
target_include_directories(Mojimask PRIVATE ../crates/mm-app/include)

# QML: in dev, loaded from native/app/qml on disk (live reload); the same files are also compiled into the binary
# (qrc:/qml/...) for runs without --dev.
file(GLOB_RECURSE MM_QML RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS qml/*.qml qml/qmldir)
qt_add_resources(Mojimask "qml" PREFIX "/" FILES ${MM_QML})

set_target_properties(Mojimask PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER org.mojimask.app
    MACOSX_BUNDLE_BUNDLE_NAME "Mojimask"
    OUTPUT_NAME Mojimask)

target_link_libraries(Mojimask PRIVATE Qt6::Gui Qt6::GuiPrivate Qt6::Quick Qt6::QuickControls2 Qt6::Qml
    Qt6::HttpServer Qt6::Test "-framework AppKit" "-framework QuartzCore" "-framework Metal")
if(MM_STATIC)
    target_compile_definitions(Mojimask PRIVATE MM_STATIC=1)
    target_link_libraries(Mojimask PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../target/release/libmm_app.a"
        "-framework CoreVideo" "-framework CoreMedia" "-framework IOSurface" "-framework CoreFoundation"
        "-framework Foundation" "-framework CoreGraphics"
        # the store's own: AVFoundation (photos, the depth movie), Accelerate (numpy's BLAS/LAPACK), CoreML (ONNX
        # Runtime's provider), CoreText (emoji), VideoToolbox (display colour, mm-media's vt.rs), CoreImage, MediaToolbox and
        # CoreAudio (GStreamer's applemedia)
        "-framework AVFoundation" "-framework Accelerate" "-framework CoreML" "-framework CoreText" "-framework CoreImage"
        "-framework VideoToolbox" "-framework MediaToolbox" "-framework CoreAudio"
        iconv objc
        "-L/opt/homebrew/lib" gstreamer-1.0 gstapp-1.0 gstvideo-1.0 gstbase-1.0 gio-2.0 gobject-2.0 glib-2.0 intl)
endif()
set_source_files_properties(src/platform_mac.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
