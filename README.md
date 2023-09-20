Lightweight vulkan wrapper and scene graph. Supports loading the following scene formats out of the box:
* Environment maps (\*.hdr, \*.exr)
* GLTF scenes (\*.glb, \*.gltf)
* NVDB or Mitsuba volumes (\*.nvdb, \*.vol)

# Optional dependencies
Optional dependences (searched via find_package in CMake) are:
- 'assimp' for loading \*.fbx scenes
- 'OpenVDB' for loading \*.vdb volumes

# Command line arguments
* --instanceExtension=`string`
* --deviceExtension=`string`
* --validationLayer=`string`
* --debugMessenger
* --noPipelineCache
* --shaderKernelPath=`path`
* --shaderInclude=`path`
* --font=`path,float`
* --width=`int`
* --height=`int`
* --presentMode=`string`
* --scene=`path`

## Required arguments
* --shaderKernelPath=${Stratum2_Dir}/src/Shaders/kernels
* --shaderInclude=${Stratum2_Dir}/extern
* --shaderInclude=${Stratum2_Dir}/src/Shaders
* --debugMessenger

## Recommended arguments
* --font=DroidSans.ttf,16
* --width=1920
* --height=1080
* --presentMode=Immediate