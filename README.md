Lightweight vulkan wrapper and scene graph. Supports loading the following scene formats out of the box:
* Environment maps (\*.hdr, \*.exr)
* GLTF scenes (\*.glb, \*.gltf)
* NVDB or Mitsuba volumes (\*.nvdb, \*.vol)

# Optional dependencies
Optional dependences (searched via find_package in CMake) are:
- 'assimp' for loading \*.fbx scenes
- 'OpenVDB' for loading \*.vdb volumes

# Command line arguments
* --instance-extension=`string`
* --device-extension=`string`
* --validation-layer=`string`
* --debug-messenger
* --no-pipeline-cache
* --shader-kernel-path=`path`
* --shader-include=`path`
* --font=`path,float`
* --gui-scale=`float`
* --width=`int`
* --height=`int`
* --min-images=`int`
* --scene=`path`
* --renderer=`string`
* --surface-format-srgb
* --no-gamma-correct

## Required arguments
* --shader-kernel-path=${Stratum2_Dir}/src/Shaders/kernels
* --shader-include=${Stratum2_Dir}/extern
* --shader-include=${Stratum2_Dir}/src/Shaders
* --debug-messenger