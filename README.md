# To Run
Make sure you've installed glslang, the vulkan sdk and spirv-tools. I installed
glslang through my system package manager, you might be using a different approach
in which case you'll need to edit the CMakeLists (see the [VkFFT documentation](https://github.com/DTolm/VkFFT/tree/master)).

With all dependencies in place, simply run
```
cmake -B build
cmake --build build
```
