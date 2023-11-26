Abstract Compute Experiment
===========================

Building
--------

### Linux

Install prerequisites

```
    sudo apt install build-essential cmake git

    # Install vulkan-sdk
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
    sudo apt update
    sudo apt install vulkan-sdk

```

Build release config

```
    mkdir -p ./build/release
    cd ./build/release
    cmake -D CMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../..
    make -j8
```

