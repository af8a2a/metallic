#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <iostream>

int main() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported on this device." << std::endl;
        return 1;
    }

    std::cout << "Metal device: " << device->name()->utf8String() << std::endl;

    pool->release();
    return 0;
}
