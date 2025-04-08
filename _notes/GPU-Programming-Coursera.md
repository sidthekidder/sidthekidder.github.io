---
layout: note
name: Intro to GPU Programming 
type: mooc
date: March 31, 2025
---

[Course 1 of the [GPU programming](https://www.coursera.org/specializations/gpu-programming) specialization on coursera by John Hopkins University]


**Introduction to Parallel Programming with GPUs** - 

Modern CPUs have 4-8 cores with 8-16 threads each. Each core is like a railway track and each thread is a train. Trains can constantly switch on or off the railway tracks. 

The scheduler and scheduling algorithms move programs/tasks between active and inactive states, move them between cores and move data around caches. 
States change due to cache misses, and schedules doesn't want to waste CPU cycles waiting for data or instructions, so switches to something else.
Memory caches are hierarchical, and follow the principle of physically closer memory is more performant.

Issues with multi-threaded programs: race conditions, deadlock, live lock etc

Famous concurrent programming challenges/algorithms:
- dining philosophers (deadlock)
- producer-consumer pattern
- sleeping barber (over/under-utilization)

Concurrent programming patterns:
- divide & conquery
- map-reduce
- repository
- pipelines & workflow DAGs
- recursion

Serial search: 
- search an array 0 to end for an element (linear, simple)
- binary search - divide left/right, check midpoint, recurse on either side (logarithmic, cost of sorting data)
Parallel search:
- for n threads, slice data into n uniform subsets. each thread searches the subset, kills other threads if found (scales on n)

Flynn's taxonomy:
- SISD: single instruction single data
- SIMD: single instruction multiple data (many GPU programs, same logic on large amounts of data)
- MISD: multiple instruction single data (most CPU programs / single object worked on in multiple steps)
- MIMD: multiple instruction multiple data (e.g multiple image filters on multiple image pixels/video frames etc)


3 main parallel programming libs in Python:
- thread/threading
- asyncio (async def func() / create_task(func1()) / await task / gather(func1(), func2()) / )
- multiprocessing (spawn/fork/process start/join/queue/pipe/lock acquire/release / pool)

C++ parallel programming syntax:
- std::thread
    - create/join/detach thread
- std::mutex
    - lock / try_lock / unlock
- std::atomic
    - guaranteed not to cause race conditions. 
- std::future
    - promise / future / async wait/async get


**Nvidia vs Integrated GPUs**

Integrated GPUs are built into CPU and share system memory. Get less hot, use less power, are less powerful.

Nvidia GPUs are vendor-specific implementations of dedicated GPUs. Alternatives are AMD GPUs. Mac uses AMD and not Nvidia due to apple's lack of support for OpenCL/CUDA. 
OpenCL is a common open source heterogenous platform. OpenACC makes it easy to use accelerators including CPUs and GPUs. Nvidia/AMD profiler help in identifying memory leaks etc.

nvcc to compile .cu files.

Cuda Software Layers:
Can use CUDA runtime or CUDA driver APIs. Or CUDA Libraries e.g cuBLAS, cuFFT.

nvcc help command - profile/debug/specify arch/code options.
cudaMalloc/cudaMemcpy/cudaMemcpyHostToDevice/cudaMemcpyDeviceToHost/cudaFree


CUDA Compilation & Execution syntax

CUDA keywords:
- execution/invocation keywords (how and where)
    - device / global / host
- threads/blocks/grids (division of computation)
    - blocksPerGrid , threadsPerBlock, sizeOfSharedMemory, cudaStream
- memory (concerning host/device memory)
    - const/shared/device


When writing CUDA code for GPU, consider memory optimization and importantly be careful of branching. Branching may cause entire warps to halve in efficiency. 
Profile/test code. Generate performance metrics to compare against sequential implementations. 