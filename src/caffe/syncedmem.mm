#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    SyncedMemory::SyncedMemory() : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED), own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {

    }
    
    SyncedMemory::SyncedMemory(size_t size) : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED), own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {

    }
    
    SyncedMemory::~SyncedMemory() {
        check_device();
        if (cpu_ptr_ && own_cpu_data_) {
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
    }
    
    
    void SyncedMemory::default_reference() {
        refer_num = 0;
    }
    
    void SyncedMemory::increase_reference() {
        refer_num++;
    }
    
    void SyncedMemory::decrease_reference() {
        refer_num--;
        if (refer_num == 0) {
            zhihan_release();
        }
    }
    
    void SyncedMemory::zhihan_release() {
        if (cpu_ptr_ && own_cpu_data_) {
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        
        cpu_ptr_ = NULL;
        own_cpu_data_ = false;
        
        head_ = UNINITIALIZED;
    }
    

    
    inline void SyncedMemory::to_cpu() {
        check_device();
        switch (head_) {
            case UNINITIALIZED:
                set_align_size(size_);
                CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, get_align_size());
                caffe_memset(size_, 0, cpu_ptr_);
                head_ = HEAD_AT_CPU;
                own_cpu_data_ = true;
                break;
            case HEAD_AT_GPU:
                NO_GPU;
                break;
            case HEAD_AT_CPU:
            case SYNCED:
                break;
        }
    }
    
    inline void SyncedMemory::to_gpu() {
        to_cpu();
    }
    
    const void* SyncedMemory::cpu_data() {
        check_device();
        to_cpu();
        return (const void*)cpu_ptr_;
    }
    
    void SyncedMemory::set_cpu_data(void* data) {
        check_device();
        CHECK(data);
        if (own_cpu_data_) {
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false;
    }
    
    const void* SyncedMemory::gpu_data() {
        check_device();
        to_cpu();
        return (const void*)cpu_ptr_;
    }
    
    void SyncedMemory::set_gpu_data(void* data) {
        set_cpu_data(data);
    }
    
    void* SyncedMemory::mutable_cpu_data() {
        check_device();
        to_cpu();
        head_ = HEAD_AT_CPU;
        return cpu_ptr_;
    }
    
    void* SyncedMemory::mutable_gpu_data() {
        check_device();
        to_cpu();
        head_ = HEAD_AT_CPU;
        return cpu_ptr_;
    }
    
    
    void SyncedMemory::check_device() {
        
    }
    
}  // namespace caffe

