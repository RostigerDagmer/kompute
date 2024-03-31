use ash::{vk, Instance, Entry};
use log::debug;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;
use ash::Device;

pub struct Tensor {
    physical_device: Arc<vk::PhysicalDevice>,
    device: Arc<Device>,
    data: *mut c_void,
    element_total_count: u32,
    element_memory_size: u32,
    data_type: TensorDataTypes,
    tensor_type: TensorTypes,
    primary_buffer: Option<vk::Buffer>,
    primary_memory: Option<vk::DeviceMemory>,
    staging_buffer: Option<vk::Buffer>,
    staging_memory: Option<vk::DeviceMemory>,
    raw_data: *mut c_void,
    size: u32,
    data_type_memory_size: u32,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("physical_device", &self.physical_device)
            .field("data", &self.data)
            .field("element_total_count", &self.element_total_count)
            .field("element_memory_size", &self.element_memory_size)
            .field("data_type", &self.data_type)
            .field("tensor_type", &self.tensor_type)
            .field("primary_buffer", &self.primary_buffer)
            .field("primary_memory", &self.primary_memory)
            .field("staging_buffer", &self.staging_buffer)
            .field("staging_memory", &self.staging_memory)
            .field("raw_data", &self.raw_data)
            .field("size", &self.size)
            .field("data_type_memory_size", &self.data_type_memory_size)
            .finish()
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TensorDataTypes {
    Bool,
    Int,
    UnsignedInt,
    Float,
    Double,
    Unknown,
}

impl<T: TensorData> From<T> for TensorDataTypes {
    fn from(_: T) -> Self {
        match std::any::type_name::<T>() {
            "bool" => TensorDataTypes::Bool,
            "i8" => TensorDataTypes::Int,
            "u8" => TensorDataTypes::UnsignedInt,
            "i16" => TensorDataTypes::Int,
            "u16" => TensorDataTypes::UnsignedInt,
            "i32" => TensorDataTypes::Int,
            "u32" => TensorDataTypes::UnsignedInt,
            "i64" => TensorDataTypes::Int,
            "u64" => TensorDataTypes::UnsignedInt,
            "f32" => TensorDataTypes::Float,
            "f64" => TensorDataTypes::Double,
            _ => TensorDataTypes::Unknown,
        }
    }
}

impl <T: TensorData> From<&[T]> for TensorDataTypes {
    fn from(_: &[T]) -> Self {
        match std::any::type_name::<T>() {
            "bool" => TensorDataTypes::Bool,
            "i8" => TensorDataTypes::Int,
            "u8" => TensorDataTypes::UnsignedInt,
            "i16" => TensorDataTypes::Int,
            "u16" => TensorDataTypes::UnsignedInt,
            "i32" => TensorDataTypes::Int,
            "u32" => TensorDataTypes::UnsignedInt,
            "i64" => TensorDataTypes::Int,
            "u64" => TensorDataTypes::UnsignedInt,
            "f32" => TensorDataTypes::Float,
            "f64" => TensorDataTypes::Double,
            _ => TensorDataTypes::Unknown,
        }
    }
}


unsafe fn to_bytes<'a, T>(d: &T) -> &'a[u8] {
    std::slice::from_raw_parts(
        d as *const T as *const u8,
        std::mem::size_of::<i32>(),
    )
}

pub trait TensorData {
    fn boxed(&self) -> Box<dyn TensorData>;
    fn bytes(&self) -> &[u8];
}

impl TensorData for f32 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for f64 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for u8 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for u16 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for u32 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for u64 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for i8 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for i16 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}

impl TensorData for i32 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for i64 {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        unsafe { to_bytes(self) }
    }
}
impl TensorData for bool {
    fn boxed(&self) -> Box<dyn TensorData> {
        Box::new(*self)
    }
    fn bytes(&self) -> &[u8] {
        if *self {
            &[1]
        } else {
            &[0]
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TensorTypes {
    Device,
    Host,
    Storage,
    Unknown,
}

impl Tensor {
    pub fn new(
        physical_device: Arc<vk::PhysicalDevice>,
        device: Arc<Device>,
        data: *mut c_void,
        element_total_count: u32,
        element_memory_size: u32,
        data_type: TensorDataTypes,
        tensor_type: TensorTypes,
    ) -> Self {
        debug!(
            "Kompute Tensor constructor data length: {}, and type: {:?}",
            element_total_count, tensor_type
        );
        
        Self {
            physical_device,
            device,
            data,
            element_total_count,
            element_memory_size,
            data_type,
            tensor_type,
            primary_buffer: None,
            primary_memory: None,
            staging_buffer: None,
            staging_memory: None,
            raw_data: ptr::null_mut(),
            size: element_total_count,
            data_type_memory_size: element_memory_size,
        }
    }

    pub fn tensor_type(&self) -> TensorTypes {
        self.tensor_type
    }

    pub fn is_init(&self) -> bool {
        debug!("Kompute Tensor is_init primary_buffer: {:?}, primary_memory: {:?}, raw_data: {:?}", self.primary_buffer, self.primary_memory, self.raw_data);
        self.primary_buffer.is_some()
            && self.primary_memory.is_some()
            && !self.raw_data.is_null()
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn data_type_memory_size(&self) -> u32 {
        self.data_type_memory_size
    }

    pub fn memory_size(&self) -> u32 {
        debug!("Kompute Tensor data type memory size: {}", self.data_type_memory_size);
        debug!("Kompute Tensor memory size: {}", self.size * self.data_type_memory_size);
        self.size * self.data_type_memory_size
    }

    pub fn data_type(&self) -> TensorDataTypes {
        self.data_type
    }

    pub fn raw_data(&self) -> *mut c_void {
        self.raw_data
    }

    pub fn set_raw_data(&self, data: *const c_void) {
        unsafe {
            std::ptr::copy_nonoverlapping(data, self.raw_data, self.memory_size() as usize);
        }
    }

    pub fn rebuild(&mut self, entry: Arc<Entry>, instance: Arc<Instance>, data: *mut c_void, element_total_count: u32, element_memory_size: u32) {
        debug!("Kompute Tensor rebuilding with size {}", element_total_count);

        self.size = element_total_count;
        self.data_type_memory_size = element_memory_size;

        if self.primary_buffer.is_some() || self.primary_memory.is_some() {
            debug!("Kompute Tensor destroying existing resources before rebuild");
            self.destroy();        
        }

        self.allocate_memory_create_gpu_resources(entry.clone(), instance.clone());

        if self.tensor_type() != TensorTypes::Storage {
            self.map_raw_data();
            unsafe {
                std::ptr::copy_nonoverlapping(data, self.raw_data, self.memory_size() as usize);
            }
        }
    }

    pub fn vector<T: TensorData>(&self) -> Vec<T> {
        // unsafe cast raw data to Vec<T>
        debug!("Kompute Tensor vector data type: {:?}", self.raw_data);
        unsafe { Vec::from_raw_parts(self.raw_data as *mut T, self.size as usize, self.size as usize) }
    }

    pub fn map_raw_data(&mut self) {
        debug!("Kompute Tensor mapping data from host buffer");

        let host_visible_memory = match self.tensor_type {
            TensorTypes::Host => self.primary_memory.as_ref(),
            TensorTypes::Device => self.staging_memory.as_ref(),
            _ => {
                debug!(
                    "Kompute Tensor mapping data not supported on {:?} tensor",
                    self.tensor_type
                );
                return;
            }
        };

        if let Some(memory) = host_visible_memory {
            let buffer_size = self.memory_size() as vk::DeviceSize;
            let result = unsafe {
                self.device.map_memory(
                    *memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
            };
            if let Ok(mapped_memory) = result {
                self.raw_data = mapped_memory;
            } else {
                debug!("Failed to map memory: {:?}", result.err());
            }
        }
    }

    pub fn unmap_raw_data(&mut self) {
        debug!("Kompute Tensor unmapping data from host buffer");

        let host_visible_memory = match self.tensor_type {
            TensorTypes::Host => self.primary_memory.as_ref(),
            TensorTypes::Device => self.staging_memory.as_ref(),
            _ => {
                debug!(
                    "Kompute Tensor unmapping data not supported on {:?} tensor",
                    self.tensor_type
                );
                return;
            }
        };

        if let Some(memory) = host_visible_memory {
            let buffer_size = self.memory_size() as vk::DeviceSize;
            let mapped_range = vk::MappedMemoryRange::builder()
                .memory(*memory)
                .offset(0)
                .size(buffer_size)
                .build();
            unsafe {
                self.device.flush_mapped_memory_ranges(&[mapped_range]).unwrap();
                self.device.unmap_memory(*memory);
            }
        }
    }

    // TODO: Implement the remaining methods of the Tensor struct

    pub fn record_copy_from(
        &self,
        command_buffer: vk::CommandBuffer,
        copy_from_tensor: Arc<Tensor>,
    ) {
        let copy_region = vk::BufferCopy::builder()
            .size(self.memory_size() as vk::DeviceSize)
            .build();

        unsafe {
            self.device.cmd_copy_buffer(
                command_buffer,
                copy_from_tensor.primary_buffer.expect("Primary buffer not initialized"),
                self.primary_buffer.expect("Primary buffer not initialized"),
                &[copy_region],
            );
        }
    }

    pub fn record_copy_from_staging_to_device(&self, command_buffer: vk::CommandBuffer) {
        let buffer_size = self.memory_size() as vk::DeviceSize;
        let copy_region = vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(buffer_size)
            .build();

        debug!("Kompute Tensor copying data size {}.", buffer_size);

        self.record_copy_buffer(
            command_buffer,
            self.staging_buffer.expect("Staging buffer not initialized"),
            self.primary_buffer.expect("Primary buffer not initialized"),
            buffer_size,
            copy_region,
        );
    }

    pub fn record_copy_from_device_to_staging(&self, command_buffer: vk::CommandBuffer) {
        let buffer_size = self.memory_size() as vk::DeviceSize;
        let copy_region = vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(buffer_size)
            .build();

        debug!("Kompute Tensor copying data size {}.", buffer_size);
        self.record_copy_buffer(
            command_buffer,
            self.primary_buffer.expect("Primary buffer not initialized"),
            self.staging_buffer.expect("Staging buffer not initialized"),
            buffer_size,
            copy_region,
        );
    }

    pub fn record_copy_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer_from: vk::Buffer,
        buffer_to: vk::Buffer,
        buffer_size: vk::DeviceSize,
        copy_region: vk::BufferCopy,
    ) {
        unsafe {
            self.device.cmd_copy_buffer(command_buffer, buffer_from, buffer_to, &[copy_region]);
        }
    }

    pub fn record_primary_buffer_memory_barrier(
        &self,
        command_buffer: vk::CommandBuffer,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) {
        debug!("Kompute Tensor recording PRIMARY buffer memory barrier");

        self.record_buffer_memory_barrier(
            command_buffer,
            self.primary_buffer.expect("Primary buffer not initialized"),
            src_access_mask,
            dst_access_mask,
            src_stage_mask,
            dst_stage_mask,
        );
    }

    pub fn record_staging_buffer_memory_barrier(
        &self,
        command_buffer: vk::CommandBuffer,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) {
        debug!("Kompute Tensor recording STAGING buffer memory barrier");

        self.record_buffer_memory_barrier(
            command_buffer,
            self.staging_buffer.expect("Staging buffer not initialized"),
            src_access_mask,
            dst_access_mask,
            src_stage_mask,
            dst_stage_mask,
        );
    }

    pub fn record_buffer_memory_barrier(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) {
        debug!("Kompute Tensor recording buffer memory barrier");

        let buffer_size = self.memory_size();

        let buffer_memory_barrier = vk::BufferMemoryBarrier::builder()
            .buffer(buffer)
            .size(buffer_size as u64)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .build();

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_memory_barrier],
                &[],
            );
        }
    }

    pub fn construct_descriptor_buffer_info(&self) -> vk::DescriptorBufferInfo {
        debug!("Kompute Tensor construct descriptor buffer info size {}", self.memory_size());

        let buffer_size = self.memory_size();

        vk::DescriptorBufferInfo {
            buffer: self.primary_buffer.expect("Primary buffer not initialized"),
            offset: 0,
            range: buffer_size as u64,
        }
    }

    pub fn get_primary_buffer_usage_flags(&self) -> vk::BufferUsageFlags {
        match self.tensor_type {
            TensorTypes::Device | TensorTypes::Host => {
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
            }
            TensorTypes::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
            _ => panic!("Kompute Tensor invalid tensor type"),
        }
    }

    pub fn get_primary_memory_property_flags(&self) -> vk::MemoryPropertyFlags {
        match self.tensor_type {
            TensorTypes::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            TensorTypes::Host => vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            TensorTypes::Storage => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            _ => panic!("Kompute Tensor invalid tensor type"),
        }
    }

    pub fn get_staging_buffer_usage_flags(&self) -> vk::BufferUsageFlags {
        match self.tensor_type {
            TensorTypes::Device => {
                vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
            }
            _ => panic!("Kompute Tensor invalid tensor type"),
        }
    }

    pub fn get_staging_memory_property_flags(&self) -> vk::MemoryPropertyFlags {
        match self.tensor_type {
            TensorTypes::Device => vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            _ => panic!("Kompute Tensor invalid tensor type"),
        }
    }

    pub fn allocate_memory_create_gpu_resources(&mut self, entry: Arc<Entry>, instance: Arc<Instance>) {
        debug!("Kompute Tensor creating buffer");

        debug!("Kompute Tensor creating primary buffer and memory");
        unsafe {
            self.primary_buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(self.memory_size() as u64)
                    .usage(self.get_primary_buffer_usage_flags())
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            ).ok();
            
            self.primary_memory = Some(self.allocate_bind_memory(
                entry.clone(),
                instance.clone(),
                self.primary_buffer.as_ref().unwrap(),
                self.get_primary_memory_property_flags(),
            ));
        };
        if self.tensor_type == TensorTypes::Device {
            // staging buffer
            debug!("Kompute Tensor creating staging buffer and memory");
            unsafe {
                self.staging_buffer = self.device.create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(self.memory_size() as u64)
                        .usage(self.get_staging_buffer_usage_flags())
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                ).ok();
                self.staging_memory = Some(self.allocate_bind_memory(
                    entry.clone(),
                    instance.clone(),
                    self.staging_buffer.as_ref().unwrap(),
                    self.get_staging_memory_property_flags(),
                ));
            };
        }

        debug!("Kompute Tensor buffer & memory creation successful");
    }

    pub fn create_buffer(&self, buffer: &mut vk::Buffer, buffer_usage_flags: vk::BufferUsageFlags) {
        debug!("Kompute Tensor creating buffer");
    }

    pub fn to_string(&self) -> String {
        match self.data_type {
            TensorDataTypes::Bool => "Bool".to_string(),
            TensorDataTypes::Int => "Int".to_string(),
            TensorDataTypes::UnsignedInt => "UnsignedInt".to_string(),
            TensorDataTypes::Float => "Float".to_string(),
            TensorDataTypes::Double => "Double".to_string(),
            TensorDataTypes::Unknown => "Unknown".to_string(),
        }
    }

    pub fn allocate_bind_memory(
        &self,
        entry: Arc<Entry>,
        instance: Arc<Instance>,
        buffer: &vk::Buffer,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> vk::DeviceMemory {
        debug!("Kompute Tensor allocating and binding memory");
        let memory_properties = unsafe {
            // little problem here: where do we get instance? 
            instance.get_physical_device_memory_properties(
                *self.physical_device.as_ref(),
            )
        };

        let memory_requirements = unsafe {
            self.device.get_buffer_memory_requirements(*buffer)
        };

        let mut memory_type_index = None;
        for i in 0..memory_properties.memory_type_count {
            if memory_requirements.memory_type_bits & (1 << i) != 0 {
                if (memory_properties.memory_types[i as usize].property_flags
                    & memory_property_flags)
                    == memory_property_flags
                {
                    memory_type_index = Some(i);
                    break;
                }
            }
        }

        let memory_type_index = memory_type_index.expect("Memory type index for buffer creation not found");

        debug!(
            "Kompute Tensor allocating memory index: {}, size {}, flags: {:?}",
            memory_type_index,
            memory_requirements.size,
            memory_property_flags
        );

        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index)
            .build();

        let device_memory = unsafe {
            self.device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Failed to allocate memory")
        };

        unsafe {
            self.device
                .bind_buffer_memory(*buffer, device_memory, 0)
                .expect("Failed to bind buffer memory");
        }
        device_memory
    }

    fn destroy(&mut self) {
        debug!("Kompute Tensor started destroy()");

        // Setting raw data to null regardless whether device is available to
        // invalidate Tensor
        self.raw_data = std::ptr::null_mut();
        self.size = 0;
        self.data_type_memory_size = 0;

        // Unmap the current memory data
        unsafe {
            debug!("Kompute Tensor destroying primary buffer");
            match self.primary_buffer {
                Some(buffer) => self.device.destroy_buffer(buffer, None),
                None => debug!("Kompute Tensor primary buffer was never initialized -> skip."),
            }
        }
        unsafe {
            debug!("Kompute Tensor destroying staging buffer");
            match self.staging_buffer {
                Some(buffer) => self.device.destroy_buffer(buffer, None),
                None => debug!("Kompute Tensor staging buffer was never initialized -> skip."),
            }
        }
        
        if self.tensor_type() != TensorTypes::Storage {
            self.unmap_raw_data();
        }
        
        unsafe {
            debug!("Kompose Tensor freeing primary memory");
            match self.primary_memory {
                Some(memory) => self.device.free_memory(memory, None),
                None => debug!("Kompute Tensor primary memory was never initialized -> skip."),
            }
        }

        unsafe{
            debug!("Kompose Tensor freeing staging memory");
            match self.staging_memory {
                Some(memory) => self.device.free_memory(memory, None),
                None => debug!("Kompute Tensor staging memory was never initialized -> skip."),
            }
        }
        debug!("Kompute Tensor successful destroy()");
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        debug!("Kompute Tensor started destroy()");

        // Setting raw data to null regardless whether device is available to
        // invalidate Tensor
        self.raw_data = std::ptr::null_mut();
        self.size = 0;
        self.data_type_memory_size = 0;

        // Unmap the current memory data
        unsafe {
            debug!("Kompute Tensor destroying primary buffer");
            match self.primary_buffer {
                Some(buffer) => self.device.destroy_buffer(buffer, None),
                None => debug!("Kompute Tensor primary buffer was never initialized -> skip."),
            }
        }
        unsafe {
            debug!("Kompute Tensor destroying staging buffer");
            match self.staging_buffer {
                Some(buffer) => self.device.destroy_buffer(buffer, None),
                None => debug!("Kompute Tensor staging buffer was never initialized -> skip."),
            }
        }
        
        if self.tensor_type() != TensorTypes::Storage {
            self.unmap_raw_data();
        }
        
        unsafe {
            debug!("Kompose Tensor freeing primary memory");
            match self.primary_memory {
                Some(memory) => self.device.free_memory(memory, None),
                None => debug!("Kompute Tensor primary memory was never initialized -> skip."),
            }
        }

        unsafe{
            debug!("Kompose Tensor freeing staging memory");
            match self.staging_memory {
                Some(memory) => self.device.free_memory(memory, None),
                None => debug!("Kompute Tensor staging memory was never initialized -> skip."),
            }
        }
        debug!("Kompute Tensor successful destroy()");
    }
}

pub struct TensorT<T> {
    physical_device: Arc<vk::PhysicalDevice>,
    device: Arc<ash::Device>,
    raw_data: *mut T,
    size: usize,
    data_type_memory_size: usize,
    tensor_type: TensorTypes,
}

impl<T> TensorT<T> {
    pub fn new(
        physical_device: Arc<vk::PhysicalDevice>,
        device: Arc<ash::Device>,
        data: &[T],
        tensor_type: TensorTypes,
    ) -> Self {
        let raw_data = data.as_ptr() as *mut T;
        let size = data.len();
        let data_type_memory_size = std::mem::size_of::<T>();

        TensorT {
            physical_device,
            device,
            raw_data,
            size,
            data_type_memory_size,
            tensor_type,
        }
    }

    pub fn data(&self) -> *mut T {
        self.raw_data
    }

    pub fn vector(&self) -> Vec<T> {
        unsafe { Vec::from_raw_parts(self.raw_data, self.size, self.size) }
    }

    pub fn set_data(&mut self, data: &[T]) {
        if data.len() != self.size {
            panic!("Cannot set data of different sizes");
        }

        self.raw_data = data.as_ptr() as *mut T;
    }
}

impl<T> Drop for TensorT<T> {
    fn drop(&mut self) {
        // Cleanup code here
    }
}

use std::ops::Index;

impl<T> Index<usize> for TensorT<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.raw_data.add(index) }
    }
}

// TODO: f16 with some crate

impl Into<Tensor> for TensorT<f32> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::Float,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<f64> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::Double,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<i64> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::Int,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<u64> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::UnsignedInt,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<i32> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::Int,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<u32> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::UnsignedInt,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<i16> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::Int,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<u16> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::UnsignedInt,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<i8> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::Int,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<u8> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::UnsignedInt,
            self.tensor_type,
        )
    }
}

impl Into<Tensor> for TensorT<bool> {
    fn into(self) -> Tensor {
        Tensor::new(
            self.physical_device.clone(),
            self.device.clone(),
            self.raw_data as *mut c_void,
            self.size as u32,
            self.data_type_memory_size as u32,
            TensorDataTypes::Bool,
            self.tensor_type,
        )
    }
}
