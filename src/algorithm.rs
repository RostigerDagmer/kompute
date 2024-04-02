use ash::vk::{self, ShaderStageFlags};
use log::{debug, info, warn};
use std::ffi::{CString, c_void};
use std::slice;
use std::sync::{Arc, Mutex};
use std::vec;

use crate::shape::Shape;
use crate::tensor::{RawTensor, Tensor, TensorData};
use crate::shaderutil::*;

type PushConstants<T: TensorData> = Vec<T>;
type SpecConstants<S: TensorData> = Vec<S>;

pub struct Algorithm {
    pipeline: Option<vk::Pipeline>,
    pipeline_cache: Option<vk::PipelineCache>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_set: Option<vk::DescriptorSet>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    shader_module: Option<vk::ShaderModule>,
    device: Arc<ash::Device>,
    tensors: Vec<Arc<Mutex<RawTensor>>>,
    spirv: Vec<u32>,
    workgroup: [u32; 3],
    push_constant_data: Option<Vec<u8>>,
    push_constant_size: usize,
    push_constant_data_type_memory_size: usize,
    specialization_constants_data: Option<Vec<u8>>,
    specialization_constants_size: usize,
    specialization_constants_data_type_memory_size: usize,
    specialization_constants: Vec<Box<dyn TensorData>>,
    push_constants: Vec<Box<dyn TensorData>>,
}

impl Algorithm {
    pub fn new<T: TensorData, S: TensorData>(
        device: Arc<ash::Device>,
        tensors: Vec<Arc<Mutex<RawTensor>>>,
        shader_source: ShaderSource,
        workgroup: [u32; 3],
        specialization_constants: Vec<T>,
        push_constants: Vec<S>,
    ) -> Self {

        let spirv = match shader_source {
            ShaderSource::SPIRV(spirv) => spirv,
            ShaderSource::Source(path) => compile_shader(path).unwrap(),
            ShaderSource::Code(code) => compile_code(code, shaderc::ShaderKind::Compute).unwrap()
        };

        let specialization_constants_size = specialization_constants.len();
        let specialization_constants_data_type_memory_size = std::mem::size_of::<T>();
        let specialization_constants_data = if(specialization_constants.len() <= 0) {
            None
        } else {
            Some(specialization_constants
            .iter()
            .map(|tensor| tensor.bytes().into_iter().cloned())
            .flatten()
            .collect::<Vec<u8>>())
        };
        let specialization_constants = specialization_constants.iter().map(|c| c.boxed()).collect();

        let push_constant_size = push_constants.len();
        let push_constant_data_type_memory_size = std::mem::size_of::<S>();
        let push_constant_data = if(push_constants.len() <= 0) {
            None
        } else {
            Some(push_constants
            .iter()
            .map(|tensor| tensor.bytes().into_iter().cloned())
            .flatten()
            .collect::<Vec<u8>>())
        };
        let push_constants = push_constants.iter().map(|c| c.boxed()).collect();

        Algorithm {
            pipeline: None,
            pipeline_cache: None,
            pipeline_layout: None,
            descriptor_pool: None,
            descriptor_set: None,
            descriptor_set_layout: None,
            shader_module: None,
            device,
            tensors,
            spirv,
            workgroup,
            push_constant_data,
            push_constant_size,
            push_constant_data_type_memory_size,
            specialization_constants_data,
            specialization_constants_size,
            specialization_constants_data_type_memory_size,
            specialization_constants,
            push_constants,
        }
    }

    pub fn init(&mut self) {
        debug!("Kompute Algorithm init started");

        self.create_parameters();
        self.create_shader_module();
        self.create_pipeline();
    }

    pub fn rebuild(
        &mut self,
        tensors: Vec<Arc<Mutex<RawTensor>>>,
        shader_source: ShaderSource,
        workgroup: Option<[u32; 3]>,
        specialization_constants: Option<Vec<Box<dyn TensorData>>>,
        push_constants: Option<Vec<Box<dyn TensorData>>>,
    ) {
        debug!("Kompute Algorithm rebuild started");

        self.tensors = tensors;
        self.spirv = match shader_source {
            ShaderSource::SPIRV(spirv) => spirv,
            ShaderSource::Source(path) => compile_shader(path).unwrap(),
            ShaderSource::Code(code) => compile_code(code, shaderc::ShaderKind::Compute).unwrap()
        };

        match specialization_constants {
            Some(specialization_constants) => {
                let data = specialization_constants
                    .iter()
                    .map(|tensor| tensor.as_ref().bytes().into_iter().cloned())
                    .flatten()
                    .collect::<Vec<u8>>();

                let memory_size = 1;
                let size = specialization_constants.len();

                self.specialization_constants = specialization_constants;
                self.specialization_constants_data = Some(data);
                self.specialization_constants_data_type_memory_size = memory_size;
                self.specialization_constants_size = size;
            },
            None => {}
        }

        match push_constants {
            Some (push_constants) => {
                // // preform non primitve cast of push constants Vec<P> to Vec<u8>
                let data = self.push_constants.iter()
                    .map(|tensor| tensor.as_ref().bytes().into_iter().cloned())
                    .flatten()
                    .collect::<Vec<u8>>();
                let size = data.len();
                let memory_size = 1;
    
                self.push_constants = push_constants;
                self.push_constant_data = Some(data);
                self.push_constant_data_type_memory_size = memory_size;
                self.push_constant_size = size;
                todo!()
            },
            None => {}
        }

        match workgroup {
            Some(workgroup) => self.set_workgroup(workgroup, if self.tensors.is_empty() {
                1
            } else {
                self.tensors[0].lock().unwrap().size()
            }),
            None => {}
        }

        // Descriptor pool is created first so if available then destroy all
        // before rebuild
        if self.is_init() {
            self.destroy();
        }

        self.create_parameters();
        self.create_shader_module();
        self.create_pipeline();
    }

    pub fn is_init(&self) -> bool {
        self.pipeline.is_some()
            && self.pipeline_cache.is_some()
            && self.pipeline_layout.is_some()
            && self.descriptor_pool.is_some()
            && self.descriptor_set.is_some()
            && self.descriptor_set_layout.is_some()
            && self.shader_module.is_some()
    }

    pub fn destroy(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            debug!("Kompute Algorithm Destroying pipeline");
            unsafe { self.device.destroy_pipeline(pipeline, None) };
        }

        if let Some(pipeline_cache) = self.pipeline_cache.take() {
            debug!("Kompute Algorithm Destroying pipeline cache");
            unsafe { self.device.destroy_pipeline_cache(pipeline_cache, None) };
        }

        if let Some(pipeline_layout) = self.pipeline_layout.take() {
            debug!("Kompute Algorithm Destroying pipeline layout");
            unsafe { self.device.destroy_pipeline_layout(pipeline_layout, None) };
        }

        if let Some(shader_module) = self.shader_module.take() {
            debug!("Kompute Algorithm Destroying shader module");
            unsafe { self.device.destroy_shader_module(shader_module, None) };
        }

        if let Some(descriptor_set_layout) = self.descriptor_set_layout.take() {
            debug!("Kompute Algorithm Destroying Descriptor Set Layout");
            unsafe { self.device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
        }

        if let Some(descriptor_pool) = self.descriptor_pool.take() {
            debug!("Kompute Algorithm Destroying Descriptor Pool");
            unsafe { self.device.destroy_descriptor_pool(descriptor_pool, None) };
        }
    }

    pub fn set_push_constants(
        &mut self,
        push_constants_data: *mut c_void,
        push_constants_size: usize,
        push_constants_data_type_memory_size: usize,
    ) {
        self.push_constant_data = Some(unsafe {
            Vec::from_raw_parts(
                push_constants_data as *mut u8,
                push_constants_size * push_constants_data_type_memory_size,
                push_constants_size * push_constants_data_type_memory_size,
            )
        });
        self.push_constant_size = push_constants_size;
        self.push_constant_data_type_memory_size = push_constants_data_type_memory_size;
    }

    pub fn create_parameters(&mut self) {
        debug!("Kompute Algorithm createParameters started");

        let descriptor_pool_sizes = vec![vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(self.tensors.len() as u32)
            .build()];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .max_sets(1)
            .pool_sizes(&descriptor_pool_sizes);

        debug!("Kompute Algorithm creating descriptor pool");
        self.descriptor_pool = Some(
            unsafe { self.device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .expect("Failed to create descriptor pool") },
        );

        let mut descriptor_set_bindings = Vec::new();

        // debug!("Kompute Algorithm creating descriptor set bindings: {:#?}", self.tensors);
        
        for (i, tensor) in self.tensors.iter().enumerate() {
            let descriptor_set_binding = vk::DescriptorSetLayoutBinding::builder()
                .binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build();
            descriptor_set_bindings.push(descriptor_set_binding);
        }

        // debug!("Kompute Algorithm creating descriptor set bindings: {:#?}", descriptor_set_bindings);
    
        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::empty())
            .bindings(&descriptor_set_bindings);

        debug!("Kompute Algorithm creating descriptor set layout");
        self.descriptor_set_layout = Some(
            unsafe { self.device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .expect("Failed to create descriptor set layout") },
        );

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool.unwrap())
            .set_layouts(slice::from_ref(self.descriptor_set_layout.as_ref().unwrap()));

        debug!("Kompute Algorithm allocating descriptor sets");
        let descriptor_sets = unsafe { self
            .device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .expect("Failed to allocate descriptor sets") };
        self.descriptor_set = Some(descriptor_sets[0]);

        debug!("Kompute Algorithm updating descriptor sets");

        let compute_write_descriptor_infos = self.tensors.iter().map(|tensor| {
            let descriptor_buffer_info = tensor.lock().unwrap().construct_descriptor_buffer_info();
            vec![descriptor_buffer_info]
        }).collect::<Vec<Vec<_>>>();
        let compute_write_descriptor_sets = compute_write_descriptor_infos.iter().enumerate().map(|(i, descriptor_buffer_info)| {
            
            // debug!("Kompute Algorithm updating descriptor sets[{:#?}]: {:#?}", i, descriptor_buffer_info);

            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set.unwrap())
                .dst_binding(i as u32)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(descriptor_buffer_info.as_slice())
                .build()
        }).collect::<Vec<vk::WriteDescriptorSet>>();

        // debug!("Kompute Algorithm updating descriptor sets: {:#?}", compute_write_descriptor_sets);

        unsafe { self.device
            .update_descriptor_sets(&compute_write_descriptor_sets, &[]) };
    }

    pub fn create_shader_module(&mut self) {
        debug!("Kompute Algorithm createshader_module started");

        let shader_module_info = vk::ShaderModuleCreateInfo::builder()
            .flags(vk::ShaderModuleCreateFlags::empty())
            .code(self.spirv.as_slice());

        debug!(
            "Kompute Algorithm Creating shader module. ShaderFileSize: {}",
            self.spirv.len()
        );
        self.shader_module = Some(
            unsafe { self.device
                .create_shader_module(&shader_module_info, None)
                .expect("Failed to create shader module") },
        );

        debug!("Kompute Algorithm create shader module success");
    }

    pub fn create_pipeline(&mut self) {
        debug!("Kompute Algorithm calling create Pipeline");

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(slice::from_ref(self.descriptor_set_layout.as_ref().unwrap()));

        let pipeline_layout_info = if self.push_constant_size > 0 {
            pipeline_layout_info.push_constant_ranges(&[
                vk::PushConstantRange::builder()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(
                        (self.push_constant_data_type_memory_size
                            * self.push_constant_size) as u32,
                    )
                    .build(),
            ]).build()
        } else {
            pipeline_layout_info.build()
        };

        self.pipeline_layout = Some(
            unsafe { self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout") },
        );

        let specialization_entries: Vec<vk::SpecializationMapEntry> = (0..self.specialization_constants_size)
            .map(|i| {
                vk::SpecializationMapEntry::builder()
                    .constant_id(i as u32)
                    .offset((self.specialization_constants_data_type_memory_size * i) as u32)
                    .size(self.specialization_constants_data_type_memory_size)
                    .build()
            })
            .collect();

        let specialization_info = vk::SpecializationInfo::builder()
            .map_entries(&specialization_entries)
            .data(self.specialization_constants_data.as_ref().expect("specialization_constants_data was not set when calling create_pipeline").as_slice());
        
        let entry_point = CString::new("main").unwrap();
        let shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader_module.unwrap())
            .name(entry_point.as_c_str())
            .specialization_info(&specialization_info);

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(shader_stage.build())
            .layout(self.pipeline_layout.unwrap())
            .base_pipeline_index(-1);

        let pipeline_cache_info = vk::PipelineCacheCreateInfo::builder();

        self.pipeline_cache = Some(
            unsafe { self.device
                .create_pipeline_cache(&pipeline_cache_info, None)
                .expect("Failed to create pipeline cache") },
        );

        let pipeline_result = unsafe { self.device.create_compute_pipelines(
            self.pipeline_cache.unwrap(),
            &[pipeline_info.build()],
            None,
        ) };

        if let Ok(pipelines) = pipeline_result {
            self.pipeline = Some(pipelines[0]);
        } else {
            warn!(
                "Failed to create pipeline result: {:?}",
                pipeline_result.err()
            );
        }

        debug!("Kompute Algorithm Create Pipeline Success");
    }

    pub fn record_bind_core(&self, command_buffer: vk::CommandBuffer) {
        debug!("Kompute Algorithm binding pipeline");

        unsafe { self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.unwrap(),
        ) };

        debug!("Kompute Algorithm binding descriptor sets");

        unsafe { self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout.unwrap(),
            0,
            &[self.descriptor_set.unwrap()],
            &[],
        ) };
    }

    pub fn record_bind_push(&self, command_buffer: vk::CommandBuffer) {
        if self.push_constant_size > 0 {
            debug!(
                "Kompute Algorithm binding push constants memory size: {}",
                self.push_constant_size * self.push_constant_data_type_memory_size
            );
            debug!("Kompute Algorithm binding push constants: {:?}", self.push_constant_data);
            unsafe { self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout.unwrap(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                self.push_constant_data.as_ref().expect("push constant data not set when record_bind_push was called").as_slice(),
            ) };
        }
    }

    pub fn record_dispatch(&self, command_buffer: vk::CommandBuffer) {
        debug!("Kompute Algorithm recording dispatch");

        unsafe { self.device.cmd_dispatch(
            command_buffer,
            self.workgroup[0],
            self.workgroup[1],
            self.workgroup[2],
        ) };
    }

    pub fn set_workgroup(&mut self, workgroup: [u32; 3], min_size: u32) {
        info!("Kompute OpAlgoCreate setting dispatch size");

        if workgroup[0] > 0 {
            self.workgroup = workgroup;
        } else {
            self.workgroup = [min_size, 1, 1];
        }

        info!(
            "Kompute OpAlgoCreate set dispatch size X: {}, Y: {}, Z: {}",
            self.workgroup[0],
            self.workgroup[1],
            self.workgroup[2]
        );
    }

    pub fn get_workgroup(&self) -> [u32; 3] {
        self.workgroup
    }

    pub fn get_tensors(&self) -> &[Arc<Mutex<RawTensor>>] {
        &self.tensors
    }
}
