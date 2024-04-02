use std::sync::{Arc, Mutex};

use crate::tensor::{RawTensor, TensorTypes};
use ash::vk;
use log::debug;

pub struct OpTensorSyncLocal {
    tensors: Vec<Arc<Mutex<RawTensor>>>,
}

impl OpTensorSyncLocal {
    pub fn new(tensors: Vec<Arc<Mutex<RawTensor>>>) -> Result<Self, String> {
        if tensors.is_empty() {
            return Err("OpTensorSyncLocal called with no tensors".to_string());
        }
        Ok(Self { tensors })
    }
}
impl super::OpBase for OpTensorSyncLocal {
    fn record(&mut self, command_buffer: vk::CommandBuffer) {
        for tensor in &self.tensors {
            let tensor = tensor.lock().unwrap();
            if tensor.tensor_type() == TensorTypes::Device {
                tensor.record_primary_buffer_memory_barrier(
                    command_buffer,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                );

                tensor.record_copy_from_device_to_staging(command_buffer);

                tensor.record_primary_buffer_memory_barrier(
                    command_buffer,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::HOST_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                );
            }
        }
    }

    fn pre_eval(&self, _command_buffer: vk::CommandBuffer) {
        // No implementation needed
    }

    fn post_eval(&self, _command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpTensorSyncLocal mapping data into tensor local");
    }
}
