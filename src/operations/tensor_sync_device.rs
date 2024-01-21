use crate::tensor::{Tensor, TensorTypes};
use ash::vk;
use log::debug;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::OpBase;

pub struct OpTensorSyncDevice {
    tensors: Vec<Arc<Tensor>>,
}

impl OpTensorSyncDevice {
    pub fn new(tensors: Vec<Arc<Tensor>>) -> Result<Self, Box<dyn Error>> {
        if tensors.is_empty() {
            return Err("OpTensorSyncDevice called with no tensors".into());
        }

        Ok(Self { tensors })
    }
}
impl OpBase for OpTensorSyncDevice {
    fn record(&mut self, command_buffer: vk::CommandBuffer) {
        for tensor in &self.tensors {
            if tensor.tensor_type() == TensorTypes::Device {
                tensor.record_copy_from_staging_to_device(command_buffer);
            }
        }
    }

    fn pre_eval(&self, _command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpTensorSyncDevice pre_eval called");
    }

    fn post_eval(&self, _command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpTensorSyncDevice post_eval called");
    }
}

impl fmt::Debug for OpTensorSyncDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpTensorSyncDevice")
    }
}
