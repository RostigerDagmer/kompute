use crate::{shape::Shape, tensor::{Tensor, TensorTypes}};
use ash::vk;

use super::OpBase;

pub struct OpTensorCopy<S: Shape> {
    tensors: Vec<std::sync::Arc<Tensor<S>>>,
}

impl<S: Shape> OpTensorCopy<S> {
    pub fn new(tensors: Vec<std::sync::Arc<Tensor<S>>>) -> Result<Self, String> {
        if tensors.len() < 2 {
            return Err("OpTensorCopy called with less than 2 tensors".to_string());
        }

        let data_type = tensors[0].data_type();
        let size = tensors[0].size();
        for tensor in &tensors {
            if tensor.data_type() != data_type {
                return Err(format!(
                    "Attempting to copy tensors of different types from {:?} to {:?}",
                    data_type,
                    tensor.data_type()
                ));
            }
            if tensor.size() != size {
                return Err(format!(
                    "Attempting to copy tensors of different sizes from {} to {}",
                    size,
                    tensor.size()
                ));
            }
        }

        Ok(Self { tensors })
    }
}

impl<S: Shape> OpBase for OpTensorCopy<S> {

    fn record(&mut self, command_buffer: vk::CommandBuffer) {
        for i in 1..self.tensors.len() {
            self.tensors[i].record_copy_from(command_buffer, self.tensors[0].clone());
        }
    }

    fn pre_eval(&self, _command_buffer: vk::CommandBuffer) {
        // Implementation for preEval
    }

    fn post_eval(&self, _command_buffer: vk::CommandBuffer) {
        if self.tensors[0].tensor_type() == TensorTypes::Storage {
            return;
        }
        let data = self.tensors[0].raw_data();

        for i in 1..self.tensors.len() {
            if self.tensors[i].tensor_type() == TensorTypes::Storage {
                continue;
            }
            self.tensors[i].set_raw_data(data);
        }
    }
}
