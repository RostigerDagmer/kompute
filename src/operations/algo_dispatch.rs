use ash::vk;
use log::debug;
use std::borrow::BorrowMut;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Mutex, Arc};
use crate::algorithm::Algorithm;
use crate::tensor::TensorData;

pub struct OpAlgoDispatch {
    push_constants_data: *mut c_void,
    push_constants_size: usize,
    push_constants_data_type_memory_size: usize,
    algorithm: Arc<Mutex<Algorithm>>,
}

impl OpAlgoDispatch {
    pub fn new(algorithm: Arc<Mutex<Algorithm>>) -> Self {
        OpAlgoDispatch {
            push_constants_data: ptr::null_mut(),
            push_constants_size: 0,
            push_constants_data_type_memory_size: 0,
            algorithm,
        }
    }
    pub fn new_with_push_constants<T: TensorData>(
        algorithm: Arc<Mutex<Algorithm>>,
        push_constants_data: Vec<T>
    ) -> Self {
        let push_constants_data_type_memory_size = std::mem::size_of::<T>();
        let push_constants_size = push_constants_data.len() * push_constants_data_type_memory_size;
        let push_constants_data = push_constants_data.as_ptr() as *mut c_void;
        OpAlgoDispatch {
            push_constants_data,
            push_constants_size,
            push_constants_data_type_memory_size,
            algorithm,
        }
    }
}
impl super::OpBase for OpAlgoDispatch {

    fn record(&mut self, command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpAlgoDispatch record called");

        let mut guard = self.algorithm.lock().unwrap();
        // Barrier to ensure the data is finished writing to buffer memory
        for tensor in guard.get_tensors() {
            tensor.lock().unwrap().record_primary_buffer_memory_barrier(
                command_buffer,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            );
        }

        if self.push_constants_size > 0 {
            guard.borrow_mut().set_push_constants(
                self.push_constants_data,
                self.push_constants_size,
                self.push_constants_data_type_memory_size,
            );
        }

        guard.record_bind_core(command_buffer);
        guard.record_bind_push(command_buffer);
        guard.record_dispatch(command_buffer);
    }

    fn pre_eval(&self, _command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpAlgoDispatch preEval called");
    }

    fn post_eval(&self, _command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpAlgoDispatch postSubmit called");
    }
}
