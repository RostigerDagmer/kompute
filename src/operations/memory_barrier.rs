use std::sync::Arc;
use ash::vk;
use crate::tensor::{RawTensor, Tensor};
use log::debug;

pub struct OpMemoryBarrier {
    tensors: Vec<Arc<RawTensor>>,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    barrier_on_primary: bool,
}

impl OpMemoryBarrier {
    pub fn new(
        tensors: Vec<Arc<RawTensor>>,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        barrier_on_primary: bool,
    ) -> Self {
        Self {
            tensors,
            src_access_mask,
            dst_access_mask,
            src_stage_mask,
            dst_stage_mask,
            barrier_on_primary,
        }
    }

    pub fn record(&self, command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpMemoryBarrier record called");

        if self.barrier_on_primary {
            for tensor in &self.tensors {
                tensor.record_primary_buffer_memory_barrier(
                    command_buffer,
                    self.src_access_mask,
                    self.dst_access_mask,
                    self.src_stage_mask,
                    self.dst_stage_mask,
                );
            }
        } else {
            for tensor in &self.tensors {
                tensor.record_staging_buffer_memory_barrier(
                    command_buffer,
                    self.src_access_mask,
                    self.dst_access_mask,
                    self.src_stage_mask,
                    self.dst_stage_mask,
                );
            }
        }
    }

    pub fn pre_eval(&self, _command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpMemoryBarrier preEval called");
    }

    pub fn post_eval(&self, _command_buffer: vk::CommandBuffer) {
        debug!("Kompute OpMemoryBarrier postSubmit called");
    }
}
