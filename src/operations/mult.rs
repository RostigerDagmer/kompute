use std::borrow::BorrowMut;
use std::error::Error;
use std::sync::{Arc, Mutex};
use crate::algorithm::Algorithm;
use crate::tensor::RawTensor;
use crate::shaderutil::ShaderSource;

use super::OpAlgoDispatch;

pub struct OpMult {
    dispatch: OpAlgoDispatch,
    tensors: Vec<Arc<Mutex<RawTensor>>>, // <- opportunity for shape checking with the wrapped Tensor Type
    // algorithm: Arc<Algorithm>,
}

impl OpMult {
    pub fn new(tensors: Vec<Arc<Mutex<RawTensor>>>, algorithm: Arc<Mutex<Algorithm>>) -> Result<Self, Box<dyn Error>> {
        if tensors.len() != 3 {
            return Err("Kompute OpMult expected 3 tensors".into());
        }

        algorithm.lock().unwrap().borrow_mut().rebuild(tensors.clone(), ShaderSource::Source("ShaderOpMult.comp".into()), None, None, None);
        let dispatch = OpAlgoDispatch::new(algorithm);
        Ok(Self {
            dispatch,
            tensors,
            // algorithm: Arc::new(algorithm),
        })
    }
}

impl super::OpBase for OpMult {

    fn record(&mut self, command_buffer: ash::vk::CommandBuffer) {
        self.dispatch.record(command_buffer);
    }

    fn pre_eval(&self, command_buffer: ash::vk::CommandBuffer) {
        self.dispatch.pre_eval(command_buffer);
    }

    fn post_eval(&self, command_buffer: ash::vk::CommandBuffer) {
        self.dispatch.post_eval(command_buffer);
    }

    
}