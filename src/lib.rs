mod algorithm;
mod core;
mod manager;
mod sequence;
mod tensor;
mod shader_logistic_regression;
mod shader_op_mult;
mod operations;
mod shaderutil;


#[cfg(test)]
mod tests {
    use super::*;
    
    use env_logger;
    use log::debug;
    use super::manager::Manager;
    use super::tensor::Tensor;
    use super::algorithm::Algorithm;
    use super::operations::OpTensorSyncDevice;
    use super::operations::OpAlgoDispatch;
    use super::operations::OpTensorSyncLocal;

    #[test]
    fn it_works() {
        std::env::set_var("RUST_LOG", "debug");
        env_logger::init();
        let vec = vec![1,2,3,4];
        let mut m = manager::Manager::new(0,&[0],&[]);
        let t1 = m.tensor(&vec, tensor::TensorTypes::Device).unwrap();
        assert_eq!(t1.tensor_type(), tensor::TensorTypes::Device);
    }

    #[test]
    fn algo() {
        
    }
    #[cfg(test)]
    mod tests {
        use std::{sync::{Arc, Mutex}, borrow::BorrowMut};

        use ash::vk;
        use log::info;

        use crate::{shaderutil::ShaderSource, operations::OpTensorCopy};

        use super::*;
        #[test]
        fn kp_library() {
            std::env::set_var("RUST_BACKTRACE", "1");
            std::env::set_var("RUST_LOG", "trace");
            env_logger::init();
            let mut mgr = Manager::new(
                0,
                &[0],
                &[]);
            let a: Vec<f32> = vec![2., 2., 2.];
            let b: Vec<f32> = vec![1., 2., 3.];
            let c: Vec<u32> = vec![0, 0, 0];
            let tensor_in_a = mgr.tensor(&a, tensor::TensorTypes::Device).unwrap();
            let tensor_in_b = mgr.tensor(&b, tensor::TensorTypes::Device).unwrap();
            let tensor_out_a = mgr.tensor(&c, tensor::TensorTypes::Device).unwrap();
            let tensor_out_b = mgr.tensor(&c, tensor::TensorTypes::Device).unwrap();

            let a = tensor_in_a.vector::<f32>();
            let b = tensor_in_b.vector::<f32>();
            info!("tensor_in_a: {:?}", a);
            info!("tensor_in_b: {:?}", b);
            let a = tensor_out_a.vector::<f32>();
            let b = tensor_out_b.vector::<f32>();
            info!("tensor_out_a: {:?}", a);
            info!("tensor_out_b: {:?}", b);
            
            let shader = r#"
                #version 460

                layout (local_size_x = 1) in;

                layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
                layout(set = 0, binding = 1) buffer buf_in_b { float in_b[]; };
                layout(set = 0, binding = 2) buffer buf_out_a { uint out_a[]; };
                layout(set = 0, binding = 3) buffer buf_out_b { uint out_b[]; };

                layout(push_constant) uniform PushConstants {
                    float val;
                } push_const;

                layout(constant_id = 0) const float const_one = 0;

                void main() {
                    uint index = gl_GlobalInvocationID.x;
                    out_a[index] += uint( in_a[index] * in_b[index] );
                    out_b[index] += uint( const_one * push_const.val );
                }
            "#;
            let shader = ShaderSource::Code(shader.to_string());

            let params = vec![
                tensor_in_a.clone(),
                tensor_in_b.clone(),
                tensor_out_a.clone(),
                tensor_out_b.clone(),
            ];

            let workgroup = [3, 1, 1];
            let spec_consts = vec![2.0 as f32];
            let push_consts_a = vec![2.0 as f32];
            let push_consts_b = vec![3.0 as f32];

            let algorithm = mgr.algorithm(
                params.clone(),
                shader,
                &workgroup,
                spec_consts,
                push_consts_a,
            ).unwrap();

            mgr.sequence(0, 0)
                .unwrap()
                .lock()
                .borrow_mut()
                .as_mut()
                .unwrap()
                .record(Arc::new(Mutex::new(OpTensorSyncDevice::new(params.clone()).unwrap())))
                .record(Arc::new(Mutex::new(OpAlgoDispatch::new(algorithm.clone()))))
                .record(Arc::new(Mutex::new(OpTensorCopy::new(vec![tensor_in_a.clone(), tensor_out_a.clone()]).unwrap())))
                // .record(Arc::new(Mutex::new(OpAlgoDispatch::new_with_push_constants(algorithm.clone(), push_consts_b))))
                .eval();
                // .eval();

            // let mut sq = mgr.sequence(0, 0).unwrap();
            // sq.lock().unwrap().eval_async();

            // sq.lock().unwrap().eval_await(4000);

            let a = tensor_out_a.vector::<f32>();
            let b = tensor_out_b.vector::<f32>();
            info!("tensor_in_a: {:?}", a);
            info!("tensor_in_b: {:?}", b);

            assert_eq!(a, vec![4. as f32, 8. as f32, 12. as f32]);
            assert_eq!(b, vec![10. as f32, 10. as f32, 10. as f32]);
        }
    }
}
