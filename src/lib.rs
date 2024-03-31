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
    use crate::operations::OpTensorCopy;

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
        use std::{borrow::BorrowMut, ffi::c_void, mem::ManuallyDrop, sync::{Arc, Mutex}};

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
            let c: Vec<f32> = vec![0., 0., 0.];
            let tensor_in_a = mgr.tensor(&a, tensor::TensorTypes::Device).unwrap();
            let tensor_in_b = mgr.tensor(&b, tensor::TensorTypes::Device).unwrap();
            let tensor_out_a = mgr.tensor(&c, tensor::TensorTypes::Device).unwrap();
            let tensor_out_b = mgr.tensor(&c, tensor::TensorTypes::Device).unwrap();
            
            info!("tensor_in_a.dtype: {:?}", tensor_in_a.data_type());
            info!("tensor_in_b.dtype: {:?}", tensor_in_b.data_type());
            info!("tensor_out_a.dtype: {:?}", tensor_out_a.data_type());
            info!("tensor_out_b.dtype: {:?}", tensor_out_b.data_type());

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
                layout(set = 0, binding = 2) buffer buf_out_a { float out_a[]; };
                layout(set = 0, binding = 3) buffer buf_out_b { float out_b[]; };

                layout(push_constant) uniform PushConstants {
                    float val;
                } push_const;

                layout(constant_id = 0) const float const_one = 0;

                void main() {
                    uint index = gl_GlobalInvocationID.x;
                    out_a[index] += float( in_a[index] * in_b[index] );
                    out_b[index] += float( const_one * push_const.val );
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
                .as_mut()
                .unwrap()
                .record(Arc::new(Mutex::new(OpTensorSyncDevice::new(params.clone()).unwrap())))
                .record(Arc::new(Mutex::new(OpAlgoDispatch::new(algorithm.clone()))))
                .record(Arc::new(Mutex::new(OpTensorSyncDevice::new(params.clone()).unwrap())))
                .record(Arc::new(Mutex::new(OpAlgoDispatch::new_with_push_constants(algorithm.clone(), push_consts_b))))
                .record(Arc::new(Mutex::new(OpTensorSyncLocal::new(params.clone()).unwrap())))
                .eval();
                // .eval();

            // let mut sq = mgr.sequence(0, 0).unwrap();
            // sq.lock().unwrap().eval_async();

            // sq.lock().unwrap().eval_await(4000);

            let a_in = tensor_in_a.vector::<f32>();
            let b_in = tensor_in_b.vector::<f32>();
            let a = tensor_out_a.vector::<f32>();
            let b = tensor_out_b.vector::<f32>();
            info!("tensor_out_a: {:?}", a);
            info!("tensor_out_b: {:?}", b);
            info!("tensor_in_a: {:?}", a_in);
            info!("tensor_in_b: {:?}", b_in);

            assert_eq!(a, vec![4. as f32, 8. as f32, 12. as f32]);
            assert_eq!(b, vec![10. as f32, 10. as f32, 10. as f32]);
        }


        #[test]
        fn simple() {
            std::env::set_var("RUST_BACKTRACE", "1");
            std::env::set_var("RUST_LOG", "trace");
            env_logger::init();
            let mut mgr = Manager::new(
                0,
                &[0],
                &[]);
            let a: ManuallyDrop<Vec<f32>> = ManuallyDrop::new(vec![1., 2., 3.]);
            let zero: ManuallyDrop<Vec<f32>> = ManuallyDrop::new(vec![0., 0., 0.]);
            let tensor_in_a = mgr.tensor(&a, tensor::TensorTypes::Device).unwrap();
            let tensor_out_a = mgr.tensor(&zero, tensor::TensorTypes::Device).unwrap();
            
            let shader = r#"
                #version 460

                layout(set = 0, binding = 0) buffer tensor_in { float in_[]; };
                layout(set = 0, binding = 1) buffer tensor_out { float out_[]; };

                layout(push_constant) uniform PushConstants {
                    float val;
                } push_const;

                layout(constant_id = 0) const float const_one = 0;

                void main() {
                    uint index = gl_GlobalInvocationID.z * gl_NumWorkGroups.x * gl_WorkGroupSize.x + gl_GlobalInvocationID.y * gl_WorkGroupSize.x + gl_GlobalInvocationID.x;
                    out_[index] = in_[index] * in_[index];
                }
            "#;
            let shader = ShaderSource::Code(shader.to_string());

            let params = vec![
                tensor_in_a.clone(),
                tensor_out_a.clone(),
            ];

            let workgroup = [3, 1, 1];
            let spec_consts = vec![2.0 as f32];
            let push_consts_a = vec![2.0 as f32];

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
                .as_mut()
                .unwrap()
                .record(Arc::new(Mutex::new(OpTensorSyncDevice::new(params.clone()).unwrap())))
                .record(Arc::new(Mutex::new(OpAlgoDispatch::new(algorithm.clone()))))
                .record(Arc::new(Mutex::new(OpTensorSyncLocal::new(params.clone()).unwrap())))
                .eval();

            let a_in = tensor_in_a.vector::<f32>();
            let a_out = tensor_out_a.vector::<f32>();
            info!("tensor_in_a: {:?}", a_in);
            info!("tensor_out_a: {:?}", a_out);
            drop(ManuallyDrop::<Vec<f32>>::into_inner(a));
            drop(ManuallyDrop::<Vec<f32>>::into_inner(zero));
            assert_eq!(a_out, vec![1. as f32, 4. as f32, 9. as f32]);
        }


        #[test]
        fn tensor_copy() {
            std::env::set_var("RUST_BACKTRACE", "1");
            std::env::set_var("RUST_LOG", "trace");
            env_logger::init();
            let mut mgr = Manager::new(0, &[0], &[]);
            let test_vector: Vec<f32> = vec![1., 2., 3., 4.];
            let zero: Vec<f32> = vec![0., 0., 0., 0.];
            let tensorA = mgr.tensor(&test_vector, tensor::TensorTypes::Device).unwrap();
            let tensorB = mgr.tensor(&zero, tensor::TensorTypes::Device).unwrap();
            let tensorC = mgr.tensor(&zero, tensor::TensorTypes::Device).unwrap();
    
            let params = vec![tensorA.clone(), tensorB.clone(), tensorC.clone()];
            mgr.sequence(0, 0)
                .unwrap()
                .lock()
                .as_mut()
                .unwrap()
                .record(Arc::new(Mutex::new(OpTensorSyncDevice::new(vec![tensorA.clone()]).unwrap())))
                .record(Arc::new(Mutex::new(OpTensorCopy::new(params.clone()).unwrap())))
                .record(Arc::new(Mutex::new(OpTensorSyncLocal::new(params.clone()).unwrap())))
                .eval();
    
            let a = tensorA.vector::<f32>();
            let b = tensorB.vector::<f32>();
            let c = tensorC.vector::<f32>();

            debug!("tensorA: {:?}", a);
            debug!("tensorB: {:?}", b);
            debug!("tensorC: {:?}", c);
    
            assert_eq!(a, test_vector);
            assert_eq!(b, test_vector);
            assert_eq!(c, test_vector);
    
        }
    }


}
