use std::{sync::{Arc, Mutex}, borrow::{BorrowMut, Borrow}, rc::Rc, ops::Deref};
use ash::{vk, Entry, Instance};
use log::{debug, info, warn, error};
use crate::operations::OpBase;

#[derive(Clone)]
pub struct Sequence {
    physical_device: Arc<vk::PhysicalDevice>,
    device: Arc<ash::Device>,
    compute_queue: Arc<vk::Queue>,
    queue_index: u32,
    command_pool: Option<Arc<vk::CommandPool>>,
    command_buffer: Option<Arc<vk::CommandBuffer>>,
    timestamp_query_pool: Option<Arc<vk::QueryPool>>,
    recording: bool,
    running: bool,
    operations: Vec<Arc<Mutex<dyn OpBase>>>,
    fence: Option<Arc<vk::Fence>>,
}

impl Sequence {
    pub fn new(
        entry: Arc<Entry>,
        instance: Arc<Instance>,
        physical_device: Arc<vk::PhysicalDevice>,
        device: Arc<ash::Device>,
        compute_queue: Arc<vk::Queue>,
        queue_index: u32,
        total_timestamps: u32,
    ) -> Self {
        debug!("Kompute Sequence Constructor with existing device & queue");

        let mut sequence = Sequence {
            physical_device,
            device,
            compute_queue,
            queue_index,
            command_pool: None,
            command_buffer: None,
            timestamp_query_pool: None,
            recording: false,
            running: false,
            operations: Vec::new(),
            fence: None,
        };

        sequence.create_command_pool();
        sequence.create_command_buffer();
        if total_timestamps > 0 {
            sequence.create_timestamp_query_pool(entry, instance, total_timestamps + 1);
        }

        sequence
    }

    pub fn begin(&mut self) {
        debug!("Kompute sequence called BEGIN");

        if self.is_recording() {
            debug!("Kompute Sequence begin called when already recording");
            return;
        }

        if self.is_running() {
            panic!("Kompute Sequence begin called when sequence still running");
        }

        info!("Kompute Sequence command now started recording");
        unsafe { self.device.begin_command_buffer(self.command_buffer.as_ref().unwrap().as_ref().to_owned(), &vk::CommandBufferBeginInfo::default()) };
        self.recording = true;

        // latch the first timestamp before any commands are submitted
        if let Some(timestamp_query_pool) = &self.timestamp_query_pool {
            unsafe { self.device.cmd_write_timestamp(self.command_buffer.as_ref().unwrap().as_ref().to_owned(), vk::PipelineStageFlags::ALL_COMMANDS, timestamp_query_pool.to_owned().as_ref().to_owned(), 0) }
        }
    }

    pub fn end(&mut self) {
        debug!("Kompute Sequence calling END");

        if self.is_running() {
            panic!("Kompute Sequence begin called when sequence still running");
        }

        if !self.is_recording() {
            warn!("Kompute Sequence end called when not recording");
            return;
        } else {
            info!("Kompute Sequence command recording END");
            unsafe { self.device.end_command_buffer(self.command_buffer.as_ref().unwrap().as_ref().to_owned()) };
            self.recording = false;
        }
    }

    pub fn clear(&mut self) {
        debug!("Kompute Sequence calling clear");
        self.operations.clear();
        if self.is_recording() {
            self.end();
        }
    }

    pub fn eval(&mut self) -> Self {
        debug!("Kompute sequence EVAL BEGIN");

        self.eval_async().eval_await(4000)
    }

    pub fn eval_with_op(&mut self, op: Arc<Mutex<dyn OpBase>>) -> Self {
        self.clear();
        self.record(op);
        self.eval()
    }

    pub fn eval_async(&mut self) -> Self {
        if self.is_recording() {
            self.end();
        }

        if self.running {
            panic!("Kompute Sequence evalAsync called when an eval async was called without successful wait");
        }

        self.running = true;

        for operation in &self.operations {
            let op = operation.lock().unwrap();
            op.borrow().pre_eval(self.command_buffer.as_ref().unwrap().as_ref().to_owned());
        }

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&[self.command_buffer.as_ref().unwrap().as_ref().to_owned()])
            .build();

        self.fence = Some(Arc::new(unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap() }));

        info!("Kompute sequence submitting command buffer into compute queue");

        let res = unsafe { self.device.queue_submit(self.compute_queue.as_ref().to_owned(), &[submit_info], self.fence.as_ref().unwrap().as_ref().to_owned()) };
        match res {
            Ok(()) => info!("Kompute sequence submitted command buffer into compute queue"),
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                panic!("Kompute sequence submitted command buffer into compute queue but device lost");
            },
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                panic!("Kompute sequence submitted command buffer into compute queue but out of host memory");
            },
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                panic!("Kompute sequence submitted command buffer into compute queue but out of device memory");
            },
            Err(e) => {
                panic!("Kompute sequence submitted command buffer error: {:?}", e);
            },

        }
        self.to_owned()
    }

    pub fn eval_async_with_op(&mut self, op: Arc<Mutex<dyn OpBase>>) -> Self {
        self.clear();
        self.record(op);
        self.eval_async()
    }

    pub fn eval_await(&mut self, wait_for: u64) -> Self {
        if !self.running {
            warn!("Kompute Sequence evalAwait called without existing eval");
            return self.clone();
        }

        let result = unsafe { self.device.wait_for_fences(&[self.fence.as_ref().unwrap().as_ref().to_owned()], true, wait_for) };
        match result {
            Err(vk::Result::TIMEOUT) => {
                error!("Kompute Sequence evalAwait fence wait timeout");
              return self.clone();
            }
            _ => info!("Kompute Sequence evalAwait fence wait finished")
        }
        unsafe { self.device.destroy_fence(self.fence.take().unwrap().as_ref().to_owned(), None) };

        self.running = false;

        match result {
            Err(vk::Result::TIMEOUT) => {
                warn!("Kompute Sequence evalAwait fence wait timeout");
              return self.clone();
            }
            _ => info!("Kompute Sequence evalAwait fence wait success")
        }

        for operation in &self.operations {
            let op = operation.lock().unwrap();
            op.borrow().post_eval(self.command_buffer.as_ref().unwrap().as_ref().to_owned());
        }

        self.clone()
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    pub fn is_init(&self) -> bool {
        self.command_pool.is_some()
        && self.command_buffer.is_some()
    }

    pub fn rerecord(&mut self) {
        self.end();
        let ops = std::mem::take(&mut self.operations);
        self.operations.clear();
        for op in ops {
            self.record(op);
        }
    }

    pub fn destroy(&mut self) {
        debug!("Kompute Sequence destroy called");

        if let Some(command_buffer) = &self.command_buffer {
                info!("Freeing CommandBuffer");
                unsafe { self.device
                    .free_command_buffers(self.command_pool.as_ref().unwrap().as_ref().to_owned(), &[*command_buffer.to_owned()]) };
                self.command_buffer = None;
                debug!("Kompute Sequence Freed CommandBuffer");
        }

        if let Some(command_pool) = &self.command_pool {
                info!("Destroying CommandPool");
                unsafe { self.device.destroy_command_pool(*command_pool.as_ref(), None) };
                self.command_pool = None;
                debug!("Kompute Sequence Destroyed CommandPool");
        }

        if !self.operations.is_empty() {
            info!("Kompute Sequence clearing operations buffer");
            self.operations.clear();
        }

        if let Some(timestamp_query_pool) = &self.timestamp_query_pool {
            info!("Destroying QueryPool");
            unsafe { self.device
                .destroy_query_pool(*timestamp_query_pool.as_ref(), None) };
            self.timestamp_query_pool = None;
            debug!("Kompute Sequence Destroyed QueryPool");
        }

    }

    pub fn record(&mut self, op: Arc<Mutex<dyn OpBase>>) -> Self {
        debug!("Kompute Sequence record function started");

        self.begin();

        debug!("Kompute Sequence running record on OpBase derived class instance");
        let mut guard = op.lock().unwrap();
        guard.borrow_mut().record(*self.command_buffer.as_ref().unwrap().as_ref());
        drop(guard);

        self.operations.push(op);
        if let Some(timestamp_query_pool) = &self.timestamp_query_pool {
            unsafe { self.device.cmd_write_timestamp(
                *self.command_buffer.as_ref().unwrap().as_ref(),
                vk::PipelineStageFlags::ALL_COMMANDS,
                *timestamp_query_pool.to_owned().as_ref(),
                self.operations.len() as u32,
            )}
        }

        self.clone()
    }

    fn create_command_pool(&mut self) {
        debug!("Kompute Sequence creating command pool");

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.queue_index);

        self.command_pool = Some(Arc::new(
            unsafe { self.device
                .create_command_pool(&command_pool_info, None)
                .unwrap() },
            )
        );

        debug!("Kompute Sequence Command Pool Created");
    }

    fn create_command_buffer(&mut self) {
        debug!("Kompute Sequence creating command buffer");

        if self.command_pool.is_none() {
            panic!("Kompute Sequence command pool is null");
        }

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*self.command_pool.as_ref().unwrap().as_ref())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        self.command_buffer = Some(
            Arc::new(
            unsafe { self.device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap() }[0],
            )
        );

        debug!("Kompute Sequence Command Buffer Created");
    }

    fn create_timestamp_query_pool(&mut self, entry: Arc<Entry>, instance: Arc<Instance>, total_timestamps: u32) {
        debug!("Kompute Sequence creating query pool");

        if !self.is_init() {
            panic!("createTimestampQueryPool() called on uninitialized Sequence");
        }

        let physical_device_properties = unsafe {
            // little problem here: where do we get instance? 
            instance.get_physical_device_properties(
                *self.physical_device.as_ref(),
            )
        };

        if physical_device_properties.limits.timestamp_compute_and_graphics > 0 {
            let query_pool_info = vk::QueryPoolCreateInfo::builder()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(total_timestamps);

            self.timestamp_query_pool = Some(
                Arc::new(
                unsafe { self.device
                    .create_query_pool(&query_pool_info, None)
                    .unwrap() },
                )
            );

            debug!("Query pool for timestamps created");
        } else {
            panic!("Device does not support timestamps");
        }
    }

    pub fn get_timestamps(&self) -> Vec<u64> {
        if self.timestamp_query_pool.is_none() {
            panic!("Timestamp latching not enabled");
        }

        let n = self.operations.len() + 1;
        let mut timestamps = vec![0u64; n];
        unsafe { self.device
            .get_query_pool_results(
                *self.timestamp_query_pool.as_ref().unwrap().as_ref(),
                0,
                n as u32,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
            .unwrap() };

        timestamps
    }
}

// Implement OpBase for your OpBase struct
