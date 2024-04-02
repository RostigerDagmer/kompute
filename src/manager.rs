use ash::extensions::ext::DebugUtils;
use ash::vk::Handle;
use ash::{vk, Device, Entry, Instance};
use log::debug;
use num::Num;
use std::any::Any;
use std::error::Error;
use std::ffi::{CString, c_void, c_char};
use std::sync::{Arc, Mutex, Weak};

use crate::algorithm::Algorithm;
use crate::sequence::Sequence;
use crate::shaderutil::ShaderSource;
use crate::shape::{ConstShape, Shape};
use crate::tensor::{RawTensor, Tensor, TensorData, TensorDataTypes, TensorT, TensorTypes};

// Assuming you have a logger setup, otherwise configure it according to your needs.
#[cfg(feature = "debug_layer")]
unsafe extern "system" fn debug_message_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT, p_layer_prefix: vk::DebugUtilsMessageTypeFlagsEXT, p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {

    let layer_prefix = match p_layer_prefix {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[GENERAL]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
        _ => "[UNKNOWN]",
    };

    let message = unsafe {
        let data = *p_callback_data;
        format!(
            "{:?}",
            std::ffi::CStr::from_ptr(data.p_message)
        )
    };

    // Here you should use your logging facility, this is just a placeholder
    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("{}: {}", layer_prefix, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("{}: {}", layer_prefix, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("{}: {}", layer_prefix, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::debug!("{}: {}", layer_prefix, message),
        _ => log::debug!("{}: {}", layer_prefix, message),
    }
    vk::FALSE
}


pub struct Manager {
    // Vulkan-related fields
    entry: Arc<Entry>,
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    compute_queues: Vec<Arc<Mutex<vk::Queue>>>,
    family_queue_indices: Vec<u32>,

    // Resource management fields
    managed_sequences: Vec<Weak<Mutex<Sequence>>>,
    managed_algorithms: Vec<Weak<Mutex<Algorithm>>>,

    // Debugging and validation layers
    #[cfg(feature = "debug_layer")]
    debug_report_loader: Option<DebugUtils>,
    #[cfg(feature = "debug_layer")]
    debug_report_callback: Option<vk::DebugUtilsMessengerEXT>,
}

#[cfg(feature = "debug_layer")]
unsafe fn create_instance(
    desired_extensions: &[&str],
) -> Result<
    (
        ash::Entry,
        ash::Instance,
        Vec<*const i8>,
        Option<DebugUtils>,
        Option<vk::DebugUtilsMessengerEXT>,
    ),
    Box<dyn Error>,
> {
    __create_instance(desired_extensions)
}

#[cfg(not(feature = "debug_layer"))]
unsafe fn create_instance(
    desired_extensions: &[&str],
) -> Result<(ash::Entry, ash::Instance, Vec<*const i8>), Box<dyn Error>> {
    __create_instance(desired_extensions).map(|(entry, instance, extensions, _, _)| (entry, instance, extensions))
}

unsafe fn __create_instance(
    desired_extensions: &[&str],
) -> Result<
    (
        ash::Entry,
        ash::Instance,
        Vec<*const i8>,
        Option<DebugUtils>,
        Option<vk::DebugUtilsMessengerEXT>,
    ),
    Box<dyn Error>,
> {
    let entry = Entry::load()?;

    let app_name = CString::new("Kompute Algorithm")?;
    let engine_name = CString::new("Kompute Driver")?;

    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app_name)
        .engine_name(&engine_name)
        .api_version(vk::API_VERSION_1_2);

    // Initialize extensions with additional debug report extension if in debug layer mode.
    let mut extension_names_cstrings = desired_extensions
        .iter()
        .map(|&extension| CString::new(extension).unwrap())
        .collect::<Vec<_>>();

    #[cfg(feature = "debug_layer")]
    {
        let debug_report_extension = CString::new(DebugUtils::name().to_str().unwrap()).unwrap();
        extension_names_cstrings.push(debug_report_extension);
    }

    let extension_ptrs = extension_names_cstrings
        .iter()
        .map(|extension| extension.as_ptr())
        .collect::<Vec<_>>();

    // Create InstanceCreateInfo with extensions and, optionally, layer names
    let mut create_info_builder = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&extension_ptrs);

    // Debug layer configuration
    #[cfg(feature = "debug_layer")]
    let mut layer_names_cstrings = Vec::new();
    #[cfg(feature = "debug_layer")]
    let layer_ptrs = {
        // Check for validation layers from environment variable (KOMPUTE_ENV_DEBUG_LAYERS)
        if let Ok(env_layers) = std::env::var("KOMPUTE_ENV_DEBUG_LAYERS") {
            debug!("Debug layers from environment variable: {:?}", env_layers);
            for layer in env_layers.split(',') {
                layer_names_cstrings.push(CString::new(layer.trim()).unwrap());
            }
        }
        
        // Add any default layers
        // ...

        if !layer_names_cstrings.is_empty() {
            layer_names_cstrings
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    };
    
    #[cfg(feature = "debug_layer")]
    let create_info = create_info_builder.enabled_layer_names(layer_ptrs.as_slice()).build();
    #[cfg(not(feature = "debug_layer"))]
    let create_info = create_info_builder.build();
    
    // Create Vulkan instance
    let instance = entry.create_instance(&create_info, None)?;
    // Debug report callback creation
    let (debug_report_loader, debug_callback) = {
        #[cfg(feature = "debug_layer")]
        {
            debug!("Debug extension: {:?}", ash::extensions::ext::DebugUtils::name());
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .pfn_user_callback(Some(debug_message_callback));

            let debug_report_loader = DebugUtils::new(&entry, &instance);
            let callback = debug_report_loader.create_debug_utils_messenger(&debug_info.build(), None)?;
            (Some(debug_report_loader), Some(callback))
        }
        #[cfg(not(feature = "debug_layer"))]
        {
            (None, None)
        }
    };

    Ok((
        entry,
        instance,
        extension_ptrs,
        debug_report_loader,
        debug_callback,
    ))
}

impl Manager {
    pub fn sequence(&mut self, queue_index: u32, total_timestamps: u32) -> Result<Arc<Mutex<Sequence>>, Box<dyn Error>> {
        // Implementation goes here
        // extract and clone compute queue at queue index out of mutex

        let guard = self.compute_queues[queue_index as usize].lock().unwrap();
        let queue = guard.clone();
        drop(guard);

        let s = Arc::new(
            Mutex::new(
            Sequence::new(
                self.entry.clone(),
                self.instance.clone(),
                Arc::new(self.physical_device), 
                self.device.clone(),
                Arc::new(queue),
                queue_index,
                total_timestamps)
            ));
        self.managed_sequences.push(Arc::downgrade(&s));
        Ok(s)
    }

    // pub fn tensor_t<T: TensorData>(&self, data: &[T], tensor_type: TensorTypes) -> Result<Arc<TensorT<T>>, Box<dyn Error>> {
    //     let t: TensorT<T> = TensorT::new(Arc::new(self.physical_device), self.device, data, tensor_type);
    //     let t = Arc::new(t.into());
    //     Ok(t)
    // }

    pub fn tensor<T: TensorData, S: Shape>(&mut self, data: &[T], tensor_type: TensorTypes) -> Result<Tensor<S>, Box<dyn Error>> {
        // cast data to *mut c_void
        let len = data.len();
        let t = TensorDataTypes::from(data);
        let data = data.as_ptr() as *mut c_void;
        let mut t_ = Tensor::new(Arc::new(self.physical_device), self.device.clone(), data, len as u32, std::mem::size_of::<T>() as u32, t, tensor_type);
        t_.rebuild(self.entry.clone(), self.instance.clone(), data, len as u32, std::mem::size_of::<T>() as u32);
        Ok(t_)
    }

    pub fn tensor_man<T, S: Shape>(&mut self, data: &[T], element_total_count: usize, element_memory_size: usize, data_type: TensorDataTypes, tensor_type: TensorTypes) -> Result<Tensor<S>, Box<dyn Error>> {
        // cast data to *mut c_void
        let data = data.as_ptr() as *mut c_void;
        let mut t_ = Tensor::new(Arc::new(self.physical_device), self.device.clone(), data, element_total_count as u32, element_memory_size as u32, data_type, tensor_type);
        t_.rebuild(self.entry.clone(), self.instance.clone(), data, element_total_count as u32, element_memory_size as u32);
        Ok(t_)
    }

    pub fn tensor_raw<S: Shape>(
        &mut self,
        data: *mut c_void,
        element_total_count: u32,
        element_memory_size: u32,
        data_type: TensorDataTypes,
        tensor_type: TensorTypes,
    ) -> Result<Tensor<S>, Box<dyn Error>> {
        let mut t_ = Tensor::new(Arc::new(self.physical_device), self.device.clone(), data, element_total_count, element_memory_size, data_type, tensor_type);
        t_.rebuild(self.entry.clone(), self.instance.clone(), data, element_total_count, element_memory_size);
        Ok(t_)
    }

    pub fn algorithm<T: TensorData, D: TensorData>(
        &mut self,
        tensors: Vec<Arc<Mutex<RawTensor>>>,
        spirv: ShaderSource,
        workgroup: &[u32; 3],
        specialization_constants: Vec<T>,
        push_constants: Vec<D>,
    ) -> Result<Arc<Mutex<Algorithm>>, Box<dyn Error>> {
        // Implementation goes here
        let mut a_ = Algorithm::new(self.device.clone(), tensors, spirv, *workgroup, specialization_constants, push_constants);
        a_.init();
        let a = Arc::new(Mutex::new(a_));
        self.managed_algorithms.push(Arc::downgrade(&a));
        Ok(a)
    }

    pub fn clear(&mut self) {
        self.managed_algorithms = Vec::new();
        self.managed_sequences = Vec::new();
    }

    pub fn get_device_properties(&self) -> vk::PhysicalDeviceProperties {
        unsafe { self.instance.get_physical_device_properties(self.physical_device) }
    }

    pub unsafe fn list_devices(&self) -> Vec<vk::PhysicalDevice> {
        self.instance.enumerate_physical_devices().unwrap()
    }
    pub fn new(
        physical_device_index: u32,
        family_queue_indices: &[u32],
        desired_extensions: &[&str],
    ) -> Self {
        // ... Initialization of Vulkan Instance ...
        #[cfg(feature = "debug_layer")]
        let (entry, instance, extensions, debug_report_loader, debug_callback) =
            unsafe { create_instance(desired_extensions).unwrap() };
        #[cfg(not(feature = "debug_layer"))]
        let (entry, instance, extensions) = unsafe { create_instance(desired_extensions).unwrap() };

        // Physical Device selection
        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        
        let physical_device = physical_devices[physical_device_index as usize];
        debug!("Physical device: {:?}", physical_device);

        let queue_create_infos = family_queue_indices
            .iter()
            .map(|&index| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(index)
                    .queue_priorities(&[1.0])
                    .build()
            })
            .collect::<Vec<_>>();

        // ... Device creation with queue creation ...
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos);
            // .enabled_extension_names(&extensions);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };

        // ... Fetch compute queues and add to compute_queues vector ...
        let compute_queues = unsafe {
            family_queue_indices
                .iter()
                .map(|&index| {
                    let queue = device.get_device_queue(index, 0);
                    Arc::new(Mutex::new(queue))
                })
                .collect::<Vec<_>>()
        };

        // Initialize the Manager struct and return
        Self {
            entry: Arc::new(entry),
            instance: Arc::new(instance),
            #[cfg(feature = "debug_layer")]
            debug_report_loader: debug_report_loader,
            #[cfg(feature = "debug_layer")]
            debug_report_callback: debug_callback,
            physical_device,
            device: Arc::new(device),
            family_queue_indices: family_queue_indices.to_vec(),
            compute_queues,
            managed_sequences: Vec::new(),
            managed_algorithms: Vec::new(),
        }
    }
}

impl Drop for Manager {
    fn drop(&mut self) {
        // Clean up Vulkan resources, destroy debug report callback, devices etc.

        // Only destroy the debug callback if debug layers are enabled
        #[cfg(feature = "debug_layer")]
        if let Some(callback) = self.debug_report_callback.take() {
            if let Some(ref report) = self.debug_report_loader {
                unsafe {
                    report.destroy_debug_utils_messenger(callback, None);
                }
            }
        }

        // Destroy the Vulkan device
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
