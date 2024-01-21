use ash::vk;

pub trait OpBase {
    fn record(&mut self, command_buffer: vk::CommandBuffer);
    fn pre_eval(&self, command_buffer: vk::CommandBuffer);
    fn post_eval(&self, command_buffer: vk::CommandBuffer);
}
