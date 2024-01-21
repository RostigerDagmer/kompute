use std::path::{Path, PathBuf};

use ash::vk;
use shaderc::{Compiler, ShaderKind, IncludeType, ResolvedInclude, CompileOptions};

pub fn get_sharerc_include(
    requested_source: &str,
    _include_type: IncludeType,
    _origin_source: &str,
    _recursion_depth: usize,
    origin_dir: &Path,
) -> Result<ResolvedInclude, String> {
    //TODO: finish implementation
    let resolved_file = origin_dir.join(requested_source);
    let resolved_name = resolved_file
        // .file_name()
        // .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    //println!("Including: {}", resolved_name);
    let error_msg = format!("Failed to open {}.", resolved_file.to_str().unwrap());
    let content = std::fs::read_to_string(resolved_file.as_path()).expect(&error_msg);
    Ok(ResolvedInclude {
        resolved_name,
        content,
    })
}

pub fn get_shaderc_stage(stage: &vk::ShaderStageFlags) -> Option<ShaderKind> {
    match *stage {
        vk::ShaderStageFlags::VERTEX => Some(ShaderKind::Vertex),
        vk::ShaderStageFlags::FRAGMENT => Some(ShaderKind::Fragment),
        vk::ShaderStageFlags::COMPUTE => Some(ShaderKind::Compute),
        vk::ShaderStageFlags::TESSELLATION_CONTROL => Some(ShaderKind::TessControl),
        vk::ShaderStageFlags::TESSELLATION_EVALUATION => Some(ShaderKind::TessEvaluation),
        vk::ShaderStageFlags::GEOMETRY => Some(ShaderKind::Geometry),
        vk::ShaderStageFlags::RAYGEN_KHR => Some(ShaderKind::RayGeneration),
        vk::ShaderStageFlags::ANY_HIT_KHR => Some(ShaderKind::AnyHit),
        vk::ShaderStageFlags::CLOSEST_HIT_KHR => Some(ShaderKind::ClosestHit),
        vk::ShaderStageFlags::MISS_KHR => Some(ShaderKind::Miss),
        vk::ShaderStageFlags::INTERSECTION_KHR => Some(ShaderKind::Intersection),
        _ => None,
    }
}

pub fn get_spirv_filepath(path: &PathBuf) -> PathBuf {
    let mut compiled_path = path.clone();
    let filename = compiled_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
        + ".spv";
    compiled_path.set_file_name(filename);
    compiled_path
}

pub fn is_more_recent(path: &PathBuf, other: &PathBuf) -> bool {
    let timestamp = std::fs::metadata(path.as_path()).unwrap().modified().unwrap();
    let other_timestamp = std::fs::metadata(other.as_path()).unwrap().modified().unwrap();
    timestamp > other_timestamp
}

pub fn compile_code(code: String, stage: ShaderKind) -> Result<Vec<u32>, std::io::Error> {
    let compiler = Compiler::new().unwrap();
    let mut options = CompileOptions::new().unwrap();
    options.set_generate_debug_info();
    options.set_target_spirv(shaderc::SpirvVersion::V1_5);
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
    let code = compiler
        .compile_into_spirv(
            &code,
            stage,
            "inline_shader",
            "main",
            Some(&options),
        )
        .unwrap();
    Ok(code.as_binary().to_owned())
}

pub fn compile_shader(path: PathBuf) -> Result<Vec<u32>, std::io::Error> {
    let extension = path.extension().unwrap().to_str().unwrap();

    if extension == "spv" {
        // The file is already a compiled SPIR-V file, read and return its contents
        // Assuming you have a function to read the SPIR-V file and return Vec<u32>
        let mut file = std::fs::File::open(path).unwrap();
        ash::util::read_spv(&mut file)
    } else {
        // The file is a source code file, compile it using shaderc
        let error_msg = format!("Failed to open {}.", path.to_str().unwrap());
        let source = std::fs::read_to_string(path.as_path()).expect(&error_msg);

        let compiler = Compiler::new().unwrap();
        let mut options = CompileOptions::new().unwrap();
        options.set_generate_debug_info();
        options.set_target_spirv(shaderc::SpirvVersion::V1_6);
        options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
        let origin_path = path.clone();
        options.set_include_callback(
            move |requested_source, include_type, origin_source, recursion_depth| {
                get_sharerc_include(
                    requested_source,
                    include_type,
                    origin_source,
                    recursion_depth,
                    origin_path.parent().unwrap(),
                )
            },
        );
        let stage_flags = vk::ShaderStageFlags::COMPUTE;
        let sc_stage = get_shaderc_stage(&stage_flags).unwrap();
        let code = compiler
            .compile_into_spirv(
                &source,
                sc_stage,
                path.file_name().unwrap().to_str().unwrap(),
                "main",
                Some(&options),
            )
            .unwrap();
        Ok(code.as_binary().to_owned())
    }
}

pub enum ShaderSource {
    SPIRV(Vec<u32>),
    Source(PathBuf),
    Code(String),
}