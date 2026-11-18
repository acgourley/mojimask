//! ONNX Runtime sessions the way mojimask/onnx_core.session() makes them: free-dimension overrides, the CoreML EP
//! with the engine's ModelFormat + MLComputeUnits and a ModelCacheDirectory, CPU EP fallback; log level error.
//! Gotcha (poc/ml/REPORT.md in the unpublished PoC): ort runs "CPU" sessions on CoreML/ANE in fp16 unless the policy is
//! PreferCPU.
use ort::{
    ep,
    environment::Environment,
    session::Session,
    value::Tensor,
};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ep {
    Cpu,
    Gpu, // MLProgram, CPUAndGPU
    Ane, // NeuralNetwork, CPUAndNeuralEngine
    WebGpu, // ONNX Runtime's WebGPU EP (Dawn; Metal on a Mac). Only with --features webgpu
}

impl Ep {
    pub fn parse(s: &str) -> Ep {
        match s {
            "cpu" => Ep::Cpu,
            "gpu" => Ep::Gpu,
            "ane" => Ep::Ane,
            "webgpu" => Ep::WebGpu,
            _ => panic!("engine {s}"),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Ep::Cpu => "cpu",
            Ep::Gpu => "gpu",
            Ep::Ane => "ane",
            Ep::WebGpu => "webgpu",
        }
    }
}

/// The process's ONNX Runtime environment. ONNX Runtime allows one per process, ever (ort refuses a second build
/// even after the first is dropped), so it is made once and kept: every caller (the detectors' worker threads, the
/// analysis models) shares it.
pub fn env() -> Environment {
    static ENV: std::sync::OnceLock<Environment> = std::sync::OnceLock::new();
    ENV.get_or_init(|| {
        #[cfg(feature = "dynamic")]
        {
            let p = std::env::var("MM_ML_DYLIB").expect("MM_ML_DYLIB: path to libonnxruntime.dylib");
            return ort::init_from(p).expect("load dylib").with_name("mm").build().expect("ort env");
        }
        #[allow(unreachable_code)]
        ort::init().with_name("mm").build().expect("ort env")
    })
    .clone()
}

pub fn session(env: &Environment, model: &Path, which: Ep, dims: &[(String, i64)], cache: Option<&Path>) -> ort::Result<Session> {
    let mut b = Session::builder(env)?.with_log_level(ort::logging::LogLevel::Error)?;
    if let Ok(n) = std::env::var("MM_ML_INTRA") {
        b = b.with_intra_threads(n.parse().unwrap())?;
    }
    for (n, v) in dims {
        b = b.with_dimension_override(n, *v)?;
    }
    if which == Ep::Cpu {
        // ort's SessionBuilder::new sets the EP selection policy to MaxEfficiency, and ONNX Runtime then picks a device
        // itself (on a Mac: CoreML, fp16 on the Neural Engine) -- even with the CPU EP named explicitly (measured:
        // 14 px off on YOLOv5-Face's raw output). PreferCPU makes "cpu" the CPU, bit-identical to Python.
        match std::env::var("MM_ML_CPU_MODE").as_deref() {
            Ok("implicit") => {}
            Ok("explicit") => b = b.with_execution_providers([ep::CPU::default().build().error_on_failure()])?,
            Ok("default_policy") => b = b.with_auto_device(ort::session::builder::AutoDevicePolicy::Default)?,
            _ => b = b.with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)?,
        }
    } else if which == Ep::WebGpu {
        #[cfg(feature = "webgpu")]
        {
            b = b.with_execution_providers([ep::WebGPU::default().build().error_on_failure(), ep::CPU::default().build()])?;
        }
        #[cfg(not(feature = "webgpu"))]
        panic!("built without --features webgpu");
    } else {
        let (fmt, cu) = match which {
            Ep::Gpu => (ep::coreml::ModelFormat::MLProgram, ep::coreml::ComputeUnits::CPUAndGPU),
            _ => (ep::coreml::ModelFormat::NeuralNetwork, ep::coreml::ComputeUnits::CPUAndNeuralEngine),
        };
        if std::env::var("MM_ML_COREML_POLICY").as_deref() == Ok("default") {
            b = b.with_auto_device(ort::session::builder::AutoDevicePolicy::Default)?;
        }
        let mut c = ep::CoreML::default().with_model_format(fmt).with_compute_units(cu);
        if let Some(d) = cache {
            std::fs::create_dir_all(d).ok();
            c = c.with_model_cache_dir(d.display());
        }
        b = b.with_execution_providers([c.build().error_on_failure(), ep::CPU::default().build()])?;
    }
    b.commit_from_file(model)
}

/// Run one f32 input, return every output as (shape, data) in the model's output order.
pub fn run(s: &mut Session, shape: &[usize], data: Vec<f32>) -> ort::Result<Vec<(Vec<usize>, Vec<f32>)>> {
    let names: Vec<String> = s.outputs().iter().map(|o| o.name().to_string()).collect();
    let input = s.inputs()[0].name().to_string();
    let t = Tensor::from_array((shape.to_vec(), data))?;
    let out = s.run(ort::inputs![input => t])?;
    let mut v = Vec::new();
    for n in &names {
        let (sh, d) = out[n.as_str()].try_extract_tensor::<f32>()?;
        v.push((sh.iter().map(|&x| x as usize).collect(), d.to_vec()));
    }
    Ok(v)
}
