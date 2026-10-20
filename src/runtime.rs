//! Explicit Burn device selection; GPU requests never fall back to scalar CPU.

use burn::tensor::{Device, DeviceKind, Tensor};

pub fn device(name: &str) -> Result<Device, String> {
    match name {
        "gpu" => {
            let device = Device::wgpu_options()
                .device_kind(DeviceKind::DiscreteGpu(0))
                .init()
                .map_err(|error| format!("cannot initialize the discrete WGPU device: {error}"))?;
            // Acquire and execute on the selected GPU before reporting success.
            let probe = Tensor::<1>::ones([1], &device) + 1.0;
            let values = probe
                .try_into_vec_as::<f32>()
                .map_err(|error| format!("WGPU execution probe failed: {error}"))?;
            if values != [2.0] {
                return Err("WGPU execution probe returned an incorrect result".into());
            }
            Ok(device)
        }
        "cpu" => Ok(Device::flex()),
        _ => Err("--device must be gpu or cpu".into()),
    }
}
