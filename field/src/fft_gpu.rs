//! GPU-accelerated FFT using zeknox CUDA library.
//!
//! This module provides GPU-accelerated NTT (Number Theoretic Transform) operations
//! for the Goldilocks field using the zeknox CUDA library.

extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use std::sync::Mutex;

use once_cell::sync::OnceCell;
use plonky2_util::log2_strict;
use zeknox::types::{NTTConfig, NTTInputOutputOrder, NTTType};
use zeknox::{init_twiddle_factors_rs, init_coset_rs, ntt_batch, intt_batch, lde_batch};

use crate::goldilocks_field::GoldilocksField;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::types::{Field, PrimeField64};

// Track GPU initialization state
static GPU_INITIALIZED: OnceCell<Mutex<bool>> = OnceCell::new();

fn get_gpu_id() -> i32 {
    std::env::var("ZEKNOX_GPU_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Initialize GPU with twiddle factors and coset for common sizes.
/// This is called once on first use.
fn ensure_gpu_initialized() {
    let state = GPU_INITIALIZED.get_or_init(|| Mutex::new(false));
    let mut initialized = state.lock().unwrap();

    if !*initialized {
        let gpu_id = get_gpu_id() as usize;
        let coset_gen = GoldilocksField::coset_shift().to_canonical_u64();

        // Initialize coset factors for large size (covers all smaller sizes)
        init_coset_rs(gpu_id, 24, coset_gen);

        // Initialize twiddle factors for common sizes
        for lg_n in 2..=24 {
            init_twiddle_factors_rs(gpu_id, lg_n);
        }

        *initialized = true;
    }
}

/// Initialize twiddle factors for a given log size if not already initialized.
fn ensure_twiddle_factors(lg_n: usize) {
    ensure_gpu_initialized();
}

/// Initialize coset factors for a given log size if not already initialized.
fn ensure_coset_factors(lg_n: usize) {
    ensure_gpu_initialized();
}

/// Minimum size for GPU acceleration to be beneficial.
/// Below this threshold, CPU is faster due to GPU transfer overhead.
const MIN_GPU_SIZE: usize = 1 << 14; // 16K elements

/// Check if GPU FFT should be used based on input size.
pub fn should_use_gpu(n: usize) -> bool {
    n >= MIN_GPU_SIZE
}

/// GPU-accelerated FFT for Goldilocks field.
/// Falls back to CPU for small inputs.
pub fn fft_gpu(poly: PolynomialCoeffs<GoldilocksField>) -> PolynomialValues<GoldilocksField> {
    let n = poly.len();

    if !should_use_gpu(n) {
        return crate::fft::fft(poly);
    }

    let lg_n = log2_strict(n);
    ensure_twiddle_factors(lg_n);

    let gpu_id = get_gpu_id() as usize;
    let mut buffer = poly.coeffs;

    // Perform NTT on GPU
    let cfg = NTTConfig {
        batches: 1,
        order: NTTInputOutputOrder::NN,
        ntt_type: NTTType::Standard,
        extension_rate_bits: 0,
        are_inputs_on_device: false,
        are_outputs_on_device: false,
        with_coset: false,
        is_multi_gpu: false,
        salt_size: 0,
    };

    ntt_batch(gpu_id, buffer.as_mut_ptr(), lg_n, cfg);

    PolynomialValues::new(buffer)
}

/// GPU-accelerated inverse FFT for Goldilocks field.
/// Falls back to CPU for small inputs.
pub fn ifft_gpu(poly: PolynomialValues<GoldilocksField>) -> PolynomialCoeffs<GoldilocksField> {
    let n = poly.len();

    if !should_use_gpu(n) {
        return crate::fft::ifft(poly);
    }

    let lg_n = log2_strict(n);
    ensure_twiddle_factors(lg_n);

    let gpu_id = get_gpu_id() as usize;
    let n_inv = GoldilocksField::inverse_2exp(lg_n);

    let mut buffer = poly.values;

    // Perform inverse NTT on GPU
    let cfg = NTTConfig {
        batches: 1,
        order: NTTInputOutputOrder::NN,
        ntt_type: NTTType::Standard,
        extension_rate_bits: 0,
        are_inputs_on_device: false,
        are_outputs_on_device: false,
        with_coset: false,
        is_multi_gpu: false,
        salt_size: 0,
    };

    intt_batch(gpu_id, buffer.as_mut_ptr(), lg_n, cfg);

    // Scale by n^-1 (GPU INTT doesn't include this scaling)
    for coeff in &mut buffer {
        *coeff *= n_inv;
    }

    PolynomialCoeffs::new(buffer)
}

/// GPU-accelerated batched FFT for multiple polynomials.
/// This is more efficient than individual FFTs due to better GPU utilization.
pub fn fft_batch_gpu(polys: Vec<PolynomialCoeffs<GoldilocksField>>) -> Vec<PolynomialValues<GoldilocksField>> {
    if polys.is_empty() {
        return Vec::new();
    }

    let n = polys[0].len();
    let num_polys = polys.len();

    // For small inputs or few polynomials, use CPU
    if !should_use_gpu(n) || num_polys < 4 {
        return polys.into_iter().map(|p| crate::fft::fft(p)).collect();
    }

    let lg_n = log2_strict(n);
    ensure_twiddle_factors(lg_n);

    let gpu_id = get_gpu_id() as usize;

    // Flatten all polynomials into a single buffer
    let mut buffer: Vec<GoldilocksField> = Vec::with_capacity(n * num_polys);
    for poly in polys {
        buffer.extend(poly.coeffs);
    }

    // Perform batched NTT on GPU
    let cfg = NTTConfig {
        batches: num_polys as u32,
        order: NTTInputOutputOrder::NN,
        ntt_type: NTTType::Standard,
        extension_rate_bits: 0,
        are_inputs_on_device: false,
        are_outputs_on_device: false,
        with_coset: false,
        is_multi_gpu: false,
        salt_size: 0,
    };

    ntt_batch(gpu_id, buffer.as_mut_ptr(), lg_n, cfg);

    // Split buffer back into individual polynomials
    buffer
        .chunks(n)
        .map(|chunk| PolynomialValues::new(chunk.to_vec()))
        .collect()
}

/// GPU-accelerated batched inverse FFT for multiple polynomials.
pub fn ifft_batch_gpu(polys: Vec<PolynomialValues<GoldilocksField>>) -> Vec<PolynomialCoeffs<GoldilocksField>> {
    if polys.is_empty() {
        return Vec::new();
    }

    let n = polys[0].len();
    let num_polys = polys.len();

    // For small inputs or few polynomials, use CPU
    if !should_use_gpu(n) || num_polys < 4 {
        return polys.into_iter().map(|p| crate::fft::ifft(p)).collect();
    }

    let lg_n = log2_strict(n);
    ensure_twiddle_factors(lg_n);

    let gpu_id = get_gpu_id() as usize;
    let n_inv = GoldilocksField::inverse_2exp(lg_n);

    // Flatten all polynomials into a single buffer
    let mut buffer: Vec<GoldilocksField> = Vec::with_capacity(n * num_polys);
    for poly in polys {
        buffer.extend(poly.values);
    }

    // Perform batched inverse NTT on GPU
    let cfg = NTTConfig {
        batches: num_polys as u32,
        order: NTTInputOutputOrder::NN,
        ntt_type: NTTType::Standard,
        extension_rate_bits: 0,
        are_inputs_on_device: false,
        are_outputs_on_device: false,
        with_coset: false,
        is_multi_gpu: false,
        salt_size: 0,
    };

    intt_batch(gpu_id, buffer.as_mut_ptr(), lg_n, cfg);

    // Scale by n^-1 and split buffer back into individual polynomials
    buffer
        .chunks_mut(n)
        .map(|chunk| {
            for coeff in chunk.iter_mut() {
                *coeff *= n_inv;
            }
            PolynomialCoeffs::new(chunk.to_vec())
        })
        .collect()
}

/// GPU-accelerated batched LDE with coset FFT.
/// This combines polynomial extension and coset FFT into a single GPU operation.
/// Returns evaluation of each polynomial on the coset shift*H where H is the larger subgroup.
pub fn lde_batch_coset_gpu(
    polynomials: &[PolynomialCoeffs<GoldilocksField>],
    rate_bits: usize,
) -> Vec<Vec<GoldilocksField>> {
    if polynomials.is_empty() {
        return Vec::new();
    }

    let degree = polynomials[0].len();
    let num_polys = polynomials.len();
    let output_size = degree << rate_bits;

    // For small inputs, use CPU
    if !should_use_gpu(output_size) || num_polys < 2 {
        return polynomials
            .iter()
            .map(|p| {
                p.lde(rate_bits)
                    .coset_fft_with_options(GoldilocksField::coset_shift(), Some(rate_bits), None)
                    .values
            })
            .collect();
    }

    let lg_input = log2_strict(degree);
    let lg_output = lg_input + rate_bits;

    // Initialize twiddle and coset factors for output size
    ensure_twiddle_factors(lg_output);
    ensure_coset_factors(lg_output);

    let gpu_id = get_gpu_id() as usize;

    // Flatten input polynomials (no zero padding - lde_batch handles it via extension_rate_bits)
    let total_num_input_elements = num_polys * degree;
    let total_num_output_elements = num_polys * output_size;

    let mut input_buffer: Vec<GoldilocksField> = Vec::with_capacity(total_num_input_elements);
    for poly in polynomials {
        input_buffer.extend_from_slice(&poly.coeffs);
    }

    // Allocate output buffer
    let mut output_buffer: Vec<GoldilocksField> = vec![GoldilocksField::ZERO; total_num_output_elements];

    // Configure LDE with coset - matching okx/plonky2 configuration
    let mut cfg = NTTConfig::default();
    cfg.batches = num_polys as u32;
    cfg.extension_rate_bits = rate_bits as u32;
    cfg.are_inputs_on_device = false;
    cfg.are_outputs_on_device = false;
    cfg.with_coset = true;
    cfg.is_multi_gpu = false;
    cfg.salt_size = 0;

    // Perform batched LDE on GPU
    lde_batch(
        gpu_id,
        output_buffer.as_mut_ptr(),
        input_buffer.as_mut_ptr(),
        lg_input,
        cfg,
    );

    // Split output back into individual polynomials
    output_buffer
        .chunks(output_size)
        .map(|chunk: &[GoldilocksField]| chunk.to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Sample, PrimeField64};

    #[test]
    fn test_gpu_fft_roundtrip() {
        let n = 1 << 16; // 64K elements
        let coeffs: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField::from_canonical_u64(i as u64))
            .collect();
        let poly = PolynomialCoeffs::new(coeffs.clone());

        let values = fft_gpu(poly);
        let recovered = ifft_gpu(values);

        assert_eq!(coeffs, recovered.coeffs);
    }

    #[test]
    fn test_gpu_fft_matches_cpu() {
        use plonky2_util::reverse_index_bits_in_place;

        // Use small size for easier debugging
        let n = 1 << 4; // 16 elements

        // Generate simple test data
        let coeffs: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField::from_canonical_u64(i as u64))
            .collect();
        let v1: Vec<u64> = coeffs.iter().map(|x| x.to_canonical_u64()).collect();

        // CPU version: use plonky2 FFT
        let poly_cpu = PolynomialCoeffs::new(coeffs.clone());
        let values_cpu = crate::fft::fft(poly_cpu);
        let cpu_results: Vec<u64> = values_cpu
            .values
            .iter()
            .map(|x| x.to_canonical_u64())
            .collect();

        // GPU version: use zeknox ntt_batch directly with u64
        ensure_gpu_initialized();
        let gpu_id = get_gpu_id() as usize;
        let lg_n = log2_strict(n);

        println!("Input: {:?}", v1);
        println!("CPU FFT result: {:?}", cpu_results);

        // Try different NTT orders
        for (order_name, order) in [
            ("NN", NTTInputOutputOrder::NN),
            ("NR", NTTInputOutputOrder::NR),
            ("RN", NTTInputOutputOrder::RN),
            ("RR", NTTInputOutputOrder::RR),
        ] {
            let mut gpu_buffer = v1.clone();
            let mut cfg = NTTConfig::default();
            cfg.order = order;
            ntt_batch(gpu_id, gpu_buffer.as_mut_ptr(), lg_n, cfg);
            let matches = gpu_buffer == cpu_results;
            println!("GPU NTT ({}) result: {:?} - Match: {}", order_name, gpu_buffer, matches);

            // Also try bit-reversing the output
            let mut gpu_reversed = gpu_buffer.clone();
            reverse_index_bits_in_place(&mut gpu_reversed);
            let matches_rev = gpu_reversed == cpu_results;
            println!("GPU NTT ({}) + bit-rev: {:?} - Match: {}", order_name, gpu_reversed, matches_rev);
        }

        // Now test with bit-reversed input + NN order (how plonky2 does it internally)
        let mut input_reversed = v1.clone();
        reverse_index_bits_in_place(&mut input_reversed);
        let mut gpu_buffer = input_reversed;
        let cfg = NTTConfig::default(); // NN order
        ntt_batch(gpu_id, gpu_buffer.as_mut_ptr(), lg_n, cfg);
        let matches = gpu_buffer == cpu_results;
        println!("GPU NTT with bit-reversed input (simulating plonky2): {:?} - Match: {}", gpu_buffer, matches);

        assert!(false, "Test complete - check outputs above");
    }

    #[test]
    fn test_gpu_coset_ntt_matches_cpu() {
        // Test just the coset NTT (without LDE) to isolate the issue
        let n = 1 << 14; // 16K elements
        let coeffs = GoldilocksField::rand_vec(n);

        // CPU: apply coset shift then FFT
        let modified_poly: PolynomialCoeffs<GoldilocksField> = GoldilocksField::coset_shift()
            .powers()
            .zip(&coeffs)
            .map(|(r, &c)| r * c)
            .collect::<Vec<_>>()
            .into();
        let cpu_result = crate::fft::fft(modified_poly).values;

        // GPU: use ntt_batch with with_coset=true
        ensure_gpu_initialized();
        let gpu_id = get_gpu_id() as usize;
        let lg_n = log2_strict(n);

        let mut gpu_buffer = coeffs.clone();
        let mut cfg = NTTConfig::default();
        cfg.batches = 1;
        cfg.with_coset = true;
        ntt_batch(gpu_id, gpu_buffer.as_mut_ptr(), lg_n, cfg);

        assert_eq!(cpu_result, gpu_buffer, "Coset NTT mismatch");
    }

    #[test]
    fn test_gpu_lde_coset_matches_cpu() {
        let n = 1 << 12; // 4K elements
        let rate_bits = 2;
        let coeffs = GoldilocksField::rand_vec(n);
        let poly = PolynomialCoeffs::new(coeffs);

        // CPU version
        let cpu_result = poly.lde(rate_bits)
            .coset_fft_with_options(GoldilocksField::coset_shift(), Some(rate_bits), None)
            .values;

        // GPU version
        let gpu_results = lde_batch_coset_gpu(&[poly.clone()], rate_bits);

        assert_eq!(cpu_result, gpu_results[0]);
    }

    #[test]
    fn test_gpu_lde_coset_batch_matches_cpu() {
        // Use larger size and multiple polynomials to actually exercise GPU code path
        let n = 1 << 14; // 16K elements
        let rate_bits = 2;
        let num_polys = 4;

        let polys: Vec<PolynomialCoeffs<GoldilocksField>> = (0..num_polys)
            .map(|_| PolynomialCoeffs::new(GoldilocksField::rand_vec(n)))
            .collect();

        // CPU version
        let cpu_results: Vec<Vec<GoldilocksField>> = polys
            .iter()
            .map(|p| {
                p.lde(rate_bits)
                    .coset_fft_with_options(GoldilocksField::coset_shift(), Some(rate_bits), None)
                    .values
            })
            .collect();

        // GPU version
        let gpu_results = lde_batch_coset_gpu(&polys, rate_bits);

        // Compare all results
        for i in 0..num_polys {
            assert_eq!(cpu_results[i], gpu_results[i], "Mismatch at polynomial {}", i);
        }
    }
}
