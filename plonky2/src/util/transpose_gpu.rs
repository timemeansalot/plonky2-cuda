//! GPU-accelerated transpose with bit-reversal using zeknox CUDA library.
//!
//! This module provides a GPU-accelerated combined transpose and bit-reversal
//! operation that replaces the separate `transpose()` and `reverse_index_bits_in_place()`
//! calls in the proving pipeline.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::hash::hash_types::RichField;
use crate::util::log2_strict;

#[cfg(feature = "cuda")]
use zeknox::types::TransposeConfig;
#[cfg(feature = "cuda")]
use zeknox::transpose_rev_batch;

/// Minimum size for GPU transpose to be beneficial
#[cfg(feature = "cuda")]
const MIN_GPU_TRANSPOSE_SIZE: usize = 1 << 12; // 4096

/// Maximum size for GPU transpose to be beneficial (beyond this, CPU is faster)
#[cfg(feature = "cuda")]
const MAX_GPU_TRANSPOSE_SIZE: usize = 1 << 13; // 8192

#[cfg(feature = "cuda")]
fn get_gpu_id() -> i32 {
    std::env::var("ZEKNOX_GPU_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Check if GPU transpose should be used
#[cfg(feature = "cuda")]
pub fn should_use_gpu_transpose(n: usize, batch_size: usize) -> bool {
    n >= MIN_GPU_TRANSPOSE_SIZE && n <= MAX_GPU_TRANSPOSE_SIZE && batch_size >= 2
}

/// GPU-accelerated transpose with bit-reversal.
///
/// Takes a matrix of shape [batch_size x n] and produces [n x batch_size]
/// with row indices bit-reversed.
///
/// This combines two operations:
/// 1. Transpose: [batch_size x n] -> [n x batch_size]
/// 2. Bit-reverse row indices
///
/// Input layout (row-major): matrix[i][j] = data[i * n + j]
/// Output layout (row-major): result[bit_rev(j)][i] = matrix[i][j]
#[cfg(feature = "cuda")]
pub fn transpose_rev_gpu<F: RichField>(matrix: &[Vec<F>]) -> Vec<Vec<F>> {
    let batch_size = matrix.len();
    if batch_size == 0 {
        return Vec::new();
    }

    let n = matrix[0].len();
    let lg_n = log2_strict(n);
    let gpu_id = get_gpu_id();

    // Flatten input matrix (row-major: batch_size rows of n elements each)
    let mut input_flat: Vec<u64> = Vec::with_capacity(batch_size * n);
    for row in matrix {
        for val in row {
            input_flat.push(val.to_canonical_u64());
        }
    }

    // Allocate output buffer
    let mut output_flat: Vec<u64> = vec![0u64; n * batch_size];

    let cfg = TransposeConfig {
        batches: batch_size as u32,
        are_inputs_on_device: false,
        are_outputs_on_device: false,
    };

    // Call zeknox GPU transpose
    transpose_rev_batch(
        gpu_id,
        output_flat.as_mut_ptr(),
        input_flat.as_ptr(),
        lg_n,
        cfg,
    );

    // Convert output back to Vec<Vec<F>>
    // Output shape: [n x batch_size] with bit-reversed row indices
    let mut result: Vec<Vec<F>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<F> = Vec::with_capacity(batch_size);
        for j in 0..batch_size {
            let val = output_flat[i * batch_size + j];
            row.push(F::from_canonical_u64(val));
        }
        result.push(row);
    }

    result
}

/// Combined transpose and bit-reversal for the proving pipeline.
///
/// When CUDA is enabled and conditions are met, uses GPU acceleration.
/// Otherwise falls back to CPU implementation.
pub fn transpose_and_reverse<F: RichField>(matrix: &[Vec<F>]) -> Vec<Vec<F>> {
    #[cfg(feature = "cuda")]
    {
        use core::any::TypeId;
        use crate::field::goldilocks_field::GoldilocksField;

        if !matrix.is_empty() {
            let n = matrix[0].len();
            let batch_size = matrix.len();

            // Only use GPU for GoldilocksField and sufficient size
            if TypeId::of::<F>() == TypeId::of::<GoldilocksField>()
                && should_use_gpu_transpose(n, batch_size)
            {
                log::info!(
                    "GPU Transpose: n={}, batch_size={}, lg_n={}",
                    n, batch_size, log2_strict(n)
                );
                return transpose_rev_gpu(matrix);
            }
        }
    }

    // CPU fallback: separate transpose and bit-reverse
    transpose_and_reverse_cpu(matrix)
}

/// CPU implementation of transpose + bit-reversal
fn transpose_and_reverse_cpu<F: Field + Copy + Sync + Send>(matrix: &[Vec<F>]) -> Vec<Vec<F>> {
    use crate::util::reverse_index_bits_in_place;

    if matrix.is_empty() {
        return Vec::new();
    }

    let n = matrix[0].len();

    // Transpose (parallel)
    let mut result: Vec<Vec<F>> = (0..n)
        .into_par_iter()
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect();

    // Bit-reverse
    reverse_index_bits_in_place(&mut result);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::Sample;

    type F = GoldilocksField;

    #[test]
    fn test_transpose_and_reverse_cpu() {
        // Test basic functionality with small matrix
        let n = 8;
        let batch_size = 4;

        let matrix: Vec<Vec<F>> = (0..batch_size)
            .map(|_| F::rand_vec(n))
            .collect();

        let result = transpose_and_reverse_cpu(&matrix);

        // Check dimensions
        assert_eq!(result.len(), n);
        assert_eq!(result[0].len(), batch_size);
    }

    #[test]
    fn test_transpose_and_reverse_consistency() {
        use crate::util::{transpose, reverse_index_bits_in_place};

        // Compare our combined function with separate operations
        let n = 16;
        let batch_size = 8;

        let matrix: Vec<Vec<F>> = (0..batch_size)
            .map(|_| F::rand_vec(n))
            .collect();

        // Method 1: Combined
        let result1 = transpose_and_reverse_cpu(&matrix);

        // Method 2: Separate operations
        let mut result2 = transpose(&matrix);
        reverse_index_bits_in_place(&mut result2);

        // Should be identical
        assert_eq!(result1.len(), result2.len());
        for i in 0..result1.len() {
            assert_eq!(result1[i], result2[i], "Row {} mismatch", i);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_transpose_rev_gpu_correctness() {
        use crate::util::{transpose, reverse_index_bits_in_place};

        // Test GPU vs CPU consistency
        for lg_n in [12, 14, 16] {
            let n = 1 << lg_n;
            let batch_size = 64;

            let matrix: Vec<Vec<F>> = (0..batch_size)
                .map(|_| F::rand_vec(n))
                .collect();

            // GPU result
            let result_gpu = transpose_rev_gpu(&matrix);

            // CPU result (separate operations)
            let mut result_cpu = transpose(&matrix);
            reverse_index_bits_in_place(&mut result_cpu);

            // Compare
            assert_eq!(result_gpu.len(), result_cpu.len(),
                "Length mismatch for n={}", n);

            for i in 0..result_gpu.len() {
                assert_eq!(result_gpu[i], result_cpu[i],
                    "Row {} mismatch for n={}", i, n);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore] // Run with --ignored for benchmark
    fn bench_transpose_gpu_vs_cpu() {
        use crate::util::{transpose, reverse_index_bits_in_place};
        use std::time::Instant;

        println!("\n=== Transpose GPU vs CPU Benchmark ===\n");

        for lg_n in [12, 14, 16, 18] {
            let n = 1 << lg_n;
            let batch_size = 128;

            let matrix: Vec<Vec<F>> = (0..batch_size)
                .map(|_| F::rand_vec(n))
                .collect();

            // Warmup GPU
            let _ = transpose_rev_gpu(&matrix);

            // GPU timing
            let start = Instant::now();
            let _result_gpu = transpose_rev_gpu(&matrix);
            let gpu_time = start.elapsed();

            // CPU timing
            let start = Instant::now();
            let mut result_cpu = transpose(&matrix);
            reverse_index_bits_in_place(&mut result_cpu);
            let cpu_time = start.elapsed();

            let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

            println!("n=2^{} ({} x {}):", lg_n, batch_size, n);
            println!("  GPU: {:?}", gpu_time);
            println!("  CPU: {:?}", cpu_time);
            println!("  Speedup: {:.2}x\n", speedup);
        }
    }
}
