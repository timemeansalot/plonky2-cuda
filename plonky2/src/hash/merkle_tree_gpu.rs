//! GPU-accelerated Merkle tree using zeknox CUDA library.
//!
//! This module provides GPU-accelerated Merkle tree construction for Goldilocks field
//! using the zeknox CUDA library. It uses a "linear" digest layout compatible with zeknox.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use plonky2_maybe_rayon::*;

use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::merkle_proofs::MerkleProof;
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::config::{GenericHashOut, Hasher, HasherType};
use crate::util::log2_strict;

#[cfg(feature = "cuda")]
use zeknox::device::memory::HostOrDeviceSlice;
#[cfg(feature = "cuda")]
use zeknox::device::stream::CudaStream;
#[cfg(feature = "cuda")]
use zeknox::fill_digests_buf_linear_gpu_with_gpu_ptr;

/// Minimum leaf size for GPU acceleration (must be > NUM_HASH_OUT_ELTS = 4)
const MIN_GPU_LEAF_SIZE: usize = 5;

/// Minimum number of leaves for GPU to be beneficial
const MIN_GPU_LEAVES: usize = 1 << 10; // 1024

/// Maximum number of leaves for GPU to be beneficial (beyond this, CPU is faster due to memory transfer overhead)
const MAX_GPU_LEAVES: usize = 1 << 17; // 131072

fn get_gpu_id() -> u64 {
    std::env::var("ZEKNOX_GPU_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Check if GPU Merkle tree should be used
pub fn should_use_gpu_merkle(leaves_count: usize, leaf_size: usize, hasher_type: HasherType) -> bool {
    // GPU Merkle requires leaf_size > 4 (NUM_HASH_OUT_ELTS) due to zeknox constraint
    // Also skip Keccak as zeknox's Keccak implementation may differ
    // Beyond MAX_GPU_LEAVES, CPU is faster due to memory transfer overhead
    leaf_size > MIN_GPU_LEAF_SIZE
        && leaves_count >= MIN_GPU_LEAVES
        && leaves_count <= MAX_GPU_LEAVES
        && hasher_type != HasherType::Keccak
}

/// GPU-accelerated Merkle tree with linear digest layout.
///
/// This tree uses a different digest layout than the standard MerkleTree:
/// - Digests are stored in "linear" order per subtree
/// - Each subtree's digests are contiguous
/// - This layout is compatible with zeknox GPU acceleration
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleTreeGpu<F: RichField, H: Hasher<F>> {
    /// Flattened leaf data (all leaves concatenated)
    pub leaves: Vec<F>,
    /// Size of each leaf
    pub leaf_size: usize,
    /// Digests in linear layout
    pub digests: Vec<H::Hash>,
    /// Merkle cap
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTreeGpu<F, H> {
    fn default() -> Self {
        Self {
            leaves: Vec::new(),
            leaf_size: 0,
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe { slice::from_raw_parts_mut(v_ptr, len) }
}

/// Fill subtree using CPU (linear layout)
///
/// Linear layout stores:
/// - Leaf hashes at indices [digests_buf.len() - leaves_count, digests_buf.len() - 1]
/// - Internal nodes from index 0, level by level bottom-up
fn fill_subtree_linear<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_size: usize,
) -> H::Hash {
    let leaves_count = leaves.len() / leaf_size;

    // Single leaf: return its hash
    if leaves_count == 1 {
        let hash = H::hash_or_noop(leaves);
        if !digests_buf.is_empty() {
            digests_buf[0].write(hash);
        }
        return hash;
    }

    // Two leaves: hash each and combine
    if leaves_count == 2 {
        let (leaf1, leaf2) = leaves.split_at(leaf_size);
        let hash_left = H::hash_or_noop(leaf1);
        let hash_right = H::hash_or_noop(leaf2);
        if digests_buf.len() >= 2 {
            digests_buf[0].write(hash_left);
            digests_buf[1].write(hash_right);
        }
        return H::two_to_one(hash_left, hash_right);
    }

    assert_eq!(leaves_count, digests_buf.len() / 2 + 1);

    // Hash leaves first (stored at end of buffer in linear layout)
    let (_, digests_leaves) = digests_buf.split_at_mut(digests_buf.len() - leaves_count);
    digests_leaves
        .par_iter_mut()
        .enumerate()
        .for_each(|(leaf_idx, digest)| {
            let start = leaf_idx * leaf_size;
            let leaf = &leaves[start..start + leaf_size];
            digest.write(H::hash_or_noop(leaf));
        });

    // Build internal nodes bottom-up (linear layout)
    // Following okx/plonky2's algorithm
    let mut last_index = digests_buf.len() - leaves_count;

    for level_log in (1..log2_strict(leaves_count)).rev() {
        let level_size = 1 << level_log;
        let (_, digests_slice) = digests_buf.split_at_mut(last_index - level_size);
        let (digests_slice, next_digests) = digests_slice.split_at_mut(level_size);

        digests_slice
            .par_iter_mut()
            .zip(last_index - level_size..last_index)
            .for_each(|(digest, idx)| {
                let left_idx = 2 * (idx + 1) - last_index;
                let right_idx = left_idx + 1;

                unsafe {
                    let left_digest = next_digests[left_idx].assume_init();
                    let right_digest = next_digests[right_idx].assume_init();
                    digest.write(H::two_to_one(left_digest, right_digest));
                }
            });
        last_index -= level_size;
    }

    // Return root hash (combine first two digests)
    unsafe {
        let left = digests_buf[0].assume_init();
        let right = digests_buf[1].assume_init();
        H::two_to_one(left, right)
    }
}

/// Fill digests buffer using CPU with linear layout (pure Rust, no zeknox)
fn fill_digests_buf_cpu_rust<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_size: usize,
    cap_height: usize,
) {
    let leaves_count = leaves.len() / leaf_size;

    // Special case: tree is all cap
    if digests_buf.is_empty() {
        cap_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(leaf_idx, cap)| {
                let start = leaf_idx * leaf_size;
                let leaf = &leaves[start..start + leaf_size];
                cap.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves_count >> cap_height;

    digests_buf
        .par_chunks_exact_mut(subtree_digests_len)
        .zip(cap_buf.par_iter_mut())
        .zip(leaves.par_chunks_exact(subtree_leaves_len * leaf_size))
        .for_each(|((subtree_digests, subtree_cap), subtree_leaves)| {
            subtree_cap.write(fill_subtree_linear::<F, H>(
                subtree_digests,
                subtree_leaves,
                leaf_size,
            ));
        });
}

/// Union for converting between u64 array and bytes
#[repr(C)]
union U8U64 {
    f1: [u8; 32],
    f2: [u64; 4],
}

/// Fill digests buffer using GPU via zeknox
#[cfg(feature = "cuda")]
fn fill_digests_buf_gpu<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_size: usize,
    cap_height: usize,
) {
    let leaves_count = leaves.len() / leaf_size;
    let gpu_id = get_gpu_id() as i32;

    // Allocate GPU buffers
    let digests_size = if digests_buf.is_empty() { NUM_HASH_OUT_ELTS } else { digests_buf.len() * NUM_HASH_OUT_ELTS };
    let caps_size = if cap_buf.is_empty() { NUM_HASH_OUT_ELTS } else { cap_buf.len() * NUM_HASH_OUT_ELTS };

    // Copy leaves to GPU as u64
    let mut gpu_leaves: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(gpu_id, leaves.len()).unwrap();
    let leaves_u64: Vec<u64> = leaves.iter().map(|f| f.to_canonical_u64()).collect();
    gpu_leaves.copy_from_host(&leaves_u64).expect("copy leaves to GPU");

    // Allocate output buffers on GPU
    let mut gpu_digests: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(gpu_id, digests_size).unwrap();
    let mut gpu_caps: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(gpu_id, caps_size).unwrap();

    // Call zeknox GPU kernel
    unsafe {
        fill_digests_buf_linear_gpu_with_gpu_ptr(
            gpu_digests.as_mut_ptr() as *mut core::ffi::c_void,
            gpu_caps.as_mut_ptr() as *mut core::ffi::c_void,
            gpu_leaves.as_mut_ptr() as *mut core::ffi::c_void,
            digests_buf.len() as u64,
            cap_buf.len() as u64,
            leaves_count as u64,
            leaf_size as u64,
            cap_height as u64,
            H::HASHER_TYPE as u64,
            gpu_id as u64,
        );
    }

    // Copy results back to host
    let mut host_digests: Vec<u64> = vec![0u64; digests_size];
    let mut host_caps: Vec<u64> = vec![0u64; caps_size];

    let stream1 = CudaStream::create().unwrap();
    let stream2 = CudaStream::create().unwrap();

    gpu_digests.copy_to_host_async(&mut host_digests, &stream1).expect("copy digests");
    gpu_caps.copy_to_host_async(&mut host_caps, &stream2).expect("copy caps");

    stream1.synchronize().expect("sync stream1");
    stream2.synchronize().expect("sync stream2");
    stream1.destroy().expect("destroy stream1");
    stream2.destroy().expect("destroy stream2");

    // Convert to Hash format
    if !digests_buf.is_empty() {
        host_digests
            .chunks_exact(NUM_HASH_OUT_ELTS)
            .zip(digests_buf.iter_mut())
            .for_each(|(chunk, digest)| {
                unsafe {
                    let mut parts = U8U64 { f1: [0; 32] };
                    parts.f2[0] = chunk[0];
                    parts.f2[1] = chunk[1];
                    parts.f2[2] = chunk[2];
                    parts.f2[3] = chunk[3];
                    let (slice, _) = parts.f1.split_at(H::HASH_SIZE);
                    digest.write(H::Hash::from_bytes(slice));
                }
            });
    }

    if !cap_buf.is_empty() {
        host_caps
            .chunks_exact(NUM_HASH_OUT_ELTS)
            .zip(cap_buf.iter_mut())
            .for_each(|(chunk, cap)| {
                unsafe {
                    let mut parts = U8U64 { f1: [0; 32] };
                    parts.f2[0] = chunk[0];
                    parts.f2[1] = chunk[1];
                    parts.f2[2] = chunk[2];
                    parts.f2[3] = chunk[3];
                    let (slice, _) = parts.f1.split_at(H::HASH_SIZE);
                    cap.write(H::Hash::from_bytes(slice));
                }
            });
    }
}

impl<F: RichField, H: Hasher<F>> MerkleTreeGpu<F, H> {
    /// Create a new Merkle tree from flattened leaf data
    pub fn new(leaves: Vec<F>, leaf_size: usize, cap_height: usize) -> Self {
        let leaves_count = leaves.len() / leaf_size;
        let log2_leaves_len = log2_strict(leaves_count);
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves_count)={}",
            cap_height,
            log2_leaves_len
        );

        let num_digests = 2 * (leaves_count - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);
        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);

        // Choose CPU or GPU based on size and hasher type
        #[cfg(feature = "cuda")]
        {
            if should_use_gpu_merkle(leaves_count, leaf_size, H::HASHER_TYPE) {
                log::info!("GPU Merkle: leaves_count={}, leaf_size={}, cap_height={}",
                    leaves_count, leaf_size, cap_height);
                fill_digests_buf_gpu::<F, H>(digests_buf, cap_buf, &leaves, leaf_size, cap_height);
            } else {
                fill_digests_buf_cpu_rust::<F, H>(digests_buf, cap_buf, &leaves, leaf_size, cap_height);
            }
        }

        #[cfg(not(feature = "cuda"))]
        fill_digests_buf_cpu_rust::<F, H>(digests_buf, cap_buf, &leaves, leaf_size, cap_height);

        unsafe {
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        Self {
            leaves,
            leaf_size,
            digests,
            cap: MerkleCap(cap),
        }
    }

    /// Create from 2D leaf data (Vec<Vec<F>>)
    pub fn from_2d(leaves_2d: Vec<Vec<F>>, cap_height: usize) -> Self {
        let leaf_size = leaves_2d[0].len();
        let leaves: Vec<F> = leaves_2d.into_iter().flatten().collect();
        Self::new(leaves, leaf_size, cap_height)
    }

    /// Get a leaf by index
    pub fn get(&self, i: usize) -> &[F] {
        &self.leaves[i * self.leaf_size..(i + 1) * self.leaf_size]
    }

    /// Get leaves as 2D vector
    pub fn get_leaves_2d(&self) -> Vec<Vec<F>> {
        self.leaves
            .chunks_exact(self.leaf_size)
            .map(|c| c.to_vec())
            .collect()
    }

    /// Get number of leaves
    pub fn leaves_count(&self) -> usize {
        self.leaves.len() / self.leaf_size
    }

    /// Create a Merkle proof for a leaf (linear layout)
    ///
    /// The linear layout stores digests as:
    /// - Level (num_layers-1) at indices 0..2 (closest to root)
    /// - Level (num_layers-2) at indices 2..6
    /// - ...
    /// - Level 0 (leaf hashes) at indices (leaves_per_subtree - 2)..(subtree_digest_size)
    ///
    /// Level i starts at index: (1 << (num_layers - i)) - 2
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let leaves_count = self.leaves_count();
        let num_layers = log2_strict(leaves_count) - cap_height;

        if num_layers == 0 {
            return MerkleProof { siblings: Vec::new() };
        }

        let leaves_per_subtree = 1 << num_layers;
        let subtree_digest_size = 2 * leaves_per_subtree - 2;
        let subtree_idx = leaf_index >> num_layers;
        let idx_in_subtree = leaf_index & (leaves_per_subtree - 1);

        // Leaf level (i=0) starts at: (1 << num_layers) - 2 = leaves_per_subtree - 2
        let leaf_level_start = leaves_per_subtree - 2;
        let mut idx = leaf_level_start + idx_in_subtree;

        let siblings = (0..num_layers)
            .map(|i| {
                // Level i starts at: (1 << (num_layers - i)) - 2
                let level_size = 1 << (num_layers - i);
                let level_start = level_size - 2;
                let pos_in_level = idx - level_start;

                // Get sibling (toggle last bit of position)
                let sibling_pos = pos_in_level ^ 1;
                let sibling_idx_in_subtree = level_start + sibling_pos;
                let abs_idx = subtree_idx * subtree_digest_size + sibling_idx_in_subtree;
                let sibling = self.digests[abs_idx];

                // Move up to parent level (i+1)
                // Parent level starts at: (1 << (num_layers - i - 1)) - 2 = (level_size >> 1) - 2
                if i + 1 < num_layers {
                    let parent_level_start = (level_size >> 1) - 2;
                    idx = parent_level_start + pos_in_level / 2;
                }

                sibling
            })
            .collect();

        MerkleProof { siblings }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::hash::poseidon::PoseidonHash;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2_field::types::Sample;

    type F = GoldilocksField;
    type H = PoseidonHash;

    fn random_leaves(n: usize, leaf_size: usize) -> Vec<F> {
        F::rand_vec(n * leaf_size)
    }

    #[test]
    fn test_merkle_tree_gpu_basic() {
        let n = 16;
        let leaf_size = 8;
        let cap_height = 2;

        let leaves = random_leaves(n, leaf_size);
        let tree = MerkleTreeGpu::<F, H>::new(leaves.clone(), leaf_size, cap_height);

        assert_eq!(tree.leaves_count(), n);
        assert_eq!(tree.cap.len(), 1 << cap_height);
    }

    #[test]
    fn test_merkle_tree_gpu_proof() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let n = 64;
        let leaf_size = 8;
        let cap_height = 2;

        let leaves = random_leaves(n, leaf_size);
        let tree = MerkleTreeGpu::<F, <C as GenericConfig<D>>::Hasher>::new(
            leaves.clone(),
            leaf_size,
            cap_height
        );

        // Verify all leaves
        for i in 0..n {
            let leaf = tree.get(i).to_vec();
            let proof = tree.prove(i);
            let result = verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof);
            assert!(result.is_ok(), "Proof failed for leaf {}: {:?}", i, result);
        }
    }

    #[test]
    fn test_merkle_tree_gpu_various_sizes() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        for log_n in 4..10 {
            let n = 1 << log_n;
            for cap_height in 0..=log_n {
                let leaf_size = 8;
                let leaves = random_leaves(n, leaf_size);
                let tree = MerkleTreeGpu::<F, <C as GenericConfig<D>>::Hasher>::new(
                    leaves.clone(),
                    leaf_size,
                    cap_height
                );

                // Test a few random leaves
                for i in [0, n/4, n/2, n-1] {
                    let leaf = tree.get(i).to_vec();
                    let proof = tree.prove(i);
                    let result = verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof);
                    assert!(result.is_ok(),
                        "Proof failed for n={}, cap_height={}, leaf {}: {:?}",
                        n, cap_height, i, result);
                }
            }
        }
    }

    #[test]
    fn test_merkle_tree_gpu_vs_cpu_correctness() {
        use crate::hash::merkle_tree::MerkleTree;

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type H = <C as GenericConfig<D>>::Hasher;

        // Test sizes too small for GPU (will use CPU path)
        for log_n in 4..8 {
            let n = 1 << log_n;
            let leaf_size = 8;
            let cap_height = 2;

            let leaves_flat = random_leaves(n, leaf_size);
            let leaves_2d: Vec<Vec<F>> = leaves_flat.chunks_exact(leaf_size)
                .map(|c| c.to_vec())
                .collect();

            // Build both trees
            let tree_gpu = MerkleTreeGpu::<F, H>::new(leaves_flat.clone(), leaf_size, cap_height);
            let tree_cpu = MerkleTree::<F, H>::new(leaves_2d.clone(), cap_height);

            // Verify cap matches
            assert_eq!(tree_gpu.cap.0.len(), tree_cpu.cap.0.len(),
                "Cap length mismatch for n={}", n);

            // Both caps should produce the same root hashes
            for (i, (gpu_cap, cpu_cap)) in tree_gpu.cap.0.iter().zip(tree_cpu.cap.0.iter()).enumerate() {
                assert_eq!(gpu_cap, cpu_cap,
                    "Cap {} mismatch for n={}: GPU={:?}, CPU={:?}", i, n, gpu_cap, cpu_cap);
            }

            // Verify proofs are compatible
            for leaf_idx in [0, n/4, n/2, n-1] {
                let leaf = tree_gpu.get(leaf_idx).to_vec();
                let proof_gpu = tree_gpu.prove(leaf_idx);
                let proof_cpu = tree_cpu.prove(leaf_idx);

                // Both proofs should verify
                let result_gpu = verify_merkle_proof_to_cap(leaf.clone(), leaf_idx, &tree_gpu.cap, &proof_gpu);
                let result_cpu = verify_merkle_proof_to_cap(leaf.clone(), leaf_idx, &tree_cpu.cap, &proof_cpu);

                assert!(result_gpu.is_ok(),
                    "GPU proof failed for n={}, leaf {}: {:?}", n, leaf_idx, result_gpu);
                assert!(result_cpu.is_ok(),
                    "CPU proof failed for n={}, leaf {}: {:?}", n, leaf_idx, result_cpu);
            }
        }
    }

    #[test]
    #[ignore] // Run with --ignored for benchmark
    fn bench_merkle_tree_gpu_vs_cpu() {
        use crate::hash::merkle_tree::MerkleTree;
        use std::time::Instant;

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type H = <C as GenericConfig<D>>::Hasher;

        println!("\n=== Merkle Tree GPU vs CPU Benchmark ===\n");

        for log_n in [12, 14, 16, 18, 20] {
            let n = 1 << log_n;
            let leaf_size = 8;
            let cap_height = 4;

            let leaves_flat = random_leaves(n, leaf_size);
            let leaves_2d: Vec<Vec<F>> = leaves_flat.chunks_exact(leaf_size)
                .map(|c| c.to_vec())
                .collect();

            // Warmup
            let _ = MerkleTreeGpu::<F, H>::new(leaves_flat.clone(), leaf_size, cap_height);

            // GPU timing
            let start = Instant::now();
            let tree_gpu = MerkleTreeGpu::<F, H>::new(leaves_flat.clone(), leaf_size, cap_height);
            let gpu_time = start.elapsed();

            // CPU timing
            let start = Instant::now();
            let tree_cpu = MerkleTree::<F, H>::new(leaves_2d.clone(), cap_height);
            let cpu_time = start.elapsed();

            let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

            // Verify correctness
            for (gpu_cap, cpu_cap) in tree_gpu.cap.0.iter().zip(tree_cpu.cap.0.iter()) {
                assert_eq!(gpu_cap, cpu_cap, "Cap mismatch for n={}", n);
            }

            println!("n=2^{} ({} leaves):", log_n, n);
            println!("  GPU: {:?}", gpu_time);
            println!("  CPU: {:?}", cpu_time);
            println!("  Speedup: {:.2}x\n", speedup);
        }
    }
}
