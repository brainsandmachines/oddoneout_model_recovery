import torch
import math
import time
import numpy as np

def calculate_batch_size(n_triplets: int, feature_dim: int, max_memory_gb: float = 12.0) -> int:
    """
    Calculate the maximum batch size that fits within the specified GPU memory limit.
    
    Args:
        n_triplets: Total number of triplets
        feature_dim: Dimension of the feature vectors
        max_memory_gb: Maximum GPU memory to use in gigabytes
    
    Returns:
        Optimal batch size
    """
    # Memory calculations (in bytes):
    bytes_per_float = 4  # float32
    
    # For matmul method:
    # 1. Selected features: batch_size × 3 × feature_dim
    # 2. Similarity matrix: batch_size × 3 × 3
    # 3. Output: batch_size × 3
    # Plus some buffer for computation
    
    max_memory_bytes = max_memory_gb * 1024**3  # Convert GB to bytes
    memory_per_triplet = (3 * feature_dim + 9 + 3) * bytes_per_float
    
    # Add 20% buffer for computational overhead
    safe_memory = max_memory_bytes * 0.4  # More conservative
    
    batch_size = int(safe_memory / memory_per_triplet)
    
    # Round down to nearest 1000 for good measure
    batch_size = (batch_size // 1000) * 1000
    
    return max(1000, min(batch_size, n_triplets))
def S_pairs_matmul_batch(triplets: torch.Tensor, 
                        features: torch.Tensor, 
                        max_memory_gb: float = 12.0) -> torch.Tensor:
    """Matmul method with batching"""
    device = features.device
    n_triplets = triplets.shape[0]
    feature_dim = features.shape[1]
    
    batch_size = calculate_batch_size(n_triplets, feature_dim, max_memory_gb)
    n_batches = math.ceil(n_triplets / batch_size)  
    
    result = torch.empty(n_triplets, 3, device=device)
    
    for i in range(n_batches):
        torch.cuda.empty_cache()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_triplets)
        
        batch_triplets = triplets[start_idx:end_idx]
        selected_features = features[batch_triplets]
        #calculate the similarity scores
        S_pairs_full = torch.matmul(selected_features, selected_features.transpose(1,2))
        #store the similarity scores in the result tensor
        result[start_idx:end_idx] = torch.stack([
            S_pairs_full[:,1,2],  # x_j·x_k
            S_pairs_full[:,0,2],  # x_i·x_k
            S_pairs_full[:,0,1]   # x_i·x_j
        ], dim=1)
        #delete the batch images_features and similarity scores
        del selected_features, S_pairs_full
    return result


def S_pairs_einsum_batch(triplets: torch.Tensor, 
                        features: torch.Tensor, 
                        max_memory_gb: float = 12.0) -> torch.Tensor:
    """Einsum method with batching"""
    device = features.device
    n_triplets = triplets.shape[0]
    feature_dim = features.shape[1]
    
    batch_size = calculate_batch_size(n_triplets, feature_dim, max_memory_gb)
    n_batches = math.ceil(n_triplets / batch_size)
    
    result = torch.empty(n_triplets, 3, device=device)
    
    for i in range(n_batches):
        torch.cuda.empty_cache()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_triplets)
        
        batch_triplets = triplets[start_idx:end_idx]
        selected_features = features[batch_triplets]
        
        # For triplet [i,j,k], compute:
        result[start_idx:end_idx, 0] = torch.einsum('bd,bd->b', 
            selected_features[:, 1], selected_features[:, 2])  # x_j·x_k
        result[start_idx:end_idx, 1] = torch.einsum('bd,bd->b', 
            selected_features[:, 0], selected_features[:, 2])  # x_i·x_k
        result[start_idx:end_idx, 2] = torch.einsum('bd,bd->b', 
            selected_features[:, 0], selected_features[:, 1])  # x_i·x_j
        
        del selected_features
    
    return result

def S_pairs_vectorized_batch(triplets: torch.Tensor, 
                           features: torch.Tensor, 
                           max_memory_gb: float = 12.0) -> torch.Tensor:
    """Vectorized method with batching that avoids creating full similarity matrix"""
    device = features.device
    n_triplets = triplets.shape[0]
    feature_dim = features.shape[1]
    
    batch_size = calculate_batch_size(n_triplets, feature_dim, max_memory_gb)
    n_batches = math.ceil(n_triplets / batch_size)
    
    result = torch.empty(n_triplets, 3, device=device)
    
    for i in range(n_batches):
        torch.cuda.empty_cache()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_triplets)
        
        batch_triplets = triplets[start_idx:end_idx]
        
        # Get features for i, j, k positions
        features_i = features[batch_triplets[:, 0]]  # (batch, dim)
        features_j = features[batch_triplets[:, 1]]  # (batch, dim)
        features_k = features[batch_triplets[:, 2]]  # (batch, dim)
        
        # Compute dot products directly without creating similarity matrix
        result[start_idx:end_idx, 0] = torch.sum(features_j * features_k, dim=1)  # x_j·x_k
        result[start_idx:end_idx, 1] = torch.sum(features_i * features_k, dim=1)  # x_i·x_k
        result[start_idx:end_idx, 2] = torch.sum(features_i * features_j, dim=1)  # x_i·x_j
        
        del features_i, features_j, features_k
    
    return result

def S_pairs_precomputed_batch(
    triplets: torch.Tensor, 
    features: torch.Tensor, 
    max_memory_gb: float = 12.0
    ) -> torch.Tensor:
    """Method that precomputes similarity matrix in batches if needed"""
    device = features.device
    n_images = features.shape[0]
    
    # If similarity matrix would be too large, use other methods
    if n_images > 2000:  # Arbitrary threshold
        return S_pairs_vectorized_batch(triplets, features, max_memory_gb)
    
    # Compute full similarity matrix once
    S_matrix = torch.matmul(features, features.T)
    
    # Extract required similarities using indexing
    i, j, k = triplets.T
    S_pairs = S_matrix[torch.stack([j,i,i]), torch.stack([k,k,j])].T
    
    return S_pairs

def S_pairs_optimized_direct(triplets: torch.Tensor, 
                            features: torch.Tensor, 
                            max_memory_gb: float = 12.0) -> torch.Tensor:
    """
    Optimized method that directly computes required similarities using batch operations
    and avoiding redundant computations by intelligent feature selection and CUDA optimizations
    """
    device = features.device
    n_triplets = triplets.shape[0]
    
    # Get unique indices to avoid redundant feature fetching
    unique_indices = torch.unique(triplets)
    
    # Create a mapping from original indices to condensed indices
    index_map = torch.empty(features.shape[0], dtype=torch.long, device=device)
    index_map[unique_indices] = torch.arange(len(unique_indices), device=device)
    
    # Map triplets to condensed indices
    mapped_triplets = index_map[triplets]
    
    # Get only the needed features
    condensed_features = features[unique_indices]
    
    # Prepare indices for the three pairs we need
    i, j, k = mapped_triplets.T
    
    # Compute all required dot products in one go using optimized operations
    result = torch.empty(n_triplets, 3, device=device)
    
    # Compute j·k, i·k, i·j in parallel using batched operations
    result[:, 0] = torch.sum(condensed_features[j] * condensed_features[k], dim=1)  # j·k
    result[:, 1] = torch.sum(condensed_features[i] * condensed_features[k], dim=1)  # i·k
    result[:, 2] = torch.sum(condensed_features[i] * condensed_features[j], dim=1)  # i·j
    
    return result

def S_pairs_optimized_direct_v2(triplets: torch.Tensor, 
                               features: torch.Tensor, 
                               max_memory_gb: float = 12.0) -> torch.Tensor:
    """
    Improved version that leverages better batching and reduces overhead
    """
    device = features.device
    n_triplets = triplets.shape[0]
    
    # Extract all needed features at once without unique operation
    i, j, k = triplets.T
    
    # Prepare features for batch computation
    features_i = features[i]  # (n_triplets, dim)
    features_j = features[j]  # (n_triplets, dim)
    features_k = features[k]  # (n_triplets, dim)
    
    # Stack features for batch computation
    # Shape: (n_triplets, 3, dim)
    stacked_features = torch.stack([features_j, features_i, features_i], dim=1)
    # Shape: (n_triplets, 3, dim)
    stacked_features2 = torch.stack([features_k, features_k, features_j], dim=1)
    
    # Compute all similarities at once using batch matrix multiplication
    # (n_triplets, 3) = bmm((n_triplets, 3, dim), (n_triplets, dim, 1))
    result = torch.bmm(stacked_features, stacked_features2.transpose(1, 2))
    
    return result.diagonal(dim1=1, dim2=2)

def benchmark_method(method_func, triplets, features, max_memory_gb, n_runs=2):
    """Benchmark a method's performance with detailed metrics"""
    torch.cuda.empty_cache()
    
    times = []
    peak_memory = 0
    
    # First run to warm up
    _ = method_func(triplets, features, max_memory_gb)
    torch.cuda.empty_cache()
    
    for run in range(n_runs):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        
        result = method_func(triplets, features, max_memory_gb)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1024**3)
        
        # Clear memory after each run
        torch.cuda.empty_cache()
    
    return {
        'times': times,
        'peak_memory': peak_memory,
        'throughput': triplets.shape[0] / (sum(times) / len(times))
    }

def run_sanity_checks():
    print("Running sanity checks...")
    
    features = torch.tensor([
        [1.0, 0.0],  # vector 0
        [0.0, 1.0],  # vector 1
        [1.0, 1.0],  # vector 2
        [-1.0, 0.0]  # vector 3
    ], device='cuda')
    
    triplets = torch.tensor([
        [0, 1, 2],  # i=0, j=1, k=2: [dot(1,2)=1, dot(0,2)=1, dot(0,1)=0]
        [1, 2, 3],  # i=1, j=2, k=3: [dot(2,3)=-1, dot(1,3)=0, dot(1,2)=1]
        [0, 3, 2]   # i=0, j=3, k=2: [dot(3,2)=-1, dot(0,2)=1, dot(0,3)=-1]
    ], device='cuda')
    
    expected = torch.tensor([
        [1.0, 1.0, 0.0],    # [dot(1,2)=1, dot(0,2)=1, dot(0,1)=0]
        [-1.0, 0.0, 1.0],   # [dot(2,3)=-1, dot(1,3)=0, dot(1,2)=1]
        [-1.0, 1.0, -1.0]   # [dot(3,2)=-1, dot(0,2)=1, dot(0,3)=-1]
    ], device='cuda')
    
    methods = {
        "Matmul": S_pairs_matmul_batch,
        "Einsum": S_pairs_einsum_batch,
        "Vectorized": S_pairs_vectorized_batch,
        "Precomputed": S_pairs_precomputed_batch,
        "OptimizedDirect": S_pairs_optimized_direct,
        "OptimizedDirectV2": S_pairs_optimized_direct_v2
    }
    
    tolerance = 1e-5
    all_tests_passed = True
    
    for name, method in methods.items():
        print(f"\nTesting {name} method:")
        try:
            result = method(triplets, features, max_memory_gb=1.0)
            max_diff = torch.max(torch.abs(result - expected)).item()
            
            if max_diff > tolerance:
                print(f"❌ {name} test failed!")
                print("Expected:", expected)
                print("Got:", result)
                print(f"Max difference: {max_diff:.2e}")
                all_tests_passed = False
            else:
                print(f"✅ {name} test passed!")
                print(f"Max difference: {max_diff:.2e}")
        except Exception as e:
            print(f"❌ {name} test failed with error: {str(e)}")
            all_tests_passed = False
    
    return all_tests_passed

def run_multiple_simulations(n_sims=100):
    print(f"Running {n_sims} simulations...")
    
    # Reduced test parameters
    N_IMAGES = 1000        # Reduced from 1854
    EMBEDDING_SIZE = 512   # Reduced from 1024
    N_TRIPLETS = 100000   # Reduced from 4,700,000
    MAX_GPU_MEMORY = 20.0
    
    methods = {
        "Matmul": S_pairs_matmul_batch,
        "Einsum": S_pairs_einsum_batch,
        "Vectorized": S_pairs_vectorized_batch,
        "Precomputed": S_pairs_precomputed_batch,
        "OptimizedDirect": S_pairs_optimized_direct,
        "OptimizedDirectV2": S_pairs_optimized_direct_v2
    }
    
    # Initialize results dictionary
    all_results = {name: {"times": [], "peak_memory": [], "throughput": []} for name in methods}
    
    for sim in range(n_sims):
        print(f"\nSimulation {sim + 1}/{n_sims}")
        
        X = torch.randn(N_IMAGES, EMBEDDING_SIZE)
        Y = torch.randint(0, N_IMAGES, (N_TRIPLETS, 3))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        Y = Y.to(device)
        
        for name, func in methods.items():
            print(f"  Testing {name}...", end='', flush=True)
            try:
                stats = benchmark_method(func, Y, X, MAX_GPU_MEMORY)
                all_results[name]["times"].append(np.mean(stats["times"]))
                all_results[name]["peak_memory"].append(stats["peak_memory"])
                all_results[name]["throughput"].append(stats["throughput"])
                print(" Done")
            except RuntimeError as e:
                print(f" Error: {e}")
                
        # Clear GPU memory after each simulation
        torch.cuda.empty_cache()
    
    # Print detailed report
    print("\nDetailed Performance Report")
    print("=" * 80)
    
    # Calculate statistics
    stats = {}
    for name in methods:
        stats[name] = {
            "time_mean": np.mean(all_results[name]["times"]),
            "time_std": np.std(all_results[name]["times"]),
            "memory_mean": np.mean(all_results[name]["peak_memory"]),
            "memory_std": np.std(all_results[name]["peak_memory"]),
            "throughput_mean": np.mean(all_results[name]["throughput"]),
            "throughput_std": np.std(all_results[name]["throughput"])
        }
    
    # Find best method for each metric
    best_time = min(stats.items(), key=lambda x: x[1]["time_mean"])[0]
    best_memory = min(stats.items(), key=lambda x: x[1]["memory_mean"])[0]
    best_throughput = max(stats.items(), key=lambda x: x[1]["throughput_mean"])[0]
    
    # Print individual method statistics
    for name in methods:
        print(f"\n{name} Method:")
        print("-" * 40)
        print(f"Time:       {stats[name]['time_mean']:.3f} ± {stats[name]['time_std']:.3f} seconds")
        print(f"Memory:     {stats[name]['memory_mean']:.2f} ± {stats[name]['memory_std']:.2f} GB")
        print(f"Throughput: {stats[name]['throughput_mean']:,.0f} ± {stats[name]['throughput_std']:,.0f} triplets/second")
    
    # Print comparative analysis
    print("\nComparative Analysis")
    print("=" * 80)
    print(f"Fastest method: {best_time}")
    print(f"Most memory efficient: {best_memory}")
    print(f"Highest throughput: {best_throughput}")
    
    # Print relative performance
    baseline = "Matmul"
    print(f"\nRelative Performance (compared to {baseline}):")
    for name in [n for n in methods if n != baseline]:
        time_diff = ((stats[name]["time_mean"] - stats[baseline]["time_mean"]) / 
                    stats[baseline]["time_mean"]) * 100
        memory_diff = ((stats[name]["memory_mean"] - stats[baseline]["memory_mean"]) / 
                      stats[baseline]["memory_mean"]) * 100
        throughput_diff = ((stats[name]["throughput_mean"] - stats[baseline]["throughput_mean"]) / 
                          stats[baseline]["throughput_mean"]) * 100
        
        print(f"\n{name} vs {baseline}:")
        print(f"Time:       {abs(time_diff):.1f}% {'slower' if time_diff > 0 else 'faster'}")
        print(f"Memory:     {abs(memory_diff):.1f}% {'more' if memory_diff > 0 else 'less'}")
        print(f"Throughput: {abs(throughput_diff):.1f}% {'lower' if throughput_diff < 0 else 'higher'}")

# if __name__ == "__main__":
#     if not run_sanity_checks():
#         print("\nAborting further tests due to failed sanity checks!")
#         exit(1)
    
#     print("\n" + "="*50 + "\n")
#     run_multiple_simulations(n_sims=100)