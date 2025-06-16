import cupy as cp
import numpy as np
import allel
import tskit
from collections import Counter

class HaplotypeMatrix:
    """
    Represents a haplotype matrix, which is a matrix of haplotypes across multiple variants.
    This class provides methods for manipulating and analyzing the haplotype matrix, including
    extracting subsets based on positions, calculating allele frequency spectra, and computing
    nucleotide diversity (π).
    """
    """
    A class for representing a haplotype matrix.
    
    Attributes:
        haplotypes (cp.ndarray): The genotype/haplotype matrix.
        positions (cp.ndarray): The array of variant positions.
        chrom_start (int): Chromosome start position.
        chrom_end (int): Chromosome end position.
    """
    def __init__(self, 
                 genotypes, # either a numpy array or a cupy array
                 positions, # either a numpy array or a cupy array
                 chrom_start: int = None,
                 chrom_end: int = None,
                 sample_sets: dict = None  # new optional parameter for population sample sets
                ):
        # test for empty genotypes
        if genotypes.size == 0:
            raise ValueError("genotypes cannot be empty")
        # test for empty positions
        if positions.size == 0:
            raise ValueError("positions cannot be empty")
        # make sure genotypes and positions are either numpy or cupy arrays
        if not isinstance(genotypes, np.ndarray) and not isinstance(genotypes, cp.ndarray):
            raise ValueError("genotypes must be a numpy or cupy array")
        if not isinstance(positions, np.ndarray) and not isinstance(positions, cp.ndarray):
            raise ValueError("positions must be a numpy or cupy array")
        
        # Determine device based on genotypes.
        # transfer positions if necessary
        if isinstance(genotypes, cp.ndarray):
            self._device = 'GPU'
            if isinstance(positions, np.ndarray):
                positions = cp.array(positions)
        else:
            self._device = 'CPU'
            if isinstance(positions, cp.ndarray):
                positions = positions.get()
       
        # set attributes
        self.haplotypes = genotypes
        self.positions = positions
        self.chrom_start = chrom_start
        self.chrom_end = chrom_end
        self._sample_sets = sample_sets  # store the sample set info (optional)

    @property
    def device(self):
        """Returns the current device (CPU or GPU)."""
        return self._device

    @property
    def sample_sets(self):
        """
        Defines groups of haplotypes that belong to populations.

        Returns:
            dict: A dictionary mapping population names to lists of haplotype indices.
                  If _sample_sets was not specified at construction, returns a default 
                  dictionary with a single key 'all' containing all haplotype indices.
        """
        if self._sample_sets is None:
            # All haplotypes belong to a single population labeled "all"
            return {"all": list(range(self.haplotypes.shape[0]))}
        return self._sample_sets
    
    @sample_sets.setter
    def sample_sets(self, sample_sets: dict):
        """
        Set the sample sets.
        """
        if not isinstance(sample_sets, dict):
            raise ValueError("sample_sets must be a dictionary")
        # check that the values are lists
        for key, value in sample_sets.items():
            if not isinstance(value, list):
                raise ValueError("values in sample_sets must be lists")
        self._sample_sets = sample_sets

    def transfer_to_gpu(self):
        """Transfer data from CPU to GPU."""
        if self.device == 'CPU':
            self.haplotypes = cp.asarray(self.haplotypes)
            self.positions = cp.asarray(self.positions)
            self._device = 'GPU'

    def transfer_to_cpu(self):
        """Transfer data from GPU to CPU."""
        if self.device == 'GPU':
            self.haplotypes = np.asarray(self.haplotypes.get())
            self.positions = np.asarray(self.positions.get())
            self._device = 'CPU'

    @classmethod
    def from_vcf(cls, path: str):
        """
        Construct a HaplotypeMatrix from a VCF file.
        
        Parameters:
            path (str): The file path to the VCF file.
            
        Returns:
            HaplotypeMatrix: An instance created from the VCF data.
            Assumes that the VCF is phased.
            Sets the chromosome start and end to the first and last variant positions.
        """
        vcf = allel.read_vcf(path)
        genotypes = allel.GenotypeArray(vcf['calldata/GT'])
        num_variants, num_samples, ploidy = genotypes.shape
        
        # assert that the ploidy is 2
        assert ploidy == 2
       
        # convert to haplotype matrix
        haplotypes = np.empty((num_variants, 2*num_samples), dtype=genotypes.dtype)
        # fill the haplotypes array
        haplotypes[:, 0:num_samples] = genotypes[:, :, 0]  # First allele for all variants
        haplotypes[:, num_samples:2*num_samples] = genotypes[:, :, 1]  # Second allele for all variants
       
        # transpose the haplotypes array
        haplotypes = haplotypes.T
        positions = np.array(vcf['variants/POS'])   
        
        # get the chromosome start and end
        chrom_start = positions[0]
        chrom_end = positions[-1]
        return cls(haplotypes, positions, chrom_start, chrom_end)

    @classmethod
    def from_ts(cls, ts: tskit.TreeSequence, device: str = 'CPU') -> 'HaplotypeMatrix':
        """
        Create a HaplotypeMatrix from a tskit.TreeSequence.
        
        Args:
            ts: A tskit.TreeSequence object
            
        Returns:
            HaplotypeMatrix: A new HaplotypeMatrix instance
        """
        # Convert ts to haplotype matrix
        haplotypes = ts.genotype_matrix().T
        positions = ts.tables.sites.position
        # get the chromosome start and end
        chrom_start = 0
        chrom_end = ts.sequence_length
        if device == 'GPU':
            # Convert to CuPy arrays
            haplotypes = cp.array(haplotypes)
            positions = cp.array(positions)
        
        return cls(haplotypes, positions, chrom_start, chrom_end)

    def get_matrix(self) -> cp.ndarray:
        """
        Returns the haplotype matrix.
        
        Returns:
            cp.ndarray: The array representing the haplotype/genotype matrix.
        """
        return self.haplotypes

    def get_positions(self) -> cp.ndarray:
        """
        Returns the variant positions.
        
        Returns:
            cp.ndarray: The array of positions.
        """
        return self.positions

    @property
    def shape(self):
        """
        Returns the shape of the haplotype matrix.
        
        Returns:
            tuple: A tuple representing the dimensions (variants, samples)
                   of the haplotype matrix.
        """
        return self.haplotypes.shape
    
    @property
    def num_variants(self):
        """
        Returns the number of variants in the haplotype matrix.
        """
        return self.haplotypes.shape[1]
    
    @property
    def num_haplotypes(self):
        """
        Returns the number of haplotypes in the haplotype matrix.
        """
        return self.haplotypes.shape[0]
    
    def __repr__(self):
        first_pos = self.positions[0] if self.positions.size > 0 else None
        last_pos = self.positions[-1] if self.positions.size > 0 else None
        return (f"HaplotypeMatrix(shape={self.shape}, "
                f"first_position={first_pos}, last_position={last_pos})")
    
    def get_subset(self, positions) -> "HaplotypeMatrix":
        """
        Get a subset of the haplotype matrix based on the provided positions.
        
        Parameters:
            positions: A one-dimensional array of indices to select from the haplotype matrix.
                       This can be either a NumPy array or a CuPy array.
            
        Returns:
            HaplotypeMatrix: A new instance containing the subset of the haplotype matrix.
        """
        # Ensure positions is one-dimensional
        if positions.ndim != 1:
            raise ValueError("Positions must be a one-dimensional array.")
        
        # Convert positions to match the device of the haplotype matrix.
        if self.device == 'CPU' and isinstance(positions, cp.ndarray):
            positions = cp.asnumpy(positions)
        elif self.device == 'GPU' and isinstance(positions, np.ndarray):
            positions = cp.array(positions)
       
        # Validate that positions are valid indices.
        # Ensure positions are valid indices and convert to integer type
        positions = cp.asarray(positions, dtype=np.int64) if self.device == 'GPU' else np.asarray(positions, dtype=np.int64)
        
        # Handle empty positions array
        if len(positions) == 0:
            # Create empty subset maintaining the same structure
            # Need to create arrays that have non-zero size to satisfy constructor
            if self.device == 'GPU':
                empty_haplotypes = cp.empty((self.haplotypes.shape[0], 0), dtype=self.haplotypes.dtype)
                empty_positions = cp.array([], dtype=self.positions.dtype)
            else:
                empty_haplotypes = np.empty((self.haplotypes.shape[0], 0), dtype=self.haplotypes.dtype)
                empty_positions = np.array([], dtype=self.positions.dtype)
            
            # For empty subsets, bypass constructor validation by setting size to non-zero temporarily
            result = object.__new__(HaplotypeMatrix)
            result.haplotypes = empty_haplotypes
            result.positions = empty_positions
            result.chrom_start = self.chrom_start
            result.chrom_end = self.chrom_end
            result._sample_sets = self._sample_sets
            result._device = self._device
            return result
        
        if not (positions >= 0).all() or not (positions < self.haplotypes.shape[1]).all():
            raise ValueError("Positions must be valid indices within the haplotype matrix.")

        subset_haplotypes = self.haplotypes[:, positions]
        subset_positions = self.positions[positions]
        
        # Create and return a new instance, maintaining the device state and sample sets.
        return HaplotypeMatrix(
            subset_haplotypes, 
            subset_positions,
            sample_sets=self._sample_sets
        )
    
    def get_subset_from_range(self, low: int, high: int) -> "HaplotypeMatrix":
        """
        Get a subset of the haplotype matrix based on a range of positions.
        
        Parameters:
            low (int): The lower bound of the range (inclusive).
            high (int): The upper bound of the range (exclusive).
            
        Returns:
            HaplotypeMatrix: A new instance containing the subset of the haplotype matrix.
        """
        # Validate range
        if low < 0 or high > self.positions.size or low >= high:
            raise ValueError("Invalid range specified")
        
        # Check device and find indices of positions within the specified range
        positions = cp.asarray(self.positions) if self.device == 'GPU' else np.asarray(self.positions)
        indices = cp.where((positions >= low) & (positions < high))[0] if self.device == 'GPU' else np.where((positions >= low) & (positions < high))[0]

        # Create the subset of haplotypes based on the found indices
        return HaplotypeMatrix(
            self.haplotypes[:, indices], 
            self.positions[indices], 
            chrom_start=low, 
            chrom_end=high,
            sample_sets=self._sample_sets
        )
    
    def apply_biallelic_filter(self) -> "HaplotypeMatrix":
        """
        Apply biallelic filter to remove variants that are not strictly biallelic.
        
        This filter matches the behavior of moments' get_genotypes function, which uses
        is_biallelic_01() to remove variants that:
        1. Have more than 2 alleles present in the data
        2. Don't have both reference (0) and alternate (1) alleles present
        
        This is the actual filtering that moments does by default, not an AC filter.
        
        Returns:
            HaplotypeMatrix: A new HaplotypeMatrix instance with filtered variants.
        
        Note:
            This replicates moments' is_biallelic_01() filtering behavior.
        """
        if self.device == 'GPU':
            xp = cp
        else:
            xp = np
            
        # For biallelic filtering, we need to check across ALL haplotypes
        # Count alleles for each variant across all samples
        n_variants = self.num_variants
        
        # Count occurrences of each allele value
        alt_count = xp.sum(self.haplotypes == 1, axis=0)  # Count of allele 1
        ref_count = xp.sum(self.haplotypes == 0, axis=0)  # Count of allele 0  
        other_count = xp.sum(self.haplotypes >= 2, axis=0) + xp.sum(self.haplotypes < 0, axis=0)  # Count of other alleles (2+, -1, etc.)
        
        # A variant is biallelic if:
        # 1. No other alleles (> 1 or < 0) are present
        # 2. Both reference (0) and alternate (1) alleles are present
        is_biallelic = (other_count == 0) & (ref_count > 0) & (alt_count > 0)
        
        keep_mask = is_biallelic
        
        # Get indices of variants to keep
        keep_indices = xp.where(keep_mask)[0]
        
        # Create filtered HaplotypeMatrix
        filtered_haplotypes = self.haplotypes[:, keep_indices]
        filtered_positions = self.positions[keep_indices]
        
        # Update chromosome boundaries if needed
        if len(keep_indices) > 0:
            new_chrom_start = int(filtered_positions[0].get()) if self.device == 'GPU' else int(filtered_positions[0])
            new_chrom_end = int(filtered_positions[-1].get()) if self.device == 'GPU' else int(filtered_positions[-1])
        else:
            new_chrom_start = self.chrom_start
            new_chrom_end = self.chrom_end
        
        # Create new instance with same sample sets
        filtered_matrix = HaplotypeMatrix(
            filtered_haplotypes,
            filtered_positions,
            chrom_start=new_chrom_start,
            chrom_end=new_chrom_end,
            sample_sets=self._sample_sets
        )
        
        return filtered_matrix
        
    ####### Missing data methods #######
    def is_missing(self, axis=None):
        """
        Detect missing calls (-1 values).
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to compute. If None, returns boolean array same shape as haplotypes.
            If 0, returns missing per variant. If 1, returns missing per sample.
            
        Returns
        -------
        missing : array
            Boolean array indicating missing data
        """
        if self.device == 'GPU':
            missing = self.haplotypes < 0
            if axis is not None:
                return cp.any(missing, axis=axis)
            return missing
        else:
            missing = self.haplotypes < 0
            if axis is not None:
                return np.any(missing, axis=axis)
            return missing
    
    def is_called(self, axis=None):
        """
        Detect valid (non-missing) calls.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to compute. If None, returns boolean array same shape as haplotypes.
            If 0, returns called per variant. If 1, returns called per sample.
            
        Returns
        -------
        called : array
            Boolean array indicating valid data
        """
        return ~self.is_missing(axis=axis)
    
    def count_missing(self, axis=None):
        """
        Count missing calls per variant or sample.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to count. If None, returns total count.
            If 0, returns count per variant. If 1, returns count per sample.
            
        Returns
        -------
        count : int or array
            Count of missing calls
        """
        missing = self.is_missing()
        if self.device == 'GPU':
            return cp.sum(missing, axis=axis)
        else:
            return np.sum(missing, axis=axis)
    
    def count_called(self, axis=None):
        """
        Count valid calls per variant or sample.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to count. If None, returns total count.
            If 0, returns count per variant. If 1, returns count per sample.
            
        Returns
        -------
        count : int or array
            Count of valid calls
        """
        called = self.is_called()
        if self.device == 'GPU':
            return cp.sum(called, axis=axis)
        else:
            return np.sum(called, axis=axis)
    
    def get_span(self, span_denominator='total'):
        """
        Get the genomic span for normalization calculations.
        
        Parameters
        ----------
        span_denominator : str
            'total' - Use total genomic span (chrom_end - chrom_start)
            'sites' - Use number of variant sites
            'callable' - Use span from first to last variant position
            
        Returns
        -------
        span : int
            The span to use for normalization
        """
        if span_denominator == 'total':
            if self.chrom_start is not None and self.chrom_end is not None:
                return self.chrom_end - self.chrom_start
            else:
                # Fall back to callable span
                span_denominator = 'callable'
        
        if span_denominator == 'sites':
            return self.num_variants
        
        if span_denominator == 'callable':
            if self.device == 'GPU':
                if len(self.positions) > 0:
                    return int((cp.max(self.positions) - cp.min(self.positions)).get()) + 1
                else:
                    return 0
            else:
                if len(self.positions) > 0:
                    return int(np.max(self.positions) - np.min(self.positions)) + 1
                else:
                    return 0
        
        raise ValueError(f"Invalid span_denominator: {span_denominator}")
    
    def filter_variants_by_missing(self, max_missing_freq=0.1):
        """
        Return a new HaplotypeMatrix with variants filtered by missing data frequency.
        
        Parameters
        ----------
        max_missing_freq : float
            Maximum allowed frequency of missing data per variant
            
        Returns
        -------
        filtered : HaplotypeMatrix
            New HaplotypeMatrix with filtered variants
        """
        missing_freq = self.count_missing(axis=0) / self.num_haplotypes
        if self.device == 'GPU':
            valid_mask = missing_freq <= max_missing_freq
            valid_indices = cp.where(valid_mask)[0]
            return self.get_subset(valid_indices)
        else:
            valid_mask = missing_freq <= max_missing_freq
            valid_indices = np.where(valid_mask)[0]
            return self.get_subset(valid_indices)
    
    def summarize_missing_data(self):
        """
        Get summary statistics about missing data patterns.
        
        Returns
        -------
        summary : dict
            Dictionary with missing data statistics
        """
        total_missing = self.count_missing()
        total_calls = self.num_haplotypes * self.num_variants
        missing_per_variant = self.count_missing(axis=0)
        missing_per_sample = self.count_missing(axis=1)
        
        if self.device == 'GPU':
            return {
                'total_missing_calls': int(total_missing.get()),
                'total_calls': total_calls,
                'missing_freq_overall': float((total_missing / total_calls).get()),
                'variants_with_no_missing': int(cp.sum(missing_per_variant == 0).get()),
                'samples_with_no_missing': int(cp.sum(missing_per_sample == 0).get()),
                'max_missing_per_variant': int(cp.max(missing_per_variant).get()),
                'max_missing_per_sample': int(cp.max(missing_per_sample).get())
            }
        else:
            return {
                'total_missing_calls': int(total_missing),
                'total_calls': total_calls,
                'missing_freq_overall': float(total_missing / total_calls),
                'variants_with_no_missing': int(np.sum(missing_per_variant == 0)),
                'samples_with_no_missing': int(np.sum(missing_per_sample == 0)),
                'max_missing_per_variant': int(np.max(missing_per_variant)),
                'max_missing_per_sample': int(np.max(missing_per_sample))
            }
        
    ####### some polymorphism statistics #######
    def allele_frequency_spectrum(self) -> cp.ndarray:
        """
        Calculate the allele frequency spectrum for a haplotype matrix.
        
        Note: This method is deprecated. Use diversity.allele_frequency_spectrum() instead.
        """
        from . import diversity
        return diversity.allele_frequency_spectrum(self)
    
    def diversity(self, span_normalize: bool = True) -> float:
        """
        Calculate the nucleotide diversity (π) for the haplotype matrix.

        Note: This method is deprecated. Use diversity.pi() instead.

        Parameters:
            span_normalize (bool, optional): If True, the result is normalized by the span of the haplotype matrix. Defaults to True.

        Returns:
            float: The nucleotide diversity (π) for the haplotype matrix.
        """
        from . import diversity
        return diversity.pi(self, span_normalize=span_normalize)
        
    def watersons_theta(self, span_normalize: bool = True) -> float:
        """
        Calculate Waterson's theta for the haplotype matrix.
        
        Note: This method is deprecated. Use diversity.theta_w() instead.
        """
        from . import diversity
        return diversity.theta_w(self, span_normalize=span_normalize)
    
    def Tajimas_D(self) -> float:
        """
        Calculate Tajima's D for the haplotype matrix.
        
        Note: This method is deprecated. Use diversity.tajimas_d() instead.
        """
        from . import diversity
        return diversity.tajimas_d(self)


    def pairwise_LD_v(self) -> cp.ndarray:
        """
        Optimized pairwise linkage disequilibrium (D statistic) computation 
        using matrix multiplication for CuPy acceleration.
        """
        # Ensure data is on GPU
        if self.device == 'CPU':
            self.transfer_to_gpu()

        n_haplotypes = self.num_haplotypes
        
        # Compute allele frequencies for all variants
        p = cp.sum(self.haplotypes, axis=0) / n_haplotypes  # (n_variants,)
        
        # Compute p_AB for all variant pairs using matrix multiplication
        p_AB = (self.haplotypes.T @ self.haplotypes) / n_haplotypes  # (n_variants, n_variants)
        
        # Compute outer product of allele frequencies: p_A * p_B
        p_Ap_B = cp.outer(p, p)  # (n_variants, n_variants)
        # Compute D = p_AB - p_A * p_B
        D = p_AB - p_Ap_B
        # set the diagonal to 0
        cp.fill_diagonal(D, 0)

        return D

    def pairwise_r2(self) -> cp.ndarray:
        """
        Calculate the pairwise r2 (correlation coefficient) for all pairs of variants
        in the haplotype matrix.
        """
        # Ensure data is on GPU
        if self.device == 'CPU':
            self.transfer_to_gpu()
        
        n_haplotypes = self.num_haplotypes
        
        # Compute allele frequencies for all variants
        p = cp.sum(self.haplotypes, axis=0) / n_haplotypes  # (n_variants,)
        
        # Compute p_AB for all variant pairs using matrix multiplication
        p_AB = (self.haplotypes.T @ self.haplotypes) / n_haplotypes  # (n_variants, n_variants)
        
        # Compute outer product of allele frequencies: p_A * p_B
        p_Ap_B = cp.outer(p, p)  # (n_variants, n_variants)
        # Compute D = p_AB - p_A * p_B
        D = p_AB - p_Ap_B
       
        # compute the denominator: p_A * (1 - p_A) * p_B * (1 - p_B)
        denom_squared = cp.outer(p * (1 - p), p * (1 - p))  
        # compute r2
        r2 = cp.where(denom_squared > 0, (D ** 2) / denom_squared, 0)
        
        # set the diagonal to 0
        cp.fill_diagonal(r2, 0)

        return r2


    def tally_gpu_haplotypes(self, pop=None):
        """
        GPU implementation of computing pairwise haplotype tallies.
        Automatically detects and handles missing data if present.
        
        Parameters:
            pop (str, optional): Population key from sample_sets to use. If None, uses all samples.
        
        Returns:
            tuple: (counts, n_valid) where:
                - counts: Array of shape (#pairs, 4) containing [n11, n10, n01, n00] for each variant pair
                - n_valid: Array of shape (#pairs,) containing the number of valid haplotypes for each pair
                          or None if no missing data is present
        """
        # Ensure data is on the GPU
        if self.device == 'CPU':
            self.transfer_to_gpu()

        # Get the appropriate subset of haplotypes
        if pop is not None:
            if self._sample_sets is None:
                raise ValueError("sample_sets must be defined to use pop parameter")
            if pop not in self._sample_sets:
                raise KeyError(f"Population key {pop} must exist in sample_sets")
            X = self.haplotypes[self._sample_sets[pop], :]
        else:
            X = self.haplotypes
        
        # Check if there's any missing data
        has_missing = cp.any(X == -1)
        
        if has_missing:
            # Use the missing data implementation
            return self._tally_gpu_haplotypes_with_missing_impl(X)
        else:
            # Use the faster non-missing implementation
            m = X.shape[1]  # number of variants
            
            # Count ones per variant
            ones_per_variant = cp.sum(X, axis=0)
            
            # Compute n11 matrix
            n11_mat = X.T @ X
            
            # Get indices for upper triangle
            idx_i, idx_j = cp.triu_indices(m, k=1)
            
            # Compute counts
            n11_pairs = n11_mat[idx_i, idx_j]
            n10_pairs = ones_per_variant[idx_i] - n11_pairs
            n01_pairs = ones_per_variant[idx_j] - n11_pairs
            n00_pairs = X.shape[0] - (n11_pairs + n10_pairs + n01_pairs)
            
            # Stack all results
            counts = cp.stack([n11_pairs, n10_pairs, n01_pairs, n00_pairs], axis=1)
            
            return counts, None
    
    def _tally_gpu_haplotypes_with_missing_impl(self, X):
        """
        Internal implementation of computing pairwise haplotype tallies with missing data support.
        
        For each variant pair, only counts haplotypes where both variants are non-missing.
        Missing data is encoded as -1 in the haplotype matrix.
        
        Parameters:
            X (cp.ndarray): Haplotype matrix to process
        
        Returns:
            tuple: (counts, n_valid) where:
                - counts: Array of shape (#pairs, 4) containing [n11, n10, n01, n00] for each variant pair
                - n_valid: Array of shape (#pairs,) containing the number of valid haplotypes for each pair
        """
        
        m = X.shape[1]  # number of variants
        n_haps = X.shape[0]  # number of haplotypes
        
        # Create missing mask for each variant (True where data is missing)
        missing_mask = (X == -1)
        
        # Get indices for upper triangle
        idx_i, idx_j = cp.triu_indices(m, k=1)
        n_pairs = len(idx_i)
        
        # Initialize arrays for results
        n11_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        n10_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        n01_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        n00_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        n_valid = cp.zeros(n_pairs, dtype=cp.int32)
        
        # Process pairs (this could be optimized with custom kernels)
        for pair_idx in range(n_pairs):
            i = idx_i[pair_idx]
            j = idx_j[pair_idx]
            
            # Create valid mask for this pair (where both variants are non-missing)
            valid_mask = ~(missing_mask[:, i] | missing_mask[:, j])
            n_valid[pair_idx] = cp.sum(valid_mask)
            
            if n_valid[pair_idx] > 0:
                # Extract valid haplotypes for this pair
                valid_haps_i = X[valid_mask, i]
                valid_haps_j = X[valid_mask, j]
                
                # Count haplotype combinations
                n11_pairs[pair_idx] = cp.sum((valid_haps_i == 1) & (valid_haps_j == 1))
                n10_pairs[pair_idx] = cp.sum((valid_haps_i == 1) & (valid_haps_j == 0))
                n01_pairs[pair_idx] = cp.sum((valid_haps_i == 0) & (valid_haps_j == 1))
                n00_pairs[pair_idx] = cp.sum((valid_haps_i == 0) & (valid_haps_j == 0))
        
        # Stack all results
        counts = cp.stack([n11_pairs, n10_pairs, n01_pairs, n00_pairs], axis=1)
        
        return counts, n_valid
    
    def tally_gpu_haplotypes_two_pops_with_missing(self, pop1: str, pop2: str):
        """
        GPU implementation of computing pairwise haplotype tallies for two populations with missing data support.
        
        For each variant pair, only counts haplotypes where both variants are non-missing in both populations.
        Missing data is encoded as -1 in the haplotype matrix.
        
        Parameters:
            pop1 (str): First population key from sample_sets
            pop2 (str): Second population key from sample_sets
        
        Returns:
            tuple: (counts, n_valid1, n_valid2) where:
                - counts: Array of shape (#pairs, 8) containing counts for both populations
                  [n11_1, n10_1, n01_1, n00_1, n11_2, n10_2, n01_2, n00_2]
                - n_valid1: Array of shape (#pairs,) with valid haplotypes for pop1
                - n_valid2: Array of shape (#pairs,) with valid haplotypes for pop2
        """
        import cupy as cp
        
        if self.device == 'CPU':
            self.transfer_to_gpu()
        
        # Check populations
        if self._sample_sets is None:
            raise ValueError("sample_sets must be defined to use this function")
        if pop1 not in self._sample_sets or pop2 not in self._sample_sets:
            raise KeyError(f"Population keys {pop1} and {pop2} must exist in sample_sets")
        
        # Get indices for each population
        idx1 = self._sample_sets[pop1]
        idx2 = self._sample_sets[pop2]
        
        # Extract submatrices for each population
        X1 = self.haplotypes[idx1, :]
        X2 = self.haplotypes[idx2, :]
        m = self.num_variants
        
        # Create missing masks for each population
        missing_mask1 = (X1 == -1)
        missing_mask2 = (X2 == -1)
        
        # Get indices for upper triangle
        idx_i, idx_j = cp.triu_indices(m, k=1)
        n_pairs = len(idx_i)
        
        # Initialize arrays for results
        counts = cp.zeros((n_pairs, 8), dtype=cp.int32)
        n_valid1 = cp.zeros(n_pairs, dtype=cp.int32)
        n_valid2 = cp.zeros(n_pairs, dtype=cp.int32)
        
        # Process pairs (this could be optimized with custom kernels)
        for pair_idx in range(n_pairs):
            i = idx_i[pair_idx]
            j = idx_j[pair_idx]
            
            # Create valid masks for each population
            valid_mask1 = ~(missing_mask1[:, i] | missing_mask1[:, j])
            valid_mask2 = ~(missing_mask2[:, i] | missing_mask2[:, j])
            n_valid1[pair_idx] = cp.sum(valid_mask1)
            n_valid2[pair_idx] = cp.sum(valid_mask2)
            
            # Population 1 counts
            if n_valid1[pair_idx] > 0:
                valid_haps1_i = X1[valid_mask1, i]
                valid_haps1_j = X1[valid_mask1, j]
                counts[pair_idx, 0] = cp.sum((valid_haps1_i == 1) & (valid_haps1_j == 1))  # n11
                counts[pair_idx, 1] = cp.sum((valid_haps1_i == 1) & (valid_haps1_j == 0))  # n10
                counts[pair_idx, 2] = cp.sum((valid_haps1_i == 0) & (valid_haps1_j == 1))  # n01
                counts[pair_idx, 3] = cp.sum((valid_haps1_i == 0) & (valid_haps1_j == 0))  # n00
            
            # Population 2 counts
            if n_valid2[pair_idx] > 0:
                valid_haps2_i = X2[valid_mask2, i]
                valid_haps2_j = X2[valid_mask2, j]
                counts[pair_idx, 4] = cp.sum((valid_haps2_i == 1) & (valid_haps2_j == 1))  # n11
                counts[pair_idx, 5] = cp.sum((valid_haps2_i == 1) & (valid_haps2_j == 0))  # n10
                counts[pair_idx, 6] = cp.sum((valid_haps2_i == 0) & (valid_haps2_j == 1))  # n01
                counts[pair_idx, 7] = cp.sum((valid_haps2_i == 0) & (valid_haps2_j == 0))  # n00
        
        return counts, n_valid1, n_valid2

    def tally_gpu_haplotypes_two_pops(self, pop1: str, pop2: str):
        """
        GPU version of tallying haplotype counts between all pairs of variants for two populations.
        Automatically detects and handles missing data if present.
        
        Returns:
            tuple: (counts, n_valid1, n_valid2) where:
                - counts: Array of shape (#pairs, 8) containing counts for both populations
                - n_valid1: Array of shape (#pairs,) with valid haplotypes for pop1 (or None if no missing data)
                - n_valid2: Array of shape (#pairs,) with valid haplotypes for pop2 (or None if no missing data)
        """
        import cupy as cp
        
        if self.device == 'CPU':
            self.transfer_to_gpu()
        
        # Check populations
        if self._sample_sets is None:
            raise ValueError("sample_sets must be defined to use this function")
        if pop1 not in self._sample_sets or pop2 not in self._sample_sets:
            raise KeyError(f"Population keys {pop1} and {pop2} must exist in sample_sets")
        
        # Get indices for each population
        idx1 = self._sample_sets[pop1]
        idx2 = self._sample_sets[pop2]
        
        # Extract submatrices for each population
        X1 = self.haplotypes[idx1, :]
        X2 = self.haplotypes[idx2, :]
        
        # Check if there's any missing data
        has_missing = cp.any(self.haplotypes == -1)
        
        if has_missing:
            # Use the missing data implementation
            return self.tally_gpu_haplotypes_two_pops_with_missing(pop1, pop2)
        else:
            # Use the faster non-missing implementation
            n1 = len(idx1)
            n2 = len(idx2)
            m = self.num_variants
            
            # Count ones per variant for each population
            ones_per_variant1 = cp.sum(X1, axis=0)
            ones_per_variant2 = cp.sum(X2, axis=0)
            
            # Compute n11 matrices for each population
            n11_mat1 = X1.T @ X1
            n11_mat2 = X2.T @ X2
            
            # Get indices for upper triangle only
            idx_i, idx_j = cp.triu_indices(m, k=1)
            
            # Compute counts for population 1
            n11_pairs1 = n11_mat1[idx_i, idx_j]
            n10_pairs1 = ones_per_variant1[idx_i] - n11_pairs1
            n01_pairs1 = ones_per_variant1[idx_j] - n11_pairs1
            n00_pairs1 = n1 - (n11_pairs1 + n10_pairs1 + n01_pairs1)
            
            # Compute counts for population 2
            n11_pairs2 = n11_mat2[idx_i, idx_j]
            n10_pairs2 = ones_per_variant2[idx_i] - n11_pairs2
            n01_pairs2 = ones_per_variant2[idx_j] - n11_pairs2
            n00_pairs2 = n2 - (n11_pairs2 + n10_pairs2 + n01_pairs2)
            
            # Stack all results
            counts = cp.stack([
                n11_pairs1, n10_pairs1, n01_pairs1, n00_pairs1,
                n11_pairs2, n10_pairs2, n01_pairs2, n00_pairs2
            ], axis=1)
            
            return counts, None, None

    # TODO: this is not correct
    def count_haplotypes_between_populations_gpu(self, missing: bool = False) -> dict:
        """
        GPU implementation of counting haplotype tallies between different populations defined
        in self.sample_sets. The haplotype matrix is assumed to contain data for multiple populations
        (i.e. sample_sets is a dict mapping population names to sets of haplotype indices).

        For each unique pair of populations (pop1, pop2), let:
            subX1 = haplotype data for pop1 with shape (n1, m)
            subX2 = haplotype data for pop2 with shape (n2, m)
            ones1   = count of allele 1 in subX1, for each of the m variants
            ones2   = count of allele 1 in subX2, for each of the m variants

        Then, for every variant pair (i, j) the tallies for the 2x2 table are computed as:
            - n11 = ones1[i] * ones2[j]
            - n10 = ones1[i] * (n2 - ones2[j])
            - n01 = (n1 - ones1[i]) * ones2[j]
            - n00 = (n1 - ones1[i]) * (n2 - ones2[j])
            
        The tallies for each population pair are returned in a dictionary, with keys given by
        (pop1, pop2) tuples and values a CuPy array of shape (m*m, 4) (one row per variant pair).
        
        Missing data is not supported.

        Parameters:
            missing (bool): If True, raises NotImplementedError (missing data not implemented).

        Returns:
            dict: A dictionary mapping (pop1, pop2) to a CuPy array of tallies.
        """
        if missing:
            raise NotImplementedError("Missing data support is not implemented in this function.")
        
        # Ensure the haplotype data is on the GPU.
        if self.device == 'CPU':
            self.transfer_to_gpu()
        
        X = self.haplotypes  # Shape: (n_total, m)
        m = self.num_variants
        pops = self.sample_sets
        
        if len(pops) < 2:
            raise ValueError("At least two populations are required in sample_sets for between-population tallies.")
        
        results = {}
        pop_keys = sorted(pops.keys())
        for i in range(len(pop_keys)):
            for j in range(i + 1, len(pop_keys)):
                pop1, pop2 = pop_keys[i], pop_keys[j]
                
                # Get haplotype indices for each population.
                indices1 = sorted(list(pops[pop1]))
                indices2 = sorted(list(pops[pop2]))
                if len(indices1) == 0 or len(indices2) == 0:
                    continue  # Skip if one of the populations has no samples.
                
                # Extract the submatrices: rows corresponding to each population.
                subX1 = X[indices1, :]  # shape: (n1, m)
                subX2 = X[indices2, :]  # shape: (n2, m)
                n1 = subX1.shape[0]
                n2 = subX2.shape[0]
                
                # Compute the number of ones for each variant.
                ones1 = cp.sum(subX1, axis=0).astype(cp.int32)  # shape: (m,)
                ones2 = cp.sum(subX2, axis=0).astype(cp.int32)  # shape: (m,)
                
                # Compute cross-population tallies using outer products:
                n11_matrix = ones1.reshape(m, 1) * ones2.reshape(1, m)
                n10_matrix = ones1.reshape(m, 1) * ((n2 - ones2).reshape(1, m))
                n01_matrix = ( (n1 - ones1).reshape(m, 1) ) * ones2.reshape(1, m)
                n00_matrix = (n1 - ones1).reshape(m, 1) * ((n2 - ones2).reshape(1, m))
                
                # Stack the tallies along a new third axis and reshape into a 2D array.
                tallies = cp.stack([n11_matrix, n10_matrix, n01_matrix, n00_matrix], axis=2).reshape(-1, 4)
                results[(pop1, pop2)] = tallies
        
        return results

    def compute_ld_statistics_gpu_single_pop(self, bp_bins, raw=False, ac_filter=True):
        """
        GPU-based implementation of computing LD statistics for a single population using tallies
        from tally_gpu_haplotypes, followed by binning by base-pair distance.
        
        Steps:
          1. Compute pairwise haplotype tallies using tally_gpu_haplotypes. Each row in the resulting 
             CuPy array represents the 2x2 haplotype count table [n11, n10, n01, n00] for a variant pair.
          2. Compute the positions for each variant pair from the upper-triangle indices (using cp.triu_indices)
             so that distance = pos[j] - pos[i].
          3. Using the GPU stats module (ld_statistics), compute for each pair: D, D^2, Dz, and π₂.
          4. Bin the variant pairs by distance using the provided bp_bins.
          5. Depending on the `raw` flag:
               * If raw is False (default), return the mean (averaged over all pairs in the bin) of each statistic.
               * If raw is True, return the raw sums for each statistic (which should match the Moments aggregation).
        
        Parameters:
            bp_bins (array-like): Array of bin boundaries in base pairs (e.g. [0, 50, 100, ...]).
            raw (bool): If True, return the raw sums aggregated in each bin, rather than mean values.
            ac_filter (bool): If True, apply biallelic filtering (matching moments' is_biallelic_01 behavior).
        
        Returns:
            dict: A dictionary mapping each bin (tuple: (bin_start, bin_end)) to a tuple:
                  (D2, Dz, pi2, D). If raw is False these values are averaged over pairs; if raw is True
                  they are the raw sums.
        """
        # Apply biallelic filter if requested (matches moments' default behavior)
        if ac_filter:
            # Apply biallelic filtering to match moments' is_biallelic_01() behavior
            filtered_self = self.apply_biallelic_filter()
            # Use the filtered matrix for computation
            return filtered_self.compute_ld_statistics_gpu_single_pop(
                bp_bins=bp_bins, raw=raw, ac_filter=False
            )

        # Ensure the matrix (and positions) are on the GPU.
        if self.device == 'CPU':
            self.transfer_to_gpu()

        # Get the haplotype data and positions.
        X = self.haplotypes  # shape: (n_haplotypes, num_variants)
        pos = self.positions  # assumed to be sorted; shape: (num_variants,)
        m = self.num_variants

        # Ensure positions are on the GPU.
        import cupy as cp
        if not isinstance(pos, cp.ndarray):
            pos = cp.array(pos)

        # Compute pairwise tallies for all variant pairs using our efficient GPU routine.
        # tally_gpu_haplotypes now auto-detects missing data and returns tuple
        counts, n_valid = self.tally_gpu_haplotypes()

        # Get the variant-pair indices corresponding to the tallies.
        idx_i, idx_j = cp.triu_indices(m, k=1)
        # Compute the physical distance between variant pairs.
        distances = pos[idx_j] - pos[idx_i]  # shape: (#pairs,)

        # Import the new unified LD statistics module.
        from pg_gpu import ld_statistics

        # Compute the LD statistics for all variant pairs.
        # New API handles missing data automatically
        DD_vals  = ld_statistics.dd(counts, n_valid=n_valid)   # shape: (#pairs,)
        Dz_vals  = ld_statistics.dz(counts, n_valid=n_valid)   # shape: (#pairs,)
        pi2_vals = ld_statistics.pi2(counts, n_valid=n_valid)  # shape: (#pairs,)

        # Convert bp_bins to a CuPy array.
        bp_bins_cp = cp.array(bp_bins)
        # Define bins as intervals [bp_bins[i], bp_bins[i+1]); use cp.digitize.
        # cp.digitize returns an index in [0, len(bp_bins_cp)] so subtract one.
        bin_inds = cp.digitize(distances, bp_bins_cp) - 1
        
        # Only consider pairs that fall into a valid bin interval.
        n_bins = len(bp_bins_cp) - 1
        valid_mask = (bin_inds >= 0) & (bin_inds < n_bins)
        bin_inds = bin_inds[valid_mask]
        # D_vals   = D_vals[valid_mask]
        DD_vals  = DD_vals[valid_mask]
        Dz_vals  = Dz_vals[valid_mask]
        pi2_vals = pi2_vals[valid_mask]

        # Initialize output dictionary.
        out = {}
        # Loop over each bin index.
        for i in range(n_bins):
            bin_start = float(bp_bins_cp[i].get())
            bin_end   = float(bp_bins_cp[i+1].get())
            # Get indices within this bin.
            mask = (bin_inds == i)
            count_pairs = int(cp.sum(mask).get())
            if count_pairs > 0:
                if raw:
                    sum_D2  = float(cp.sum(DD_vals[mask]).get())
                    sum_Dz  = float(cp.sum(Dz_vals[mask]).get())
                    sum_pi2 = float(cp.sum(pi2_vals[mask]).get())
                    # sum_D   = float(cp.sum(D_vals[mask]).get())
                    out[(bin_start, bin_end)] = (sum_D2, sum_Dz, sum_pi2)
                else:
                    mean_D2  = float(cp.mean(DD_vals[mask]).get())
                    mean_Dz  = float(cp.mean(Dz_vals[mask]).get())
                    mean_pi2 = float(cp.mean(pi2_vals[mask]).get())
                    # mean_D   = float(cp.mean(D_vals[mask]).get())
                    out[(bin_start, bin_end)] = (mean_D2, mean_Dz, mean_pi2)
            else:
                # For an empty bin, return zeros.
                out[(bin_start, bin_end)] = (0.0, 0.0, 0.0)

        return out
    def compute_ld_statistics_gpu_moments_compatible(self, bp_bins, pop1: str, pop2: str, raw=False):
        """
        GPU-based LD statistics computation that exactly matches moments output format.
        
        This method computes all statistics in the order expected by moments:
        DD_0_0, DD_0_1, DD_1_1, Dz_0_0_0, Dz_0_0_1, Dz_0_1_1, Dz_1_0_0, Dz_1_0_1, Dz_1_1_1,
        pi2_0_0_0_0, pi2_0_0_0_1, pi2_0_0_1_1, pi2_0_1_0_1, pi2_0_1_1_1, pi2_1_1_1_1
        
        Parameters:
            bp_bins: Array of bin boundaries
            pop1: Name of first population (maps to index 0 in moments)
            pop2: Name of second population (maps to index 1 in moments)
            raw: If True, return raw sums; if False, return means
            
        Returns:
            dict: Dictionary mapping bin ranges to tuples of statistics
        """
        from .compute_ld_moments_compatible import compute_ld_statistics_moments_compatible
        return compute_ld_statistics_moments_compatible(self, bp_bins, pop1, pop2, raw=raw)
    
    def compute_ld_statistics_gpu_two_pops(self, bp_bins, pop1: str, pop2: str, raw=False, ac_filter=True):
        """GPU-based implementation of computing LD statistics for two populations.
        
        This method computes statistics for each variant pair and then sums them,
        matching the moments implementation approach.
        
        Parameters:
            bp_bins: Array of bin boundaries in base pairs
            pop1: Name of first population
            pop2: Name of second population
            raw: If True, return raw sums; if False, return means
            ac_filter: If True, apply biallelic filtering (matching moments' is_biallelic_01 behavior)
        """
        # Apply biallelic filter if requested (matches moments' default behavior)
        if ac_filter:
            # Apply biallelic filtering to match moments' is_biallelic_01() behavior
            filtered_self = self.apply_biallelic_filter()
            # Use the filtered matrix for computation
            return filtered_self.compute_ld_statistics_gpu_two_pops(
                bp_bins=bp_bins, pop1=pop1, pop2=pop2, raw=raw, ac_filter=False
            )
        
        # Ensure GPU setup
        if self.device == 'CPU':
            self.transfer_to_gpu()

        import cupy as cp
        from pg_gpu import ld_statistics
        from collections import OrderedDict

        # Get positions and ensure they're on GPU
        pos = self.positions
        m = self.num_variants
        if not isinstance(pos, cp.ndarray):
            pos = cp.array(pos)

        # Get indices for upper triangle
        idx_i, idx_j = cp.triu_indices(m, k=1)
        distances = pos[idx_j] - pos[idx_i]

        # Get counts - tally functions now auto-detect missing data
        # Get within-population counts
        counts_pop1, n_valid_pop1 = self.tally_gpu_haplotypes(pop=pop1)
        counts_pop2, n_valid_pop2 = self.tally_gpu_haplotypes(pop=pop2)
        
        # Get between-population counts
        counts_between, n_valid1_between, n_valid2_between = self.tally_gpu_haplotypes_two_pops(pop1, pop2)
        counts1_between = counts_between[:, :4]
        counts2_between = counts_between[:, 4:]

        # Bin the pairs
        bp_bins_cp = cp.array(bp_bins)
        bin_inds = cp.digitize(distances, bp_bins_cp) - 1
        n_bins = len(bp_bins_cp) - 1
        valid_mask = (bin_inds >= 0) & (bin_inds < n_bins)

        # Initialize output dictionary
        out = {}
        
        # Process each bin
        for i in range(n_bins):
            bin_start = float(bp_bins[i])
            bin_end = float(bp_bins[i+1])
            
            # Get mask for this bin
            mask = (bin_inds == i) & valid_mask
            
            # Get counts for pairs in this bin
            pairs_counts_pop1 = counts_pop1[mask]
            pairs_counts_pop2 = counts_pop2[mask]
            pairs_counts1_between = counts1_between[mask]
            pairs_counts2_between = counts2_between[mask]
            
            if n_valid_pop1 is not None or n_valid_pop2 is not None:
                # Also get valid sample sizes for missing data
                pairs_n_valid_pop1 = n_valid_pop1[mask] if n_valid_pop1 is not None else None
                pairs_n_valid_pop2 = n_valid_pop2[mask] if n_valid_pop2 is not None else None
                pairs_n_valid1_between = n_valid1_between[mask] if n_valid1_between is not None else None
                pairs_n_valid2_between = n_valid2_between[mask] if n_valid2_between is not None else None
            
            # Compute statistics for each pair and sum them
            # This matches the moments approach: sum(statistic(counts_i))
            if pairs_counts_pop1.shape[0] == 0:
                # No pairs in this bin
                stats_dict = OrderedDict([
                    ('DD_0_0', 0.0),
                    ('DD_0_1', 0.0),
                    ('DD_1_1', 0.0),
                    ('Dz_0_0_0', 0.0),
                    ('Dz_0_0_1', 0.0),
                    ('Dz_0_1_1', 0.0),
                    ('Dz_1_0_0', 0.0),
                    ('Dz_1_0_1', 0.0),
                    ('Dz_1_1_1', 0.0),
                    ('pi2_0_0_0_0', 0.0),
                    ('pi2_0_0_0_1', 0.0),
                    ('pi2_0_0_1_1', 0.0),
                    ('pi2_0_1_0_1', 0.0),
                    ('pi2_0_1_1_1', 0.0),
                    ('pi2_1_1_1_1', 0.0)
                ])
            else:
                # Use vectorized operations - the stats functions handle arrays
                # Compute all statistics at once for all pairs
                # Map population indices: 0 = pop1, 1 = pop2
                
                # Use the masked valid sample sizes directly (don't reassign the original variables)
                n_valid1 = pairs_n_valid_pop1 if n_valid_pop1 is not None else None
                n_valid2 = pairs_n_valid_pop2 if n_valid_pop2 is not None else None
                n_valid1_between_masked = pairs_n_valid1_between if n_valid1_between is not None else None
                n_valid2_between_masked = pairs_n_valid2_between if n_valid2_between is not None else None
                
                # DD statistics - use new unified API
                DD_0_0_vec = ld_statistics.dd(pairs_counts_pop1, n_valid=n_valid1)
                DD_0_1_vec = ld_statistics.dd(counts_between[mask], populations=(0, 1), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                DD_1_1_vec = ld_statistics.dd(pairs_counts_pop2, n_valid=n_valid2)
                
                # Dz statistics - use new unified API
                Dz_0_0_0_vec = ld_statistics.dz(pairs_counts_pop1, n_valid=n_valid1)
                
                # Dz_0_0_1: average of Dz(0,0,1) and Dz(0,1,0)
                Dz_0_0_1_part1 = ld_statistics.dz(counts_between[mask], populations=(0, 0, 1), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                Dz_0_0_1_part2 = ld_statistics.dz(counts_between[mask], populations=(0, 1, 0), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                Dz_0_0_1_vec = 0.5 * Dz_0_0_1_part1 + 0.5 * Dz_0_0_1_part2
                
                Dz_0_1_1_vec = ld_statistics.dz(counts_between[mask], populations=(0, 1, 1), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                Dz_1_0_0_vec = ld_statistics.dz(counts_between[mask], populations=(1, 0, 0), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                
                # Dz_1_0_1: average of Dz(1,0,1) and Dz(1,1,0)
                Dz_1_0_1_part1 = ld_statistics.dz(counts_between[mask], populations=(1, 0, 1), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                Dz_1_0_1_part2 = ld_statistics.dz(counts_between[mask], populations=(1, 1, 0), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                Dz_1_0_1_vec = 0.5 * Dz_1_0_1_part1 + 0.5 * Dz_1_0_1_part2
                
                Dz_1_1_1_vec = ld_statistics.dz(pairs_counts_pop2, n_valid=n_valid2)
                
                # pi2 statistics - use new unified API
                pi2_0_0_0_0_vec = ld_statistics.pi2(pairs_counts_pop1, n_valid=n_valid1)
                pi2_0_0_0_1_vec = ld_statistics.pi2(counts_between[mask], populations=(0, 0, 0, 1), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                pi2_0_0_1_1_vec = ld_statistics.pi2(counts_between[mask], populations=(0, 0, 1, 1), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                pi2_0_1_0_1_vec = ld_statistics.pi2(counts_between[mask], populations=(0, 1, 0, 1), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                pi2_0_1_1_1_vec = ld_statistics.pi2(counts_between[mask], populations=(0, 1, 1, 1), n_valid=(n_valid1_between_masked, n_valid2_between_masked))
                pi2_1_1_1_1_vec = ld_statistics.pi2(pairs_counts_pop2, n_valid=n_valid2)
                
                if raw:
                    # Sum the statistics for all pairs
                    stats_dict = OrderedDict([
                        ('DD_0_0', float(cp.sum(DD_0_0_vec).get())),
                        ('DD_0_1', float(cp.sum(DD_0_1_vec).get())),
                        ('DD_1_1', float(cp.sum(DD_1_1_vec).get())),
                        ('Dz_0_0_0', float(cp.sum(Dz_0_0_0_vec).get())),
                        ('Dz_0_0_1', float(cp.sum(Dz_0_0_1_vec).get())),
                        ('Dz_0_1_1', float(cp.sum(Dz_0_1_1_vec).get())),
                        ('Dz_1_0_0', float(cp.sum(Dz_1_0_0_vec).get())),
                        ('Dz_1_0_1', float(cp.sum(Dz_1_0_1_vec).get())),
                        ('Dz_1_1_1', float(cp.sum(Dz_1_1_1_vec).get())),
                        ('pi2_0_0_0_0', float(cp.sum(pi2_0_0_0_0_vec).get())),
                        ('pi2_0_0_0_1', float(cp.sum(pi2_0_0_0_1_vec).get())),
                        ('pi2_0_0_1_1', float(cp.sum(pi2_0_0_1_1_vec).get())),
                        ('pi2_0_1_0_1', float(cp.sum(pi2_0_1_0_1_vec).get())),
                        ('pi2_0_1_1_1', float(cp.sum(pi2_0_1_1_1_vec).get())),
                        ('pi2_1_1_1_1', float(cp.sum(pi2_1_1_1_1_vec).get()))
                    ])
                else:
                    # Average the statistics
                    stats_dict = OrderedDict([
                        ('DD_0_0', float(cp.mean(DD_0_0_vec).get())),
                        ('DD_0_1', float(cp.mean(DD_0_1_vec).get())),
                        ('DD_1_1', float(cp.mean(DD_1_1_vec).get())),
                        ('Dz_0_0_0', float(cp.mean(Dz_0_0_0_vec).get())),
                        ('Dz_0_0_1', float(cp.mean(Dz_0_0_1_vec).get())),
                        ('Dz_0_1_1', float(cp.mean(Dz_0_1_1_vec).get())),
                        ('Dz_1_0_0', float(cp.mean(Dz_1_0_0_vec).get())),
                        ('Dz_1_0_1', float(cp.mean(Dz_1_0_1_vec).get())),
                        ('Dz_1_1_1', float(cp.mean(Dz_1_1_1_vec).get())),
                        ('pi2_0_0_0_0', float(cp.mean(pi2_0_0_0_0_vec).get())),
                        ('pi2_0_0_0_1', float(cp.mean(pi2_0_0_0_1_vec).get())),
                        ('pi2_0_0_1_1', float(cp.mean(pi2_0_0_1_1_vec).get())),
                        ('pi2_0_1_0_1', float(cp.mean(pi2_0_1_0_1_vec).get())),
                        ('pi2_0_1_1_1', float(cp.mean(pi2_0_1_1_1_vec).get())),
                        ('pi2_1_1_1_1', float(cp.mean(pi2_1_1_1_1_vec).get()))
                    ])
            
            out[(bin_start, bin_end)] = stats_dict

        return out
