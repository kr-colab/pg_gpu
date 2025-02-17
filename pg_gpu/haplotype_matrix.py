import cupy as cp
import numpy as np
import allel
import tskit

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
            dict: A dictionary mapping population names to sets of haplotype indices.
                  If _sample_sets was not specified at construction, returns a default 
                  dictionary with a single key 'all' containing all haplotype indices.
        """
        if self._sample_sets is None:
            # All haplotypes belong to a single population labeled "all"
            return {"all": set(range(self.haplotypes.shape[0]))}
        return self._sample_sets

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
        # Ensure positions are valid indices
        positions = cp.asarray(positions) if self.device == 'GPU' else np.asarray(positions)
        if not (positions >= 0).all() or not (positions < self.haplotypes.shape[1]).all():
            raise ValueError("Positions must be valid indices within the haplotype matrix.")

        subset_haplotypes = self.haplotypes[:, positions]
        subset_positions = self.positions[positions]
        
        # Create and return a new instance, maintaining the device state.
        return HaplotypeMatrix(subset_haplotypes, subset_positions)
    
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
            chrom_end=high
            )
        
        
        
    ####### some polymorphism statistics #######
    def allele_frequency_spectrum(self) -> cp.ndarray:
        """
        Calculate the allele frequency spectrum for a haplotype matrix.
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()
        n_haplotypes = self.num_haplotypes
        freqs = cp.sum(cp.nan_to_num(self.haplotypes, nan=0).astype(cp.int32), axis=0)
        return cp.histogram(freqs, bins=cp.arange(n_haplotypes+1))[0]
    
    def diversity(self, span_normalize: bool = True) -> float:
        """
        Calculate the nucleotide diversity (π) for the haplotype matrix.

        This method calculates the nucleotide diversity (π) for the haplotype matrix. π is a measure of the genetic variation within a population. It is defined as the average number of nucleotide differences per site between two randomly chosen DNA sequences from the population.

        Parameters:
            span_normalize (bool, optional): If True, the result is normalized by the span of the haplotype matrix. Defaults to True.

        Returns:
            float: The nucleotide diversity (π) for the haplotype matrix. If span_normalize is True, the result is normalized by the span of the haplotype matrix.
        """
     
        afs = self.allele_frequency_spectrum()
        n_haplotypes = self.num_haplotypes
        # Compute the weight factor for each allele frequency
        i = cp.arange(1, n_haplotypes, dtype=cp.float64)  # Allele counts from 1 to n-1
        weight = (2 * i * (n_haplotypes - i)) / (n_haplotypes * (n_haplotypes - 1))
    
        # Compute π as a weighted sum over the allele frequency spectrum
        pi = cp.sum((weight * afs[1:]).astype(cp.float64))
        if span_normalize:
            span = cp.float64(self.chrom_end - self.chrom_start)
            return float(pi / span)
        return float(pi)
        
    def watersons_theta(self, span_normalize: bool = True) -> float:
        """
        Calculate Waterson's theta for the haplotype matrix.
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()
        n_haplotypes = self.num_haplotypes
        # Compute the harmonic number a_n
        a1 = cp.sum((1.0 / cp.arange(1, n_haplotypes, dtype=cp.float64)))
        theta = self.num_variants / a1
        if span_normalize:
            span = cp.float64(self.chrom_end - self.chrom_start)
            return float(theta / span)
        return float(theta)
    
    def Tajimas_D(self) -> float:
        """
        Calculate Tajima's D for the haplotype matrix.
        """
        # get pi
        pi = self.diversity(span_normalize=False) 
        
        # get theta       
        n_haplotypes = self.num_haplotypes
        S = self.num_variants
        a1 = cp.sum(1.0 / cp.arange(1, n_haplotypes), dtype=cp.float64)  # Harmonic sum
        theta = S / a1
        
        # Variance term for Tajima's D
        a2 = cp.sum(cp.power(cp.arange(1, n_haplotypes, dtype=cp.float64), 2))
        b1 = (n_haplotypes + 1) / (3 * (n_haplotypes - 1))
        b2 = 2 * (n_haplotypes**2 + n_haplotypes + 3) / (9 * n_haplotypes * (n_haplotypes - 1))
        c1 = b1 - (1 / a1)
        c2 = b2 - ((n_haplotypes + 2) / (a1 * n_haplotypes)) + (a2 / (a1 ** 2))
        e1 = c1 / a1
        e2 = c2 / ((a1 ** 2) + a2)
        V = cp.sqrt((e1 * S) + (e2 * S * (S - 1)))
        return float((pi - theta) / V) if V != 0 else float("nan")


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


    def tally_gpu_haplotypes(self, missing: bool = False) -> cp.ndarray:
        """
        GPU version of tallying haplotype counts between all pairs of variants.
        
        For each pair (i, j) of variants (with i < j), it computes:
            n11: the number of haplotypes with allele 1 at both i and j.
            n10: the number with allele 1 at i and allele 0 at j.
            n01: the number with allele 0 at i and allele 1 at j.
            n00: the number with allele 0 at both i and j.
        
        Returns:
            cp.ndarray: A matrix with shape (#pairs, 4) where each row is (n11, n10, n01, n00).
        
        Note:
            This implementation does not yet support missing data.
        """
        import cupy as cp
        
        if missing:
            raise NotImplementedError("Missing data support is not implemented in GPU tallying.")
        
        # Ensure data is on the GPU.
        if self.device == 'CPU':
            self.transfer_to_gpu()
        
        # X is the haplotype matrix of shape (n_haplotypes, num_variants)
        X = self.haplotypes
        n = self.num_haplotypes
        m = self.num_variants
        
        # Count of ones (allele=1) per variant – shape: (m,)
        ones_per_variant = cp.sum(X, axis=0)
        
        # Compute n11 for all variant pairs using matrix multiplication.
        # The (i,j) element of this matrix is the number of haplotypes with 1 at both variants.
        n11_mat = X.T @ X  # shape (m, m)
        
        # Get the indices for the upper triangle (excluding the diagonal)
        idx_i, idx_j = cp.triu_indices(m, k=1)
        
        n11_pairs = n11_mat[idx_i, idx_j]
        n10_pairs = ones_per_variant[idx_i] - n11_pairs
        n01_pairs = ones_per_variant[idx_j] - n11_pairs
        n00_pairs = n - (n11_pairs + n10_pairs + n01_pairs)
        
        # Stack the results into one matrix of shape (#pairs, 4)
        counts = cp.stack([n11_pairs, n10_pairs, n01_pairs, n00_pairs], axis=1)
        return counts

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

    def compute_ld_statistics_gpu(self, bp_bins, missing=False, raw=False):
        """
        GPU-based implementation of computing LD statistics for a single population using tallies
        from tally_gpu_haplotypes, followed by binning by base-pair distance.
        
        Steps:
          1. Compute pairwise haplotype tallies using tally_gpu_haplotypes. Each row in the resulting 
             CuPy array represents the 2x2 haplotype count table [n11, n10, n01, n00] for a variant pair.
          2. Compute the positions for each variant pair from the upper-triangle indices (using cp.triu_indices)
             so that distance = pos[j] - pos[i].
          3. Using the GPU stats module (stats_from_haplotype_counts_gpu), compute for each pair: D, D^2, Dz, and π₂.
          4. Bin the variant pairs by distance using the provided bp_bins.
          5. Depending on the `raw` flag:
               * If raw is False (default), return the mean (averaged over all pairs in the bin) of each statistic.
               * If raw is True, return the raw sums for each statistic (which should match the Moments aggregation).
        
        Parameters:
            bp_bins (array-like): Array of bin boundaries in base pairs (e.g. [0, 50, 100, ...]).
            missing (bool): If True, raises NotImplementedError (missing data not implemented).
            raw (bool): If True, return the raw sums aggregated in each bin, rather than mean values.
        
        Returns:
            dict: A dictionary mapping each bin (tuple: (bin_start, bin_end)) to a tuple:
                  (D2, Dz, pi2, D). If raw is False these values are averaged over pairs; if raw is True
                  they are the raw sums.
        """
        if missing:
            raise NotImplementedError("Missing data support is not implemented.")

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
        # "counts" is a CuPy array of shape (#pairs, 4) with columns: [n11, n10, n01, n00].
        counts = self.tally_gpu_haplotypes(missing=missing)

        # Get the variant-pair indices corresponding to the tallies.
        idx_i, idx_j = cp.triu_indices(m, k=1)
        # Compute the physical distance between variant pairs.
        distances = pos[idx_j] - pos[idx_i]  # shape: (#pairs,)

        # Import the GPU stats module.
        from pg_gpu import stats_from_haplotype_counts_gpu as stats

        # Compute the LD statistics for all variant pairs.
        # D_vals   = stats.D(counts)    # shape: (#pairs,)
        DD_vals  = stats.DD(counts)   # shape: (#pairs,)
        Dz_vals  = stats.Dz(counts)   # shape: (#pairs,)
        pi2_vals = stats.pi2(counts)  # shape: (#pairs,)

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
