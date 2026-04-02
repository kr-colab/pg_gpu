import cupy as cp
import numpy as np
import allel
import tskit
from collections import Counter, OrderedDict


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
                 genotypes,
                 positions,
                 chrom_start: int = None,
                 chrom_end: int = None,
                 sample_sets: dict = None,
                 n_total_sites: int = None,
                 samples: list = None,
                ):
        if genotypes.size == 0:
            raise ValueError("genotypes cannot be empty")
        if positions.size == 0:
            raise ValueError("positions cannot be empty")
        if not isinstance(genotypes, (np.ndarray, cp.ndarray)):
            raise ValueError("genotypes must be a numpy or cupy array")
        if not isinstance(positions, (np.ndarray, cp.ndarray)):
            raise ValueError("positions must be a numpy or cupy array")

        if isinstance(genotypes, cp.ndarray):
            self._device = 'GPU'
            if isinstance(positions, np.ndarray):
                positions = cp.array(positions)
        else:
            self._device = 'CPU'
            if isinstance(positions, cp.ndarray):
                positions = positions.get()

        self.haplotypes = genotypes
        self.positions = positions
        self.chrom_start = chrom_start
        self.chrom_end = chrom_end
        self._sample_sets = sample_sets
        self.n_total_sites = n_total_sites
        self.samples = samples  # diploid sample names from VCF

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

    @property
    def has_invariant_info(self):
        """Whether invariant site information is available for pairwise mode."""
        return self.n_total_sites is not None

    @property
    def n_invariant_sites(self):
        """Number of invariant (monomorphic) sites, or None if unknown.

        If n_total_sites is set, computes as n_total_sites minus the number of
        segregating sites in the matrix. If invariant sites are stored directly
        in the matrix, counts sites where all valid alleles are identical.
        """
        if self.n_total_sites is None:
            return None
        xp = cp if self.device == 'GPU' else np
        hap = self.haplotypes
        valid_mask = hap >= 0
        hap_clean = xp.where(valid_mask, hap, 0)
        derived_counts = xp.sum(hap_clean, axis=0)
        n_valid = xp.sum(valid_mask, axis=0)
        # A site is variant if 0 < derived < n_valid (i.e. polymorphic among observed)
        is_variant = (derived_counts > 0) & (derived_counts < n_valid) & (n_valid >= 2)
        n_variant = int(xp.sum(is_variant))
        return self.n_total_sites - n_variant

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
    def from_vcf(cls, path: str, region: str = None,
                 samples: list = None, include_invariant: bool = False):
        """Construct a HaplotypeMatrix from a VCF file.

        Parameters
        ----------
        path : str
            Path to VCF/BCF file (optionally gzipped + tabix-indexed).
        region : str, optional
            Genomic region to load, e.g. 'chr1:1000000-2000000'.
            Requires the VCF to be bgzipped and tabix-indexed.
        samples : list of str, optional
            Subset of samples to load. If None, loads all samples.
        include_invariant : bool
            If True, set n_total_sites from the loaded variant count.

        Returns
        -------
        HaplotypeMatrix
            Phased haplotype data with sample names stored.
        """
        vcf = allel.read_vcf(path, region=region, samples=samples)
        if vcf is None:
            raise ValueError(f"No variants found in {path}"
                             + (f" for region {region}" if region else ""))

        genotypes = allel.GenotypeArray(vcf['calldata/GT'])
        num_variants, num_samples, ploidy = genotypes.shape
        assert ploidy == 2

        haplotypes = np.empty((num_variants, 2 * num_samples), dtype=genotypes.dtype)
        haplotypes[:, :num_samples] = genotypes[:, :, 0]
        haplotypes[:, num_samples:] = genotypes[:, :, 1]
        haplotypes = haplotypes.T

        positions = np.array(vcf['variants/POS'])
        sample_names = list(vcf['samples'])

        n_total_sites = num_variants if include_invariant else None
        return cls(haplotypes, positions, positions[0], positions[-1],
                   n_total_sites=n_total_sites, samples=sample_names)

    @classmethod
    def from_zarr(cls, path: str, region: str = None):
        """Construct a HaplotypeMatrix from a Zarr store.

        Zarr provides fast columnar access, especially for large datasets.
        Create a Zarr store from VCF using allel.vcf_to_zarr() or
        HaplotypeMatrix.vcf_to_zarr().

        Parameters
        ----------
        path : str
            Path to Zarr store directory.
        region : str, optional
            Genomic region 'chrom:start-end' to load a subset.

        Returns
        -------
        HaplotypeMatrix
        """
        import zarr
        store = zarr.open(path, mode='r')

        positions = np.array(store['variants/POS'])
        gt = np.array(store['calldata/GT'])
        sample_names = list(store['samples']) if 'samples' in store else None

        # Apply region filter if specified
        if region is not None:
            chrom_str, coords = region.split(':')
            start, end = [int(x) for x in coords.split('-')]
            mask = (positions >= start) & (positions < end)
            positions = positions[mask]
            gt = gt[mask]
            if len(positions) == 0:
                raise ValueError(f"No variants in region {region}")

        num_variants, num_samples, ploidy = gt.shape
        assert ploidy == 2

        haplotypes = np.empty((num_variants, 2 * num_samples), dtype=gt.dtype)
        haplotypes[:, :num_samples] = gt[:, :, 0]
        haplotypes[:, num_samples:] = gt[:, :, 1]
        haplotypes = haplotypes.T

        return cls(haplotypes, positions, positions[0], positions[-1],
                   samples=sample_names)

    def to_zarr(self, zarr_path: str):
        """Save haplotype data to Zarr format for fast reloading.

        Parameters
        ----------
        zarr_path : str
            Output Zarr store path.
        """
        import zarr
        store = zarr.open(zarr_path, mode='w')
        hap = self.haplotypes if isinstance(self.haplotypes, np.ndarray) else self.haplotypes.get()
        pos = self.positions if isinstance(self.positions, np.ndarray) else self.positions.get()

        gt = self._haplotypes_to_gt(hap)
        store.create_dataset('calldata/GT', shape=gt.shape, dtype=gt.dtype, data=gt)
        store.create_dataset('variants/POS', shape=pos.shape, dtype=pos.dtype, data=pos)
        if self.samples is not None:
            s = np.array(self.samples, dtype='U')
            store.create_dataset('samples', shape=s.shape, dtype=s.dtype, data=s)

    @staticmethod
    def _haplotypes_to_gt(hap):
        """Convert (n_hap, n_var) haplotype matrix back to (n_var, n_samples, 2) GT array."""
        n_hap, n_var = hap.shape
        n_samples = n_hap // 2
        gt = np.empty((n_var, n_samples, 2), dtype=hap.dtype)
        gt[:, :, 0] = hap[:n_samples, :].T
        gt[:, :, 1] = hap[n_samples:, :].T
        return gt

    def load_pop_file(self, pop_file: str, pops: list = None):
        """Load population assignments from a tab-delimited file.

        Sets sample_sets from a file mapping sample names to populations.
        Requires that sample names were stored during from_vcf().

        Parameters
        ----------
        pop_file : str
            Tab-delimited file with columns: sample, pop.
            Header line starting with 'sample' is skipped.
        pops : list of str, optional
            Populations to include. If None, includes all found populations.
        """
        if self.samples is None:
            raise ValueError("No sample names stored. Use from_vcf() to load data.")

        n_samples = len(self.samples)
        pop_map = {}
        with open(pop_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] != 'sample':
                    pop_map[parts[0]] = parts[1]

        # Build sample_sets
        found_pops = set(pop_map.values())
        if pops is None:
            pops = sorted(found_pops)

        pop_sets = {p: [] for p in pops}
        for i, name in enumerate(self.samples):
            pop = pop_map.get(name)
            if pop in pop_sets:
                pop_sets[pop].append(i)
                pop_sets[pop].append(i + n_samples)

        self.sample_sets = pop_sets

    @classmethod
    def from_ts(cls, ts: tskit.TreeSequence, device: str = 'CPU',
                include_invariant: bool = False) -> 'HaplotypeMatrix':
        """
        Create a HaplotypeMatrix from a tskit.TreeSequence.

        Args:
            ts: A tskit.TreeSequence object
            device: 'CPU' or 'GPU'
            include_invariant: If True, set n_total_sites from the sequence
                length so that pairwise-mode calculations can account for
                invariant sites analytically (no extra rows stored).

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

        n_total_sites = int(ts.sequence_length) if include_invariant else None
        return cls(haplotypes, positions, chrom_start, chrom_end,
                   n_total_sites=n_total_sites)

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
            result.n_total_sites = None
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

        hap = self.haplotypes
        valid_mask = (hap >= 0).astype(cp.float64)
        hap_clean = cp.where(hap >= 0, hap, 0).astype(cp.float64)
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)  # per-site

        # Allele frequencies from valid data only
        p = cp.where(n_valid > 0, cp.sum(hap_clean, axis=0) / n_valid, 0.0)

        # p_AB: joint frequency of derived at both sites
        # Only count haplotypes valid at both sites
        joint_n = valid_mask.T @ valid_mask  # (n_var, n_var)
        joint_11 = hap_clean.T @ hap_clean
        p_AB = cp.where(joint_n > 0, joint_11 / joint_n, 0.0)

        p_Ap_B = cp.outer(p, p)
        D = p_AB - p_Ap_B
        cp.fill_diagonal(D, 0)

        return D

    def pairwise_r2(self) -> np.ndarray:
        """
        Calculate the pairwise r2 (correlation coefficient) for all pairs of variants
        in the haplotype matrix.

        Returns
        -------
        ndarray, float64, shape (n_variants, n_variants)
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()

        hap = self.haplotypes
        valid_mask = (hap >= 0).astype(cp.float64)
        hap_clean = cp.where(hap >= 0, hap, 0).astype(cp.float64)
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)

        p = cp.where(n_valid > 0, cp.sum(hap_clean, axis=0) / n_valid, 0.0)

        joint_n = valid_mask.T @ valid_mask
        joint_11 = hap_clean.T @ hap_clean
        p_AB = cp.where(joint_n > 0, joint_11 / joint_n, 0.0)

        p_Ap_B = cp.outer(p, p)
        D = p_AB - p_Ap_B

        denom_squared = cp.outer(p * (1 - p), p * (1 - p))
        r2 = cp.where(denom_squared > 0, (D ** 2) / denom_squared, 0)

        cp.fill_diagonal(r2, 0)

        return r2.get()

    def locate_unlinked(self, size=100, step=20, threshold=0.1):
        """Locate variants in approximate linkage equilibrium.

        Uses a sliding window approach to identify variants whose r-squared
        with all other variants in the window is below the threshold.

        Parameters
        ----------
        size : int
            Window size (number of variants).
        step : int
            Number of variants to advance between windows.
        threshold : float
            Maximum r-squared value to consider variants unlinked.

        Returns
        -------
        ndarray, bool, shape (n_variants,)
            True for variants in approximate linkage equilibrium.
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()

        m = self.num_variants
        n_hap = self.num_haplotypes

        # allele frequencies from valid data
        hap = self.haplotypes
        valid_mask = (hap >= 0).astype(cp.float64)
        hap_clean = cp.where(hap >= 0, hap, 0).astype(cp.float64)
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        p = cp.where(n_valid > 0, cp.sum(hap_clean, axis=0) / n_valid, 0.0)

        # pruning state kept on CPU to avoid per-scalar GPU transfers
        loc = np.ones(m, dtype=bool)

        for w_start in range(0, m, step):
            w_end = min(w_start + size, m)

            active = loc[w_start:w_end]
            if np.sum(active) <= 1:
                continue

            active_idx = np.where(active)[0] + w_start
            active_idx_gpu = cp.asarray(active_idx)
            hw = hap_clean[:, active_idx_gpu]
            vm = valid_mask[:, active_idx_gpu]
            p_window = p[active_idx_gpu]

            # pairwise r² within window via matrix multiply
            joint_n = vm.T @ vm
            joint_11 = hw.T @ hw
            p_AB = cp.where(joint_n > 0, joint_11 / joint_n, 0.0)
            p_Ap_B = cp.outer(p_window, p_window)
            D = p_AB - p_Ap_B
            denom = cp.outer(p_window * (1 - p_window),
                             p_window * (1 - p_window))
            r2_mat = cp.where(denom > 0, (D ** 2) / denom, 0.0)
            cp.fill_diagonal(r2_mat, 0.0)

            # prune on CPU to avoid per-scalar GPU transfers
            r2_mat_cpu = r2_mat.get()
            n_active = len(active_idx)
            for i in range(n_active):
                if not loc[active_idx[i]]:
                    continue
                for j in range(i + 1, n_active):
                    if not loc[active_idx[j]]:
                        continue
                    if r2_mat_cpu[i, j] > threshold:
                        loc[active_idx[j]] = False

        return loc

    def windowed_r_squared(self, bp_bins, percentile=50, pop=None):
        """Compute percentiles of r-squared in genomic distance bins.

        Parameters
        ----------
        bp_bins : array_like
            Bin edges for genomic distances in base pairs.
        percentile : float or array_like
            Percentile(s) to compute within each bin.
        pop : str, optional
            Population key to use.

        Returns
        -------
        result : ndarray, shape (n_bins,) or (n_bins, n_percentiles)
            Percentile(s) of r-squared per bin.
        counts : ndarray, int, shape (n_bins,)
            Number of variant pairs per bin.
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()

        pos = self.positions
        if not isinstance(pos, cp.ndarray):
            pos = cp.array(pos)

        m = self.num_variants

        # compute counts and r² via tally
        counts_arr, n_valid = self.tally_gpu_haplotypes(pop=pop)
        from pg_gpu import ld_statistics
        r2_vals = ld_statistics.r_squared(counts_arr, n_valid=n_valid)

        # pair distances
        idx_i, idx_j = cp.triu_indices(m, k=1)
        distances = pos[idx_j] - pos[idx_i]

        bp_bins_cp = cp.array(bp_bins)
        n_bins = len(bp_bins) - 1
        bin_inds = cp.digitize(distances, bp_bins_cp) - 1

        valid_mask = (bin_inds >= 0) & (bin_inds < n_bins) & ~cp.isnan(r2_vals)

        # transfer to CPU for percentile computation
        r2_cpu = r2_vals[valid_mask].get()
        bins_cpu = bin_inds[valid_mask].get()

        percentile = np.atleast_1d(percentile)
        result = np.full((n_bins, len(percentile)), np.nan)
        pair_counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = bins_cpu == i
            pair_counts[i] = int(np.sum(mask))
            if pair_counts[i] > 0:
                for p_idx, pct in enumerate(percentile):
                    result[i, p_idx] = np.percentile(r2_cpu[mask], pct)

        if result.shape[1] == 1:
            result = result[:, 0]

        return result, pair_counts


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

    def compute_ld_statistics_gpu_single_pop(
        self,
        bp_bins,
        raw=False,
        ac_filter=True,
        chunk_size='auto'
    ):
        """
        GPU-based LD statistics computation for a single population.

        Computes DD, Dz, and pi2 statistics for variant pairs binned by distance.
        Only processes pairs within max(bp_bins) distance for memory efficiency.

        Parameters
        ----------
        bp_bins : array-like
            Array of bin boundaries in base pairs. Pairs are binned by distance
            into intervals [bp_bins[i], bp_bins[i+1]).
        raw : bool, optional
            If True, return raw sums of statistics across pairs in each bin.
            If False (default), return means.
        ac_filter : bool, optional
            If True (default), apply biallelic filtering before computation.
        chunk_size : int or 'auto', optional
            Number of pairs to process per chunk. If 'auto' (default),
            automatically estimates optimal size based on available GPU memory.
            Can specify an integer for manual control.

        Returns
        -------
        dict
            Dictionary mapping (bin_start, bin_end) tuples to tuples of statistics.
            Each bin contains (DD, Dz, pi2) values.

        Examples
        --------
        >>> bp_bins = [0, 10000, 50000, 100000]
        >>> stats = hm.compute_ld_statistics_gpu_single_pop(bp_bins)
        >>> stats[(0.0, 10000.0)]  # (DD, Dz, pi2) for first bin
        """
        # Apply biallelic filter if requested
        if ac_filter:
            filtered_self = self.apply_biallelic_filter()
            return filtered_self.compute_ld_statistics_gpu_single_pop(
                bp_bins=bp_bins, raw=raw, ac_filter=False, chunk_size=chunk_size
            )

        # Ensure GPU setup
        if self.device == 'CPU':
            self.transfer_to_gpu()

        from pg_gpu import ld_statistics

        # Get positions and compute max distance
        pos = self.positions
        if not isinstance(pos, cp.ndarray):
            pos = cp.array(pos)

        bp_bins = np.array(bp_bins)
        max_dist = float(bp_bins[-1])
        n_bins = len(bp_bins) - 1

        # Handle chunk_size='auto'
        if chunk_size == 'auto':
            n_haps = self.num_haplotypes
            chunk_size = _estimate_ld_chunk_size(n_haps)

        # Generate distance-filtered pair indices
        idx_i, idx_j = _generate_pairs_within_distance(pos, max_dist)
        total_pairs = len(idx_i)

        if total_pairs == 0:
            # No pairs within distance - return zeros for all bins
            out = {}
            for i in range(n_bins):
                out[(float(bp_bins[i]), float(bp_bins[i + 1]))] = (0.0, 0.0, 0.0)
            return out

        # Compute distances for all pairs
        distances = pos[idx_j] - pos[idx_i]

        # Bin assignment
        bp_bins_cp = cp.array(bp_bins)
        bin_inds = cp.digitize(distances, bp_bins_cp) - 1

        # Initialize accumulators: sums and counts per bin
        # 3 statistics: DD, Dz, pi2
        bin_sums = cp.zeros((n_bins, 3), dtype=cp.float64)
        bin_counts = cp.zeros(n_bins, dtype=cp.float64)

        # Process in chunks
        for chunk_start in range(0, total_pairs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pairs)

            # Get chunk indices
            chunk_idx_i = idx_i[chunk_start:chunk_end]
            chunk_idx_j = idx_j[chunk_start:chunk_end]
            chunk_bin_inds = bin_inds[chunk_start:chunk_end]

            # Compute haplotype counts for this chunk (no population filter)
            counts, n_valid = _compute_counts_for_pairs(
                self.haplotypes, chunk_idx_i, chunk_idx_j, pop_indices=None
            )

            # Compute all 3 statistics for this chunk
            chunk_stats = _compute_single_pop_statistics_batch(
                counts, n_valid, ld_statistics
            )

            # Accumulate into bins using scatter_add
            valid_mask = (chunk_bin_inds >= 0) & (chunk_bin_inds < n_bins)
            valid_bin_inds = chunk_bin_inds[valid_mask]
            valid_stats = chunk_stats[valid_mask]

            # Accumulate sums
            for stat_idx in range(3):
                cp.add.at(bin_sums[:, stat_idx], valid_bin_inds, valid_stats[:, stat_idx])

            # Accumulate counts
            cp.add.at(bin_counts, valid_bin_inds, cp.ones(len(valid_bin_inds), dtype=cp.float64))

            # Free chunk memory
            del counts, n_valid, chunk_stats
            del chunk_idx_i, chunk_idx_j, valid_stats

        # Build output dictionary
        out = {}
        for i in range(n_bins):
            bin_start = float(bp_bins[i])
            bin_end = float(bp_bins[i + 1])
            count = int(bin_counts[i].get())

            if raw:
                out[(bin_start, bin_end)] = (
                    float(bin_sums[i, 0].get()),
                    float(bin_sums[i, 1].get()),
                    float(bin_sums[i, 2].get())
                )
            else:
                if count > 0:
                    out[(bin_start, bin_end)] = (
                        float((bin_sums[i, 0] / bin_counts[i]).get()),
                        float((bin_sums[i, 1] / bin_counts[i]).get()),
                        float((bin_sums[i, 2] / bin_counts[i]).get())
                    )
                else:
                    out[(bin_start, bin_end)] = (0.0, 0.0, 0.0)

        return out

    def _compute_ld_statistics_gpu_single_pop_all_pairs(self, bp_bins, raw=False, ac_filter=True):
        """
        DEPRECATED: Original all-pairs implementation. Use compute_ld_statistics_gpu_single_pop instead.

        GPU-based implementation of computing LD statistics for a single population using tallies
        from tally_gpu_haplotypes, followed by binning by base-pair distance.

        This implementation computes ALL pairs, which can cause OOM errors for large datasets.
        The new compute_ld_statistics_gpu_single_pop uses distance-limited chunked processing.

        Parameters:
            bp_bins (array-like): Array of bin boundaries in base pairs (e.g. [0, 50, 100, ...]).
            raw (bool): If True, return the raw sums aggregated in each bin, rather than mean values.
            ac_filter (bool): If True, apply biallelic filtering (matching moments' is_biallelic_01 behavior).

        Returns:
            dict: A dictionary mapping each bin (tuple: (bin_start, bin_end)) to a tuple:
                  (D2, Dz, pi2). If raw is False these values are averaged over pairs; if raw is True
                  they are the raw sums.
        """
        # Apply biallelic filter if requested (matches moments' default behavior)
        if ac_filter:
            # Apply biallelic filtering to match moments' is_biallelic_01() behavior
            filtered_self = self.apply_biallelic_filter()
            # Use the filtered matrix for computation
            return filtered_self._compute_ld_statistics_gpu_single_pop_all_pairs(
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

    def _compute_ld_statistics_gpu_two_pops_all_pairs(self, bp_bins, pop1: str, pop2: str, raw=False, ac_filter=True):
        """DEPRECATED: Legacy implementation that computes ALL variant pairs.

        Warning: This method has O(n^2) memory complexity and will OOM on large datasets.
        Use compute_ld_statistics_gpu_two_pops() instead, which filters by max distance.

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
            return filtered_self._compute_ld_statistics_gpu_two_pops_all_pairs(
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

    def compute_ld_statistics_gpu_two_pops(
        self,
        bp_bins,
        pop1: str,
        pop2: str,
        raw=False,
        ac_filter=True,
        chunk_size='auto'
    ):
        """
        GPU-based LD statistics computation for two populations.

        Computes DD, Dz, and pi2 statistics for variant pairs binned by distance.
        Only processes pairs within max(bp_bins) distance for memory efficiency.

        Parameters
        ----------
        bp_bins : array-like
            Array of bin boundaries in base pairs. Pairs are binned by distance
            into intervals [bp_bins[i], bp_bins[i+1]).
        pop1 : str
            Name of first population (must exist in sample_sets)
        pop2 : str
            Name of second population (must exist in sample_sets)
        raw : bool, optional
            If True, return raw sums of statistics across pairs in each bin.
            If False (default), return means.
        ac_filter : bool, optional
            If True (default), apply biallelic filtering before computation.
        chunk_size : int or 'auto', optional
            Number of pairs to process per chunk. If 'auto' (default),
            automatically estimates optimal size based on available GPU memory.
            Can specify an integer for manual control.

        Returns
        -------
        dict
            Dictionary mapping (bin_start, bin_end) tuples to OrderedDict of statistics.
            Each bin contains 15 statistics:
            - DD_0_0, DD_0_1, DD_1_1 (D squared)
            - Dz_0_0_0, Dz_0_0_1, Dz_0_1_1, Dz_1_0_0, Dz_1_0_1, Dz_1_1_1
            - pi2_0_0_0_0, pi2_0_0_0_1, pi2_0_0_1_1, pi2_0_1_0_1, pi2_0_1_1_1, pi2_1_1_1_1

        Examples
        --------
        >>> bp_bins = [0, 10000, 50000, 100000]
        >>> stats = hm.compute_ld_statistics_gpu_two_pops(bp_bins, 'pop1', 'pop2')
        >>> stats[(0.0, 10000.0)]['DD_0_0']  # D^2 for pop1 in first bin
        """
        # Apply biallelic filter if requested
        if ac_filter:
            filtered_self = self.apply_biallelic_filter()
            return filtered_self.compute_ld_statistics_gpu_two_pops(
                bp_bins=bp_bins, pop1=pop1, pop2=pop2, raw=raw,
                ac_filter=False, chunk_size=chunk_size
            )

        # Ensure GPU setup
        if self.device == 'CPU':
            self.transfer_to_gpu()

        from pg_gpu import ld_statistics

        # Get positions and compute max distance
        pos = self.positions
        if not isinstance(pos, cp.ndarray):
            pos = cp.array(pos)

        bp_bins = np.array(bp_bins)
        max_dist = float(bp_bins[-1])
        n_bins = len(bp_bins) - 1

        # Get population indices
        pop1_indices = self._sample_sets[pop1]
        pop2_indices = self._sample_sets[pop2]

        # Handle chunk_size='auto'
        if chunk_size == 'auto':
            n_haps = max(len(pop1_indices), len(pop2_indices))
            chunk_size = _estimate_ld_chunk_size(n_haps)

        # Generate distance-filtered pair indices
        idx_i, idx_j = _generate_pairs_within_distance(pos, max_dist)
        total_pairs = len(idx_i)

        if total_pairs == 0:
            # No pairs within distance - return zeros for all bins
            out = {}
            for i in range(n_bins):
                out[(float(bp_bins[i]), float(bp_bins[i + 1]))] = OrderedDict([
                    ('DD_0_0', 0.0), ('DD_0_1', 0.0), ('DD_1_1', 0.0),
                    ('Dz_0_0_0', 0.0), ('Dz_0_0_1', 0.0), ('Dz_0_1_1', 0.0),
                    ('Dz_1_0_0', 0.0), ('Dz_1_0_1', 0.0), ('Dz_1_1_1', 0.0),
                    ('pi2_0_0_0_0', 0.0), ('pi2_0_0_0_1', 0.0), ('pi2_0_0_1_1', 0.0),
                    ('pi2_0_1_0_1', 0.0), ('pi2_0_1_1_1', 0.0), ('pi2_1_1_1_1', 0.0)
                ])
            return out

        # Compute distances for all pairs
        distances = pos[idx_j] - pos[idx_i]

        # Bin assignment
        bp_bins_cp = cp.array(bp_bins)
        bin_inds = cp.digitize(distances, bp_bins_cp) - 1

        # Initialize accumulators: sums and counts per bin
        stat_names = [
            'DD_0_0', 'DD_0_1', 'DD_1_1',
            'Dz_0_0_0', 'Dz_0_0_1', 'Dz_0_1_1', 'Dz_1_0_0', 'Dz_1_0_1', 'Dz_1_1_1',
            'pi2_0_0_0_0', 'pi2_0_0_0_1', 'pi2_0_0_1_1', 'pi2_0_1_0_1', 'pi2_0_1_1_1', 'pi2_1_1_1_1'
        ]
        bin_sums = cp.zeros((n_bins, 15), dtype=cp.float64)
        bin_counts = cp.zeros(n_bins, dtype=cp.float64)  # float64 for scatter_add compatibility

        # Process in chunks
        for chunk_start in range(0, total_pairs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pairs)

            # Get chunk indices
            chunk_idx_i = idx_i[chunk_start:chunk_end]
            chunk_idx_j = idx_j[chunk_start:chunk_end]
            chunk_bin_inds = bin_inds[chunk_start:chunk_end]

            # Compute haplotype counts for this chunk
            counts_pop1, n_valid1 = _compute_counts_for_pairs(
                self.haplotypes, chunk_idx_i, chunk_idx_j, pop1_indices
            )
            counts_pop2, n_valid2 = _compute_counts_for_pairs(
                self.haplotypes, chunk_idx_i, chunk_idx_j, pop2_indices
            )

            # Compute all 15 statistics for this chunk
            chunk_stats = _compute_two_pop_statistics_batch(
                counts_pop1, counts_pop2, n_valid1, n_valid2, ld_statistics
            )

            # Accumulate into bins using scatter_add
            valid_mask = (chunk_bin_inds >= 0) & (chunk_bin_inds < n_bins)
            valid_bin_inds = chunk_bin_inds[valid_mask]
            valid_stats = chunk_stats[valid_mask]

            # Accumulate sums per bin
            for stat_idx in range(15):
                cp.add.at(bin_sums[:, stat_idx], valid_bin_inds, valid_stats[:, stat_idx])

            # Accumulate counts
            cp.add.at(bin_counts, valid_bin_inds, cp.ones(len(valid_bin_inds), dtype=cp.float64))

            # Free chunk memory
            del counts_pop1, counts_pop2, n_valid1, n_valid2, chunk_stats
            del chunk_idx_i, chunk_idx_j, valid_stats

        # Build output dictionary
        out = {}
        for i in range(n_bins):
            bin_start = float(bp_bins[i])
            bin_end = float(bp_bins[i + 1])
            count = int(bin_counts[i].get())

            if raw:
                stats_dict = OrderedDict([
                    (name, float(bin_sums[i, j].get()))
                    for j, name in enumerate(stat_names)
                ])
            else:
                if count > 0:
                    stats_dict = OrderedDict([
                        (name, float((bin_sums[i, j] / bin_counts[i]).get()))
                        for j, name in enumerate(stat_names)
                    ])
                else:
                    stats_dict = OrderedDict([
                        (name, 0.0) for name in stat_names
                    ])

            out[(bin_start, bin_end)] = stats_dict

        return out


# =============================================================================
# Helper functions for memory-efficient LD computation with max distance filter
# =============================================================================

def _estimate_ld_chunk_size(n_haplotypes_per_pop, available_memory_bytes=None,
                            num_pops=2):
    """
    Estimate optimal chunk size for LD computation based on GPU memory.

    Memory per pair (N pairs, H haplotypes per pop, P populations):
    - hap_i, hap_j arrays (P pops): 4 * H * P * N bytes
    - counts arrays: 32 * P * N bytes
    - statistics: 120 * P * N bytes
    - Overhead (~3x): accounts for intermediates, fragmentation

    Formula: bytes_per_pair ≈ (4*H*P + 150*P) * 3

    Parameters
    ----------
    n_haplotypes_per_pop : int
        Number of haplotypes in the larger population
    available_memory_bytes : int, optional
        Available GPU memory in bytes. If None, queries the GPU.
    num_pops : int
        Number of populations (default 2)

    Returns
    -------
    int
        Recommended chunk size (number of pairs per iteration)
    """
    if available_memory_bytes is None:
        # Use 50% of available GPU memory (conservative)
        available_memory_bytes = int(cp.cuda.Device().mem_info[0] * 0.5)

    bytes_per_pair = (4 * n_haplotypes_per_pop * num_pops + 150 * num_pops) * 3
    chunk_size = available_memory_bytes // bytes_per_pair

    # Bounds: min 100k, max 10M
    chunk_size = max(100_000, min(chunk_size, 10_000_000))

    return chunk_size


def _generate_pairs_within_distance(positions, max_dist):
    """
    Generate (i, j) pair indices for all variant pairs where pos[j] - pos[i] <= max_dist.

    Uses binary search for O(n log n) complexity instead of O(n^2).
    Assumes positions are sorted.

    Parameters
    ----------
    positions : cp.ndarray
        Sorted variant positions on GPU
    max_dist : float
        Maximum distance between variants to include

    Returns
    -------
    idx_i : cp.ndarray[int32]
        First index of each pair
    idx_j : cp.ndarray[int32]
        Second index of each pair
    """
    n = len(positions)

    # Binary search: for each i, find first j where pos[j] > pos[i] + max_dist
    upper_bounds = positions + max_dist
    j_max = cp.searchsorted(positions, upper_bounds, side='right')

    # Count pairs per variant: valid j's are in range [i+1, j_max)
    variant_indices = cp.arange(n, dtype=cp.int64)
    pairs_per_variant = cp.maximum(0, j_max - variant_indices - 1)
    total_pairs = int(cp.sum(pairs_per_variant).get())

    if total_pairs == 0:
        return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)

    # Generate indices on CPU (more memory efficient for large arrays with repeat)
    pairs_per_variant_cpu = pairs_per_variant.get().astype(np.int64)
    j_max_cpu = j_max.get()

    # Pre-allocate arrays
    idx_i_cpu = np.empty(total_pairs, dtype=np.int32)
    idx_j_cpu = np.empty(total_pairs, dtype=np.int32)

    # Fill arrays
    offset = 0
    for i in range(n):
        n_pairs_i = int(pairs_per_variant_cpu[i])
        if n_pairs_i > 0:
            idx_i_cpu[offset:offset + n_pairs_i] = i
            idx_j_cpu[offset:offset + n_pairs_i] = np.arange(i + 1, j_max_cpu[i], dtype=np.int32)
            offset += n_pairs_i

    return cp.array(idx_i_cpu), cp.array(idx_j_cpu)


def _compute_counts_for_pairs(haplotypes, idx_i, idx_j, pop_indices=None):
    """
    Compute haplotype counts [n11, n10, n01, n00] for specific pairs.

    Parameters
    ----------
    haplotypes : cp.ndarray
        Shape (n_haplotypes, n_variants), values 0, 1, or -1 (missing)
    idx_i, idx_j : cp.ndarray
        Pair indices, shape (n_pairs,)
    pop_indices : list or cp.ndarray, optional
        Indices of samples to include (for population-specific counts)

    Returns
    -------
    counts : cp.ndarray
        Shape (n_pairs, 4), columns [n11, n10, n01, n00]
    n_valid : cp.ndarray or None
        Shape (n_pairs,), valid sample counts per pair. None if no missing data.
    """
    # Select population if specified
    if pop_indices is not None:
        if isinstance(pop_indices, list):
            pop_indices = cp.array(pop_indices, dtype=cp.int32)
        haplotypes = haplotypes[pop_indices, :]

    n_haps = haplotypes.shape[0]
    n_pairs = len(idx_i)

    # Get haplotype values at pair positions: shape (n_haplotypes, n_pairs)
    hap_i = haplotypes[:, idx_i]
    hap_j = haplotypes[:, idx_j]

    # Check for missing data
    has_missing = cp.any(haplotypes == -1)

    if has_missing:
        # Valid mask: both variants non-missing
        valid_mask = (hap_i >= 0) & (hap_j >= 0)
        n_valid = cp.sum(valid_mask, axis=0, dtype=cp.int32)

        # Count combinations only where valid
        n11 = cp.sum((hap_i == 1) & (hap_j == 1) & valid_mask, axis=0, dtype=cp.int32)
        n10 = cp.sum((hap_i == 1) & (hap_j == 0) & valid_mask, axis=0, dtype=cp.int32)
        n01 = cp.sum((hap_i == 0) & (hap_j == 1) & valid_mask, axis=0, dtype=cp.int32)
        n00 = cp.sum((hap_i == 0) & (hap_j == 0) & valid_mask, axis=0, dtype=cp.int32)
    else:
        n_valid = None
        # Count haplotype combinations
        n11 = cp.sum((hap_i == 1) & (hap_j == 1), axis=0, dtype=cp.int32)
        n10 = cp.sum((hap_i == 1) & (hap_j == 0), axis=0, dtype=cp.int32)
        n01 = cp.sum((hap_i == 0) & (hap_j == 1), axis=0, dtype=cp.int32)
        n00 = cp.sum((hap_i == 0) & (hap_j == 0), axis=0, dtype=cp.int32)

    counts = cp.stack([n11, n10, n01, n00], axis=1)
    return counts, n_valid


def _compute_genotype_counts_for_pairs(genotypes, idx_i, idx_j, pop_indices=None):
    """Compute 9-way genotype counts for variant pairs.

    For each pair (locus_i, locus_j), counts all 3x3 combinations of
    genotype values (0/1/2) at the two loci.

    Parameters
    ----------
    genotypes : cp.ndarray
        Shape (n_individuals, n_variants), values 0, 1, 2, or -1 (missing)
    idx_i, idx_j : cp.ndarray
        Pair indices, shape (n_pairs,)
    pop_indices : list or cp.ndarray, optional
        Indices of individuals to include

    Returns
    -------
    counts : cp.ndarray
        Shape (n_pairs, 9), ordering: (n00, n01, n02, n10, n11, n12, n20, n21, n22)
    n_valid : cp.ndarray or None
        Shape (n_pairs,), valid individual counts per pair. None if no missing data.
    """
    if pop_indices is not None:
        if isinstance(pop_indices, list):
            pop_indices = cp.array(pop_indices, dtype=cp.int32)
        genotypes = genotypes[pop_indices, :]

    n_pairs = len(idx_i)

    geno_i = genotypes[:, idx_i]  # (n_ind, n_pairs)
    geno_j = genotypes[:, idx_j]

    has_missing = cp.any(genotypes == -1)

    if has_missing:
        valid_mask = (geno_i >= 0) & (geno_j >= 0)
        n_valid = cp.sum(valid_mask, axis=0, dtype=cp.int32)
        gi = cp.where(valid_mask, geno_i, 0)
        gj = cp.where(valid_mask, geno_j, 0)
    else:
        n_valid = None
        valid_mask = None
        gi = geno_i
        gj = geno_j

    # Combined index: geno_i * 3 + geno_j gives values 0-8
    combo = gi * 3 + gj  # (n_ind, n_pairs)

    cols = []
    for k in range(9):
        mask = combo == k
        if valid_mask is not None:
            mask = mask & valid_mask
        cols.append(cp.sum(mask, axis=0, dtype=cp.int32))

    counts = cp.stack(cols, axis=1)  # (n_pairs, 9)
    return counts, n_valid


def _compute_two_pop_statistics_batch(counts_pop1, counts_pop2, n_valid1, n_valid2, ld_statistics):
    """
    Compute all 15 two-population LD statistics for a batch of pairs.

    Parameters
    ----------
    counts_pop1 : cp.ndarray
        Shape (n_pairs, 4), haplotype counts for population 1
    counts_pop2 : cp.ndarray
        Shape (n_pairs, 4), haplotype counts for population 2
    n_valid1, n_valid2 : cp.ndarray or None
        Valid sample counts per pair per population
    ld_statistics : module
        The ld_statistics module with dd, dz, pi2 functions

    Returns
    -------
    statistics : cp.ndarray
        Shape (n_pairs, 15), all statistics for each pair
    """
    # Stack for between-population computations
    counts_between = cp.concatenate([counts_pop1, counts_pop2], axis=1)
    n_valid_between = (n_valid1, n_valid2) if n_valid1 is not None else None

    # DD statistics
    DD_0_0 = ld_statistics.dd(counts_pop1, n_valid=n_valid1)
    DD_0_1 = ld_statistics.dd(counts_between, populations=(0, 1), n_valid=n_valid_between)
    DD_1_1 = ld_statistics.dd(counts_pop2, n_valid=n_valid2)

    # Dz statistics
    Dz_0_0_0 = ld_statistics.dz(counts_pop1, n_valid=n_valid1)

    Dz_0_0_1_part1 = ld_statistics.dz(counts_between, populations=(0, 0, 1), n_valid=n_valid_between)
    Dz_0_0_1_part2 = ld_statistics.dz(counts_between, populations=(0, 1, 0), n_valid=n_valid_between)
    Dz_0_0_1 = 0.5 * Dz_0_0_1_part1 + 0.5 * Dz_0_0_1_part2

    Dz_0_1_1 = ld_statistics.dz(counts_between, populations=(0, 1, 1), n_valid=n_valid_between)
    Dz_1_0_0 = ld_statistics.dz(counts_between, populations=(1, 0, 0), n_valid=n_valid_between)

    Dz_1_0_1_part1 = ld_statistics.dz(counts_between, populations=(1, 0, 1), n_valid=n_valid_between)
    Dz_1_0_1_part2 = ld_statistics.dz(counts_between, populations=(1, 1, 0), n_valid=n_valid_between)
    Dz_1_0_1 = 0.5 * Dz_1_0_1_part1 + 0.5 * Dz_1_0_1_part2

    Dz_1_1_1 = ld_statistics.dz(counts_pop2, n_valid=n_valid2)

    # pi2 statistics
    pi2_0_0_0_0 = ld_statistics.pi2(counts_pop1, n_valid=n_valid1)
    pi2_0_0_0_1 = ld_statistics.pi2(counts_between, populations=(0, 0, 0, 1), n_valid=n_valid_between)
    pi2_0_0_1_1 = ld_statistics.pi2(counts_between, populations=(0, 0, 1, 1), n_valid=n_valid_between)
    pi2_0_1_0_1 = ld_statistics.pi2(counts_between, populations=(0, 1, 0, 1), n_valid=n_valid_between)
    pi2_0_1_1_1 = ld_statistics.pi2(counts_between, populations=(0, 1, 1, 1), n_valid=n_valid_between)
    pi2_1_1_1_1 = ld_statistics.pi2(counts_pop2, n_valid=n_valid2)

    # Stack all 15 statistics
    return cp.stack([
        DD_0_0, DD_0_1, DD_1_1,
        Dz_0_0_0, Dz_0_0_1, Dz_0_1_1, Dz_1_0_0, Dz_1_0_1, Dz_1_1_1,
        pi2_0_0_0_0, pi2_0_0_0_1, pi2_0_0_1_1, pi2_0_1_0_1, pi2_0_1_1_1, pi2_1_1_1_1
    ], axis=1)


def _compute_single_pop_statistics_batch(counts, n_valid, ld_statistics):
    """
    Compute single-population LD statistics (DD, Dz, pi2) for a batch of pairs.

    Parameters
    ----------
    counts : cp.ndarray
        Shape (n_pairs, 4), haplotype counts [n11, n10, n01, n00]
    n_valid : cp.ndarray or None
        Valid sample counts per pair. None if no missing data.
    ld_statistics : module
        The ld_statistics module with dd, dz, pi2 functions

    Returns
    -------
    statistics : cp.ndarray
        Shape (n_pairs, 3), columns [DD, Dz, pi2] for each pair
    """
    DD = ld_statistics.dd(counts, n_valid=n_valid)
    Dz = ld_statistics.dz(counts, n_valid=n_valid)
    pi2 = ld_statistics.pi2(counts, n_valid=n_valid)
    return cp.stack([DD, Dz, pi2], axis=1)


def _ld_names(num_pops):
    """Generate LD statistic names matching moments.LD.Util.ld_names()."""
    names = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            names.append(f"DD_{ii}_{jj}")
    for ii in range(num_pops):
        for jj in range(num_pops):
            for kk in range(jj, num_pops):
                names.append(f"Dz_{ii}_{jj}_{kk}")
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            for kk in range(ii, num_pops):
                for ll in range(kk, num_pops):
                    if kk == ii == ll and jj != ii:
                        continue
                    if ii == kk and ll < jj:
                        continue
                    names.append(f"pi2_{ii}_{jj}_{kk}_{ll}")
    return names


def _het_names(num_pops):
    """Generate heterozygosity statistic names matching moments.LD.Util.het_names()."""
    names = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            names.append(f"H_{ii}_{jj}")
    return names


def _generate_stat_specs(num_pops):
    """Generate computation specs for each LD statistic.

    Each spec is (stat_name, [(weight, stat_type, pop_indices), ...])
    encoding the exact averaging logic from moments' _call_sgc().

    Returns
    -------
    list of (str, list of (float, str, tuple))
    """
    specs = []
    names = _ld_names(num_pops)

    for name in names:
        parts = name.split("_")
        stat_type = parts[0]
        pop_nums = tuple(int(p) for p in parts[1:])

        if stat_type == "DD":
            specs.append((name, [(1.0, 'dd', pop_nums)]))

        elif stat_type == "Dz":
            ii, jj, kk = pop_nums
            if jj == kk:
                specs.append((name, [(1.0, 'dz', (ii, jj, kk))]))
            else:
                specs.append((name, [
                    (0.5, 'dz', (ii, jj, kk)),
                    (0.5, 'dz', (ii, kk, jj)),
                ]))

        elif stat_type == "pi2":
            ii, jj, kk, ll = pop_nums
            if ii == jj:
                if kk == ll:
                    if ii == kk:
                        # All same
                        specs.append((name, [(1.0, 'pi2', (ii, jj, kk, ll))]))
                    else:
                        # (i,i,k,k) i!=k
                        specs.append((name, [
                            (0.5, 'pi2', (ii, jj, kk, ll)),
                            (0.5, 'pi2', (kk, ll, ii, jj)),
                        ]))
                else:
                    # (i,i,k,l) k!=l
                    specs.append((name, [
                        (0.25, 'pi2', (ii, jj, kk, ll)),
                        (0.25, 'pi2', (ii, jj, ll, kk)),
                        (0.25, 'pi2', (kk, ll, ii, jj)),
                        (0.25, 'pi2', (ll, kk, ii, jj)),
                    ]))
            else:
                if kk == ll:
                    # (i,j,k,k) i!=j
                    specs.append((name, [
                        (0.25, 'pi2', (ii, jj, kk, ll)),
                        (0.25, 'pi2', (jj, ii, kk, ll)),
                        (0.25, 'pi2', (kk, ll, ii, jj)),
                        (0.25, 'pi2', (kk, ll, jj, ii)),
                    ]))
                else:
                    # (i,j,k,l) i!=j, k!=l
                    specs.append((name, [
                        (0.125, 'pi2', (ii, jj, kk, ll)),
                        (0.125, 'pi2', (ii, jj, ll, kk)),
                        (0.125, 'pi2', (jj, ii, kk, ll)),
                        (0.125, 'pi2', (jj, ii, ll, kk)),
                        (0.125, 'pi2', (kk, ll, ii, jj)),
                        (0.125, 'pi2', (ll, kk, ii, jj)),
                        (0.125, 'pi2', (kk, ll, jj, ii)),
                        (0.125, 'pi2', (ll, kk, jj, ii)),
                    ]))

    return specs


class _PopData:
    """Precomputed per-population arrays for batch LD computation.

    Intentionally duplicates formulas from ld_statistics.py to operate
    on pre-split arrays, avoiding concatenation and function-call overhead.
    """
    __slots__ = ('c1', 'c2', 'c3', 'c4', 'n', 'D', 'pA', 'qA', 'pB', 'qB')

    def __init__(self, counts, n_valid):
        self.c1 = counts[:, 0].astype(cp.float64)
        self.c2 = counts[:, 1].astype(cp.float64)
        self.c3 = counts[:, 2].astype(cp.float64)
        self.c4 = counts[:, 3].astype(cp.float64)
        if n_valid is not None:
            self.n = n_valid.astype(cp.float64)
        else:
            self.n = self.c1 + self.c2 + self.c3 + self.c4
        self.D = self.c2 * self.c3 - self.c1 * self.c4
        self.pA = self.c1 + self.c2
        self.qA = self.c3 + self.c4
        self.pB = self.c1 + self.c3
        self.qB = self.c2 + self.c4


def _compute_multi_pop_statistics_batch(counts_per_pop, n_valid_per_pop,
                                        ld_statistics_module, stat_specs):
    """Compute all LD statistics for N populations using pre-computed specs.

    Precomputes per-population intermediates and caches raw results to
    minimize GPU kernel launches and avoid redundant computation.

    Parameters
    ----------
    counts_per_pop : list of cp.ndarray
        List of P arrays, each shape (n_pairs, 4)
    n_valid_per_pop : list of cp.ndarray or None
        List of P arrays, each shape (n_pairs,), or None entries
    ld_statistics_module : module
        Unused, kept for API compatibility with call sites.
    stat_specs : list
        From _generate_stat_specs()

    Returns
    -------
    cp.ndarray, shape (n_pairs, n_stats)
    """
    n_pairs = counts_per_pop[0].shape[0]
    n_stats = len(stat_specs)

    pops = [_PopData(counts_per_pop[p], n_valid_per_pop[p])
            for p in range(len(counts_per_pop))]

    # Compute each unique raw call once
    unique_calls = set()
    for _, calls in stat_specs:
        for _, stat_type, pop_indices in calls:
            unique_calls.add((stat_type, pop_indices))

    raw_cache = {}

    # Batch-compute all Dz values via einsum + advanced indexing
    dz_calls = [pi for st, pi in unique_calls if st == 'dz']
    if dz_calls:
        dz_results = _compute_all_dz(pops, dz_calls)
        for pi, val in zip(dz_calls, dz_results):
            raw_cache[('dz', pi)] = val

    # Batch-compute all pi2 values via einsum + advanced indexing
    pi2_calls = [pi for st, pi in unique_calls if st == 'pi2']
    if pi2_calls:
        pi2_results = _compute_all_pi2(pops, pi2_calls)
        for pi, val in zip(pi2_calls, pi2_results):
            raw_cache[('pi2', pi)] = val

    for stat_type, pop_indices in unique_calls:
        if stat_type == 'dd':
            raw_cache[('dd', pop_indices)] = _dd_raw(pop_indices, pops)

    # Assemble weighted sums from cached raw values
    result = cp.zeros((n_pairs, n_stats), dtype=cp.float64)
    for stat_idx, (_, calls) in enumerate(stat_specs):
        if len(calls) == 1:
            w, st, pi = calls[0]
            if w == 1.0:
                result[:, stat_idx] = raw_cache[(st, pi)]
            else:
                result[:, stat_idx] = w * raw_cache[(st, pi)]
        else:
            val = sum(w * raw_cache[(st, pi)] for w, st, pi in calls)
            result[:, stat_idx] = val

    return result


def _dd_raw(pop_indices, pops):
    """Compute raw DD from precomputed per-population arrays."""
    i, j = pop_indices
    if i == j:
        p = pops[i]
        numer = (p.c1 * (p.c1 - 1) * p.c4 * (p.c4 - 1)
                 + p.c2 * (p.c2 - 1) * p.c3 * (p.c3 - 1)
                 - 2 * p.c1 * p.c2 * p.c3 * p.c4)
        denom = p.n * (p.n - 1) * (p.n - 2) * (p.n - 3)
        valid = p.n >= 4
    else:
        numer = pops[i].D * pops[j].D
        denom = pops[i].n * (pops[i].n - 1) * pops[j].n * (pops[j].n - 1)
        valid = (pops[i].n >= 2) & (pops[j].n >= 2)
    return cp.where(valid, numer / cp.maximum(denom, 1), 0.0)


def _safe_div(numer, denom, valid):
    """Divide numer/denom where valid, zero elsewhere. No intermediate copy."""
    return cp.where(valid, numer / cp.maximum(denom, 1), 0.0)


def _compute_all_pi2(pops, pi2_calls):
    """Batch-compute all raw pi2 values using einsum-derived matrices.

    Precomputes cross-population heterozygosity matrices H_A and H_B via
    einsum, then serves multiple pi2 formula cases from them. Remaining
    complex cases (shared-pop, ijij, single-pop) use per-call computation
    with advanced indexing where possible.

    Parameters
    ----------
    pops : list of _PopData
    pi2_calls : list of (i, j, k, l) tuples

    Returns
    -------
    list of cp.ndarray, one per call in pi2_calls order
    """
    P = len(pops)
    n_pairs = pops[0].n.shape[0]

    # Stack only what's needed for einsum: shape (P, N)
    pA = cp.stack([p.pA for p in pops])
    qA = cp.stack([p.qA for p in pops])
    pB = cp.stack([p.pB for p in pops])
    qB = cp.stack([p.qB for p in pops])
    n = cp.stack([p.n for p in pops])

    # H_A[i,j] = pA_i*qA_j + qA_i*pA_j via einsum: shape (P, P, N)
    pq_A = cp.einsum('in,jn->ijn', pA, qA)
    H_A = pq_A + pq_A.transpose(1, 0, 2)
    del pq_A, pA, qA

    # H_B[k,l] = pB_k*qB_l + qB_k*pB_l via einsum: shape (P, P, N)
    pq_B = cp.einsum('in,jn->ijn', pB, qB)
    H_B = pq_B + pq_B.transpose(1, 0, 2)
    del pq_B, pB, qB

    # Within-pop heterozygosity: pA_i*qA_i (NOT the diagonal of H, which is 2x)
    fA = cp.stack([pops[p].pA * pops[p].qA for p in range(P)])  # (P, N)
    fB = cp.stack([pops[p].pB * pops[p].qB for p in range(P)])  # (P, N)

    # Group calls by case type
    groups = {
        'same': [], 'triple': [], 'iikk': [], 'iikl': [],
        'ijkk': [], 'ijij': [], 'shared': [], 'alldiff': [],
    }
    call_order = {}

    for idx, (i, j, k, l) in enumerate(pi2_calls):
        cnt = {}
        for p in (i, j, k, l):
            cnt[p] = cnt.get(p, 0) + 1
        nu = len(cnt)
        mc = max(cnt.values())

        if nu == 1:
            key = 'same'
        elif mc == 3:
            key = 'triple'
        elif i == j and k == l:
            key = 'iikk'
        elif i == j:
            key = 'iikl'
        elif k == l:
            key = 'ijkk'
        elif (i == k and j == l) or (i == l and j == k):
            key = 'ijij'
        elif nu == 3:
            key = 'shared'
        else:
            key = 'alldiff'

        call_order[idx] = (key, len(groups[key]))
        groups[key].append((i, j, k, l))

    group_results = {}

    # --- Cases served by precomputed H_A, H_B, fA, fB ---

    # All different: numer = H_A[i,j] * H_B[k,l] / 4
    if groups['alldiff']:
        ii = [t[0] for t in groups['alldiff']]
        jj = [t[1] for t in groups['alldiff']]
        kk = [t[2] for t in groups['alldiff']]
        ll = [t[3] for t in groups['alldiff']]
        numer = H_A[ii, jj] * H_B[kk, ll] / 4.0
        denom = n[ii] * n[jj] * n[kk] * n[ll]
        valid = (n[ii] >= 1) & (n[jj] >= 1) & (n[kk] >= 1) & (n[ll] >= 1)
        batch = _safe_div(numer, denom, valid)
        group_results['alldiff'] = [batch[r] for r in range(len(groups['alldiff']))]

    # pi2(i,i,k,k): numer = fA[i] * fB[k]
    if groups['iikk']:
        ii = [t[0] for t in groups['iikk']]
        kk = [t[2] for t in groups['iikk']]
        numer = fA[ii] * fB[kk]
        denom = n[ii] * (n[ii] - 1) * n[kk] * (n[kk] - 1)
        valid = (n[ii] >= 2) & (n[kk] >= 2)
        batch = _safe_div(numer, denom, valid)
        group_results['iikk'] = [batch[r] for r in range(len(groups['iikk']))]

    # pi2(i,i,k,l): numer = fA[i] * H_B[k,l] / 2
    if groups['iikl']:
        ii = [t[0] for t in groups['iikl']]
        kk = [t[2] for t in groups['iikl']]
        ll = [t[3] for t in groups['iikl']]
        numer = fA[ii] * H_B[kk, ll] / 2.0
        denom = n[ii] * (n[ii] - 1) * n[kk] * n[ll]
        valid = (n[ii] >= 2) & (n[kk] >= 1) & (n[ll] >= 1)
        batch = _safe_div(numer, denom, valid)
        group_results['iikl'] = [batch[r] for r in range(len(groups['iikl']))]

    # pi2(i,j,k,k): numer = fB[k] * H_A[i,j] / 2
    if groups['ijkk']:
        ii = [t[0] for t in groups['ijkk']]
        jj = [t[1] for t in groups['ijkk']]
        kk = [t[2] for t in groups['ijkk']]
        numer = fB[kk] * H_A[ii, jj] / 2.0
        denom = n[kk] * (n[kk] - 1) * n[ii] * n[jj]
        valid = (n[kk] >= 2) & (n[ii] >= 1) & (n[jj] >= 1)
        batch = _safe_div(numer, denom, valid)
        group_results['ijkk'] = [batch[r] for r in range(len(groups['ijkk']))]

    # --- Cases computed via batched advanced indexing ---

    # Triple (3 same, 1 different): batch by (triple_pop, single_pop)
    if groups['triple']:
        trip_cache = {}
        results = []
        for i, j, k, l in groups['triple']:
            cnt = {}
            for p in (i, j, k, l):
                cnt[p] = cnt.get(p, 0) + 1
            tp = [p for p, c in cnt.items() if c == 3][0]
            sp = [p for p, c in cnt.items() if c == 1][0]
            cache_key = (tp, sp)
            if cache_key not in trip_cache:
                t, s = pops[tp], pops[sp]
                numer = (
                    -(t.pA * t.c4 * s.pB)
                    - (t.c2 * t.qA * s.pB)
                    + (t.pA * (t.c2 + t.c4) * t.qA * s.pB)
                    + (t.pA * t.qA * (-2*s.c2 - 2*s.c4))
                    + (t.pA * t.c4 * s.qB)
                    + (t.c2 * t.qA * s.qB)
                    + (t.pA * t.pB * t.qA * s.qB)
                ) / 2.0
                denom = s.n * t.n * (t.n - 1) * (t.n - 2)
                valid = (t.n >= 3) & (s.n >= 1)
                trip_cache[cache_key] = _safe_div(numer, denom, valid)
            results.append(trip_cache[cache_key])
        group_results['triple'] = results

    # Shared pop: batch by (shared, other1, other2)
    if groups['shared']:
        shared_cache = {}
        results = []
        for i, j, k, l in groups['shared']:
            if i == k:
                si, ai, bi = i, j, l
            elif i == l:
                si, ai, bi = i, j, k
            elif j == k:
                si, ai, bi = j, i, l
            elif j == l:
                si, ai, bi = j, i, k
            else:
                results.append(cp.zeros(n_pairs, dtype=cp.float64))
                continue
            cache_key = (si, ai, bi)
            if cache_key not in shared_cache:
                s, a, b = pops[si], pops[ai], pops[bi]
                numer = (
                    s.c4 ** 2 * a.pA * b.pB
                    + s.c2 ** 2 * a.qA * b.pB
                    + (-1 + s.c1 + s.c3) * (s.c3 * a.pA + s.c1 * a.qA) * b.qB
                    + s.c4 * (
                        s.c1 * a.qA * b.pB
                        + a.c1 * ((-1 + s.c3) * b.c1 + s.c3 * b.c2 - b.c3
                                  + s.c3 * b.c3 + s.c3 * b.c4 + s.c1 * b.qB)
                        + a.c2 * ((-1 + s.c3) * b.c1 + s.c3 * b.c2 - b.c3
                                  + s.c3 * b.c3 + s.c3 * b.c4 + s.c1 * b.qB)
                    )
                    + s.c2 * (
                        s.c4 * (a.pA + a.qA) * b.pB
                        + s.c3 * (a.c1 * b.pB + a.c2 * b.pB + a.qA * b.qB)
                        + a.qA * ((-1 + s.c1) * b.c1 - b.c3 + s.c1 * (b.c2 + b.c3 + b.c4))
                    )
                ) / 4.0
                denom = s.n * (s.n - 1) * a.n * b.n
                valid = (s.n >= 2) & (a.n >= 1) & (b.n >= 1)
                shared_cache[cache_key] = _safe_div(numer, denom, valid)
            results.append(shared_cache[cache_key])
        group_results['shared'] = results

    # pi2(i,j,i,j) / pi2(i,j,j,i): batch by (a, b) pair
    if groups['ijij']:
        ijij_cache = {}
        results = []
        for i, j, k, l in groups['ijij']:
            cache_key = (i, j)
            if cache_key not in ijij_cache:
                a, b = pops[i], pops[j]
                numer = (
                    (a.qB * a.qA * b.pA * b.pB) / 4.0
                    + (a.pB * a.qA * b.pA * b.qB) / 4.0
                    + (a.pA * a.qB * b.pB * b.qA) / 4.0
                    + (a.pA * a.pB * b.qB * b.qA) / 4.0
                    + (
                        -(a.c2 * a.c3 * b.c1)
                        + a.c4 * b.c1 - a.c2 * a.c4 * b.c1
                        - a.c3 * a.c4 * b.c1 - a.c4 ** 2 * b.c1
                        - a.c4 * b.c1 ** 2 + a.c3 * b.c2
                        - a.c1 * a.c3 * b.c2 - a.c3 ** 2 * b.c2
                        - a.c1 * a.c4 * b.c2 - a.c3 * a.c4 * b.c2
                        - a.c3 * b.c1 * b.c2 - a.c4 * b.c1 * b.c2
                        - a.c3 * b.c2 ** 2 + a.c2 * b.c3
                        - a.c1 * a.c2 * b.c3 - a.c2 ** 2 * b.c3
                        - a.c1 * a.c4 * b.c3 - a.c2 * a.c4 * b.c3
                        - a.c2 * b.c1 * b.c3 - a.c4 * b.c1 * b.c3
                        - a.c1 * b.c2 * b.c3 - a.c4 * b.c2 * b.c3
                        - a.c2 * b.c3 ** 2 + a.c1 * b.c4
                        - a.c1 ** 2 * b.c4 - a.c1 * a.c2 * b.c4
                        - a.c1 * a.c3 * b.c4 - a.c2 * a.c3 * b.c4
                        - a.c2 * b.c1 * b.c4 - a.c3 * b.c1 * b.c4
                        - a.c1 * b.c2 * b.c4 - a.c3 * b.c2 * b.c4
                        - a.c1 * b.c3 * b.c4 - a.c2 * b.c3 * b.c4
                        - a.c1 * b.c4 ** 2
                    ) / 4.0
                )
                denom = a.n * (a.n - 1) * b.n * (b.n - 1)
                valid = (a.n >= 2) & (b.n >= 2)
                ijij_cache[cache_key] = _safe_div(numer, denom, valid)
            results.append(ijij_cache[cache_key])
        group_results['ijij'] = results

    # Single pop: per-call
    if groups['same']:
        results = []
        for i, j, k, l in groups['same']:
            p = pops[i]
            numer = (
                p.pA * p.pB * p.qB * p.qA
                - p.c1 * p.c4 * (-1 + p.c1 + 3*p.c2 + 3*p.c3 + p.c4)
                - p.c2 * p.c3 * (-1 + 3*p.c1 + p.c2 + p.c3 + 3*p.c4)
            )
            denom = p.n * (p.n - 1) * (p.n - 2) * (p.n - 3)
            valid = p.n >= 4
            results.append(_safe_div(numer, denom, valid))
        group_results['same'] = results

    # Reassemble in original call order
    output = [None] * len(pi2_calls)
    for idx in range(len(pi2_calls)):
        grp, pos = call_order[idx]
        output[idx] = group_results[grp][pos]
    return output


def _compute_all_dz(pops, dz_calls):
    """Batch-compute all raw Dz values using einsum for all-different cases.

    Groups calls by population pattern and vectorizes within each group,
    reducing GPU kernel launches from O(n_calls) to O(n_groups).

    Parameters
    ----------
    pops : list of _PopData
    dz_calls : list of (p1, p2, p3) tuples

    Returns
    -------
    list of cp.ndarray, one per call in dz_calls order
    """
    P = len(pops)
    n_pairs = pops[0].n.shape[0]

    # Stack per-pop arrays: shape (P, N)
    D_stack = cp.stack([p.D for p in pops])
    fdA_stack = cp.stack([p.pA - p.qA for p in pops])   # pA - qA
    fdB_stack = cp.stack([p.pB - p.qB for p in pops])   # pB - qB = c1-c2+c3-c4
    n_stack = cp.stack([p.n for p in pops])

    # Group calls by case type
    groups = {'same': [], 'p1p2': [], 'p1p3': [], 'p2p3': [], 'diff': []}
    call_order = {}  # maps original index -> (group, index_within_group)
    for idx, (p1, p2, p3) in enumerate(dz_calls):
        if p1 == p2 == p3:
            call_order[idx] = ('same', len(groups['same']))
            groups['same'].append((p1, p2, p3))
        elif p1 == p2:
            call_order[idx] = ('p1p2', len(groups['p1p2']))
            groups['p1p2'].append((p1, p2, p3))
        elif p1 == p3:
            call_order[idx] = ('p1p3', len(groups['p1p3']))
            groups['p1p3'].append((p1, p2, p3))
        elif p2 == p3:
            call_order[idx] = ('p2p3', len(groups['p2p3']))
            groups['p2p3'].append((p1, p2, p3))
        else:
            call_order[idx] = ('diff', len(groups['diff']))
            groups['diff'].append((p1, p2, p3))

    group_results = {}

    # All-different: numer = -(D_i * fdA_j * fdB_k) via einsum
    if groups['diff']:
        # Full tensor: (P, P, P, N) -- only compute if P is small
        if P <= 4:
            # einsum: shape (P, P, P, N)
            numer_tensor = -cp.einsum('in,jn,kn->ijkn', D_stack, fdA_stack, fdB_stack)
            denom_tensor = (n_stack[:, None, None, :]
                            * (n_stack[:, None, None, :] - 1)
                            * n_stack[None, :, None, :]
                            * n_stack[None, None, :, :])
            valid_tensor = ((n_stack[:, None, None, :] >= 2)
                            & (n_stack[None, :, None, :] >= 1)
                            & (n_stack[None, None, :, :] >= 1))
            full = cp.where(valid_tensor, numer_tensor / cp.maximum(denom_tensor, 1), 0.0)
            results = [full[p1, p2, p3] for p1, p2, p3 in groups['diff']]
            del numer_tensor, denom_tensor, valid_tensor, full
        else:
            # Fallback to advanced indexing for large P
            ii = cp.array([t[0] for t in groups['diff']])
            jj = cp.array([t[1] for t in groups['diff']])
            kk = cp.array([t[2] for t in groups['diff']])
            numer = -(D_stack[ii] * fdA_stack[jj] * fdB_stack[kk])
            denom = (n_stack[ii] * (n_stack[ii] - 1) * n_stack[jj] * n_stack[kk])
            valid = (n_stack[ii] >= 2) & (n_stack[jj] >= 1) & (n_stack[kk] >= 1)
            batch = cp.where(valid, numer / cp.maximum(denom, 1), 0.0)
            results = [batch[r] for r in range(len(groups['diff']))]
            del numer, denom, valid, batch
        group_results['diff'] = results

    # Dz(i,i,j): numer = (-fdA_i) * (-D_i) * (-fdB_j) = -fdA_i * D_i * fdB_j
    if groups['p1p2']:
        ii = cp.array([t[0] for t in groups['p1p2']])
        kk = cp.array([t[2] for t in groups['p1p2']])
        numer = -fdA_stack[ii] * D_stack[ii] * fdB_stack[kk]
        denom = n_stack[kk] * n_stack[ii] * (n_stack[ii] - 1) * (n_stack[ii] - 2)
        valid = (n_stack[ii] >= 3) & (n_stack[kk] >= 1)
        batch = _safe_div(numer, denom, valid)
        group_results['p1p2'] = [batch[r] for r in range(len(groups['p1p2']))]

    # Dz(i,j,i): numer = (-fdB_i) * (-D_i) * (-fdA_j) = -fdB_i * D_i * fdA_j
    if groups['p1p3']:
        ii = cp.array([t[0] for t in groups['p1p3']])
        jj = cp.array([t[1] for t in groups['p1p3']])
        numer = -fdB_stack[ii] * D_stack[ii] * fdA_stack[jj]
        denom = n_stack[jj] * n_stack[ii] * (n_stack[ii] - 1) * (n_stack[ii] - 2)
        valid = (n_stack[ii] >= 3) & (n_stack[jj] >= 1)
        batch = _safe_div(numer, denom, valid)
        group_results['p1p3'] = [batch[r] for r in range(len(groups['p1p3']))]

    # Dz(i,j,j): two-term formula, computed per-call (can't easily batch)
    if groups['p2p3']:
        results = []
        for p1, p2, p3 in groups['p2p3']:
            a, b = pops[p1], pops[p2]
            numer = ((-a.D) * (-b.c1 + b.c2 + b.c3 - b.c4)
                     + (-a.D) * (-b.c1 + b.c2 - b.c3 + b.c4) * (b.qA - b.pA))
            denom = a.n * (a.n - 1) * b.n * (b.n - 1)
            valid = (a.n >= 2) & (b.n >= 2)
            results.append(_safe_div(numer, denom, valid))
        group_results['p2p3'] = results

    # Dz(i,i,i): single-pop with correction terms, computed per-call
    if groups['same']:
        results = []
        for p1, _, _ in groups['same']:
            a = pops[p1]
            diff = a.c1 * a.c4 - a.c2 * a.c3
            numer = (diff * (a.qA - a.pA) * (a.qB - a.pB)
                     + diff * ((a.c2 + a.c3) - (a.c1 + a.c4))
                     + 2 * (a.c2 * a.c3 + a.c1 * a.c4))
            denom = a.n * (a.n - 1) * (a.n - 2) * (a.n - 3)
            valid = a.n >= 4
            results.append(_safe_div(numer, denom, valid))
        group_results['same'] = results

    # Reassemble in original call order
    output = [None] * len(dz_calls)
    for idx in range(len(dz_calls)):
        grp, pos = call_order[idx]
        output[idx] = group_results[grp][pos]
    return output


# ---------------------------------------------------------------------------
# Genotype (diploid) batch computation -- parallels haplotype path above
# ---------------------------------------------------------------------------


class _PopDataGeno:
    """Precomputed per-population arrays for genotype-based LD computation.

    Stores 9 genotype configuration counts and derived frequency quantities.
    """
    __slots__ = ('g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9',
                 'n', 'D_geno', 'pA', 'qA', 'pB', 'qB', 'fdA', 'fdB')

    def __init__(self, counts, n_valid):
        # Our counting: combo = geno_i*3+geno_j gives columns:
        # col0=n00, col1=n01, col2=n02, col3=n10, col4=n11, col5=n12, col6=n20, col7=n21, col8=n22
        # Moments convention: g1=n22, g2=n21, g3=n20, g4=n12, g5=n11, g6=n10, g7=n02, g8=n01, g9=n00
        self.g1 = counts[:, 8].astype(cp.float64)  # n22
        self.g2 = counts[:, 7].astype(cp.float64)  # n21
        self.g3 = counts[:, 6].astype(cp.float64)  # n20
        self.g4 = counts[:, 5].astype(cp.float64)  # n12
        self.g5 = counts[:, 4].astype(cp.float64)  # n11
        self.g6 = counts[:, 3].astype(cp.float64)  # n10
        self.g7 = counts[:, 2].astype(cp.float64)  # n02
        self.g8 = counts[:, 1].astype(cp.float64)  # n01
        self.g9 = counts[:, 0].astype(cp.float64)  # n00
        if n_valid is not None:
            self.n = n_valid.astype(cp.float64)
        else:
            self.n = (self.g1 + self.g2 + self.g3 + self.g4 + self.g5
                      + self.g6 + self.g7 + self.g8 + self.g9)
        # D from genotype frequencies (used in between-pop DD, Dz all-diff)
        self.D_geno = (
            -(self.g2 / 2 + self.g3 + self.g5 / 4 + self.g6 / 2)
            * (self.g4 / 2 + self.g5 / 4 + self.g7 + self.g8 / 2)
            + (self.g1 + self.g2 / 2 + self.g4 / 2 + self.g5 / 4)
            * (self.g5 / 4 + self.g6 / 2 + self.g8 / 2 + self.g9)
        )
        # Allele frequency terms for between-pop formulas
        self.pA = self.g1 + self.g2 + self.g3 + self.g4 / 2 + self.g5 / 2 + self.g6 / 2
        self.qA = self.g4 / 2 + self.g5 / 2 + self.g6 / 2 + self.g7 + self.g8 + self.g9
        self.pB = self.g1 + self.g2 / 2 + self.g4 + self.g5 / 2 + self.g7 + self.g8 / 2
        self.qB = self.g2 / 2 + self.g3 + self.g5 / 2 + self.g6 + self.g8 / 2 + self.g9
        self.fdA = -self.g1 + self.g3 - self.g4 + self.g6 - self.g7 + self.g9
        self.fdB = -self.g1 - self.g2 - self.g3 + self.g7 + self.g8 + self.g9


def _compute_multi_pop_statistics_batch_geno(counts_per_pop, n_valid_per_pop,
                                             _, stat_specs):
    """Compute all LD statistics for N populations using genotype counts.

    Parallel to _compute_multi_pop_statistics_batch but uses 9-way genotype
    counts and genotype-specific formulas from ld_statistics_genotype.
    """
    from . import ld_statistics_genotype as ldg

    n_pairs = counts_per_pop[0].shape[0]
    n_stats = len(stat_specs)

    pops = [_PopDataGeno(counts_per_pop[p], n_valid_per_pop[p])
            for p in range(len(counts_per_pop))]

    # Compute all unique raw calls with caches
    unique_calls = set()
    for _, calls in stat_specs:
        for _, stat_type, pop_indices in calls:
            unique_calls.add((stat_type, pop_indices))

    raw_cache = {}

    # DD -- few calls, compute individually
    for st, pi in unique_calls:
        if st == 'dd':
            i, j = pi
            if i == j:
                raw_cache[('dd', pi)] = ldg.dd_geno_single(pops[i])
            else:
                raw_cache[('dd', pi)] = ldg.dd_geno_between(pops[i], pops[j])

    # Dz -- batch via einsum (same pattern as haplotype _compute_all_dz)
    dz_calls = [pi for st, pi in unique_calls if st == 'dz']
    if dz_calls:
        dz_results = _compute_all_dz_geno(pops, dz_calls)
        for pi, val in zip(dz_calls, dz_results):
            raw_cache[('dz', pi)] = val

    # pi2 -- batch via einsum + semantic dedup
    pi2_calls = [pi for st, pi in unique_calls if st == 'pi2']
    if pi2_calls:
        pi2_results = _compute_all_pi2_geno(pops, pi2_calls)
        for pi, val in zip(pi2_calls, pi2_results):
            raw_cache[('pi2', pi)] = val

    # Assemble weighted sums
    result = cp.zeros((n_pairs, n_stats), dtype=cp.float64)
    for stat_idx, (_, calls) in enumerate(stat_specs):
        if len(calls) == 1:
            w, st, pi = calls[0]
            if w == 1.0:
                result[:, stat_idx] = raw_cache[(st, pi)]
            else:
                result[:, stat_idx] = w * raw_cache[(st, pi)]
        else:
            val = sum(w * raw_cache[(st, pi)] for w, st, pi in calls)
            result[:, stat_idx] = val

    return result


def _compute_all_dz_geno(pops, dz_calls):
    """Batch-compute all raw Dz values for genotype data."""
    from . import ld_statistics_genotype as ldg

    P = len(pops)
    # Stack for einsum
    D_stack = cp.stack([p.D_geno for p in pops])
    fdA_stack = cp.stack([p.fdA for p in pops])
    fdB_stack = cp.stack([p.fdB for p in pops])
    n_stack = cp.stack([p.n for p in pops])

    groups = {'same': [], 'p1p2': [], 'p1p3': [], 'p2p3': [], 'diff': []}
    call_order = {}
    for idx, (p1, p2, p3) in enumerate(dz_calls):
        if p1 == p2 == p3:
            key = 'same'
        elif p1 == p2:
            key = 'p1p2'
        elif p1 == p3:
            key = 'p1p3'
        elif p2 == p3:
            key = 'p2p3'
        else:
            key = 'diff'
        call_order[idx] = (key, len(groups[key]))
        groups[key].append((p1, p2, p3))

    group_results = {}

    # All-different: D_geno_i * fdB_j * fdA_k (einsum)
    if groups['diff']:
        if P <= 4:
            numer_tensor = cp.einsum('in,jn,kn->ijkn', D_stack, fdB_stack, fdA_stack)
            denom_tensor = (n_stack[:, None, None, :]
                            * (n_stack[:, None, None, :] - 1)
                            * n_stack[None, :, None, :]
                            * n_stack[None, None, :, :])
            valid_tensor = ((n_stack[:, None, None, :] >= 2)
                            & (n_stack[None, :, None, :] >= 1)
                            & (n_stack[None, None, :, :] >= 1))
            full = 2.0 * _safe_div(numer_tensor, denom_tensor, valid_tensor)
            results = [full[p1, p2, p3] for p1, p2, p3 in groups['diff']]
            del numer_tensor, denom_tensor, valid_tensor, full
        else:
            ii = cp.array([t[0] for t in groups['diff']])
            jj = cp.array([t[1] for t in groups['diff']])
            kk = cp.array([t[2] for t in groups['diff']])
            numer = D_stack[ii] * fdB_stack[jj] * fdA_stack[kk]
            denom = n_stack[ii] * (n_stack[ii] - 1) * n_stack[jj] * n_stack[kk]
            valid = (n_stack[ii] >= 2) & (n_stack[jj] >= 1) & (n_stack[kk] >= 1)
            batch = 2.0 * _safe_div(numer, denom, valid)
            results = [batch[r] for r in range(len(groups['diff']))]
        group_results['diff'] = results

    # p1==p2 (Dz(i,i,j)): batch with advanced indexing
    if groups['p1p2']:
        results = []
        for p1, p2, p3 in groups['p1p2']:
            results.append(ldg.dz_geno_p1p2(pops[p1], pops[p3]))
        group_results['p1p2'] = results

    # p1==p3 (Dz(i,j,i))
    if groups['p1p3']:
        results = []
        for p1, p2, p3 in groups['p1p3']:
            results.append(ldg.dz_geno_p1p3(pops[p1], pops[p2]))
        group_results['p1p3'] = results

    # p2==p3 (Dz(i,j,j))
    if groups['p2p3']:
        results = []
        for p1, p2, p3 in groups['p2p3']:
            results.append(ldg.dz_geno_p2p3(pops[p1], pops[p2]))
        group_results['p2p3'] = results

    # All same (Dz(i,i,i))
    if groups['same']:
        results = []
        for p1, _, _ in groups['same']:
            results.append(ldg.dz_geno_single(pops[p1]))
        group_results['same'] = results

    output = [None] * len(dz_calls)
    for idx in range(len(dz_calls)):
        grp, pos = call_order[idx]
        output[idx] = group_results[grp][pos]
    return output


def _compute_all_pi2_geno(pops, pi2_calls):
    """Batch-compute all raw pi2 values for genotype data."""
    from . import ld_statistics_genotype as ldg

    P = len(pops)
    n_pairs = pops[0].n.shape[0]

    # Genotype pi2 formulas have finite-sample corrections that prevent
    # the H_A/H_B einsum factorization used in the haplotype path.
    # All cases use the full formula functions with semantic dedup caches.

    groups = {
        'same': [], 'triple': [], 'iikk': [], 'iikl': [],
        'ijkk': [], 'ijij': [], 'shared': [], 'alldiff': [],
    }
    call_order = {}
    for idx, (i, j, k, l) in enumerate(pi2_calls):
        cnt = {}
        for p in (i, j, k, l):
            cnt[p] = cnt.get(p, 0) + 1
        nu = len(cnt)
        mc = max(cnt.values())
        if nu == 1:
            key = 'same'
        elif mc == 3:
            key = 'triple'
        elif i == j and k == l:
            key = 'iikk'
        elif i == j:
            key = 'iikl'
        elif k == l:
            key = 'ijkk'
        elif (i == k and j == l) or (i == l and j == k):
            key = 'ijij'
        elif nu == 3:
            key = 'shared'
        else:
            key = 'alldiff'
        call_order[idx] = (key, len(groups[key]))
        groups[key].append((i, j, k, l))

    group_results = {}

    # All-different
    if groups['alldiff']:
        alldiff_cache = {}
        results = []
        for i, j, k, l in groups['alldiff']:
            cache_key = (i, j, k, l)
            if cache_key not in alldiff_cache:
                alldiff_cache[cache_key] = ldg.pi2_geno_alldiff(
                    pops[i], pops[j], pops[k], pops[l])
            results.append(alldiff_cache[cache_key])
        group_results['alldiff'] = results

    # pi2(i,i,k,k)
    if groups['iikk']:
        iikk_cache = {}
        results = []
        for i, j, k, l in groups['iikk']:
            cache_key = (i, k)
            if cache_key not in iikk_cache:
                iikk_cache[cache_key] = ldg.pi2_geno_iikk(pops[i], pops[k])
            results.append(iikk_cache[cache_key])
        group_results['iikk'] = results

    # pi2(i,i,k,l)
    if groups['iikl']:
        iikl_cache = {}
        results = []
        for i, j, k, l in groups['iikl']:
            cache_key = (i, k, l)
            if cache_key not in iikl_cache:
                iikl_cache[cache_key] = ldg.pi2_geno_iikl(pops[i], pops[k], pops[l])
            results.append(iikl_cache[cache_key])
        group_results['iikl'] = results

    # pi2(i,j,k,k)
    if groups['ijkk']:
        ijkk_cache = {}
        results = []
        for i, j, k, l in groups['ijkk']:
            cache_key = (i, j, k)
            if cache_key not in ijkk_cache:
                ijkk_cache[cache_key] = ldg.pi2_geno_ijkk(pops[k], pops[i], pops[j])
            results.append(ijkk_cache[cache_key])
        group_results['ijkk'] = results

    if groups['triple']:
        trip_cache = {}
        results = []
        for i, j, k, l in groups['triple']:
            cnt = {}
            for p in (i, j, k, l):
                cnt[p] = cnt.get(p, 0) + 1
            tp = [p for p, c in cnt.items() if c == 3][0]
            sp = [p for p, c in cnt.items() if c == 1][0]
            # Check if triple pop occupies both positions in the first pair
            first_pair_triple = (i == tp and j == tp)
            cache_key = (tp, sp, first_pair_triple)
            if cache_key not in trip_cache:
                if first_pair_triple:
                    trip_cache[cache_key] = ldg.pi2_geno_triple_123(pops[tp], pops[sp])
                else:
                    trip_cache[cache_key] = ldg.pi2_geno_triple_134(pops[tp], pops[sp])
            results.append(trip_cache[cache_key])
        group_results['triple'] = results

    if groups['shared']:
        shared_cache = {}
        results = []
        for i, j, k, l in groups['shared']:
            if i == k:
                si, ai, bi = i, j, l
            elif i == l:
                si, ai, bi = i, j, k
            elif j == k:
                si, ai, bi = j, i, l
            elif j == l:
                si, ai, bi = j, i, k
            else:
                results.append(cp.zeros(n_pairs, dtype=cp.float64))
                continue
            cache_key = (si, ai, bi)
            if cache_key not in shared_cache:
                shared_cache[cache_key] = ldg.pi2_geno_shared(pops[si], pops[ai], pops[bi])
            results.append(shared_cache[cache_key])
        group_results['shared'] = results

    if groups['ijij']:
        ijij_cache = {}
        results = []
        for i, j, k, l in groups['ijij']:
            cache_key = (i, j)
            if cache_key not in ijij_cache:
                ijij_cache[cache_key] = ldg.pi2_geno_ijij(pops[i], pops[j])
            results.append(ijij_cache[cache_key])
        group_results['ijij'] = results

    if groups['same']:
        results = []
        for i, j, k, l in groups['same']:
            results.append(ldg.pi2_geno_single(pops[i]))
        group_results['same'] = results

    output = [None] * len(pi2_calls)
    for idx in range(len(pi2_calls)):
        grp, pos = call_order[idx]
        output[idx] = group_results[grp][pos]
    return output
