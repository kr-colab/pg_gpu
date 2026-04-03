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
# Re-exports from ld_pipeline for backward compatibility
# =============================================================================
from .ld_pipeline import (  # noqa: E402, F401
    estimate_ld_chunk_size as _estimate_ld_chunk_size,
    generate_pairs_within_distance as _generate_pairs_within_distance,
    compute_counts_for_pairs as _compute_counts_for_pairs,
    compute_genotype_counts_for_pairs as _compute_genotype_counts_for_pairs,
    compute_two_pop_statistics_batch as _compute_two_pop_statistics_batch,
    compute_single_pop_statistics_batch as _compute_single_pop_statistics_batch,
    ld_names as _ld_names,
    het_names as _het_names,
    generate_stat_specs as _generate_stat_specs,
    PopData as _PopData,
)

