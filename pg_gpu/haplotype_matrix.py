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
    """
    def __init__(self, 
                 genotypes: cp.ndarray, 
                 positions: cp.ndarray,
                 chrom_start: int = None,
                 chrom_end: int = None,
                 ):
        # test for empty genotypes
        if genotypes.size == 0:
            raise ValueError("genotypes cannot be empty")
        # test for empty positions
        if positions.size == 0:
            raise ValueError("positions cannot be empty")
        self.haplotypes = genotypes
        self.positions = positions
        self.chrom_start = chrom_start
        self.chrom_end = chrom_end

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
        # convert to cupy array
        haplotypes = cp.array(haplotypes)
        # convert positions to cupy array
        positions = cp.array(vcf['variants/POS'])   
        # get the chromosome start and end
        chrom_start = positions[0]
        chrom_end = positions[-1]
        return cls(haplotypes, positions, chrom_start, chrom_end)

    @classmethod
    def from_ts(cls, ts) -> 'HaplotypeMatrix':
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
    
    def get_subset(self, positions: cp.ndarray) -> "HaplotypeMatrix":
        """
        Get a subset of the haplotype matrix based on the provided positions.
        
        Parameters:
            positions (cp.ndarray): An array of indices to select from the haplotype matrix.
            
        Returns:
            HaplotypeMatrix: A new instance containing the subset of the haplotype matrix.
        """
        # Ensure positions are valid indices
        if positions.ndim != 1 or not cp.all((positions >= 0) & (positions < self.haplotypes.shape[0])):
            raise ValueError("Positions must be valid indices within the haplotype matrix.")
        
        # Create new arrays with the subset
        subset_haplotypes = cp.array(self.haplotypes[:, positions])
        subset_positions = cp.array(self.positions[positions])
        
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
        
        # Find indices of positions within the specified range
        indices = cp.where((self.positions >= low) & (self.positions < high))[0]
        
        # Create the subset of haplotypes based on the found indices
        return HaplotypeMatrix(
            self.haplotypes[:, indices], 
            self.positions[indices], 
            self.chrom_start, 
            self.chrom_end)
        
        
        
    ####### some polymorphism statistics #######
    def allele_frequency_spectrum(self) -> cp.ndarray:
        """
        Calculate the allele frequency spectrum for a haplotype matrix.
        """
        n_haplotypes = self.num_haplotypes
        freqs = cp.sum(self.haplotypes, axis=0)
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
        i = cp.arange(1, n_haplotypes)  # Allele counts from 1 to n-1
        weight = (2 * i * (n_haplotypes - i)) / (n_haplotypes * (n_haplotypes - 1))
    
        # Compute π as a weighted sum over the allele frequency spectrum
        pi = float(cp.sum(weight * afs[1:]))
        if span_normalize:
            span = float(self.chrom_end - self.chrom_start)
            return pi / span
        else:
            return pi
        
    def watersons_theta(self, span_normalize: bool = True) -> float:
        """
        Calculate Waterson's theta for the haplotype matrix.
        """
        n_haplotypes = self.num_haplotypes
        # Compute the harmonic number a_n
        a1 = cp.sum(1.0 / cp.arange(1, n_haplotypes))  # Harmonic sum
        theta = self.num_variants / a1
        if span_normalize:
            span = float(self.chrom_end - self.chrom_start)
            return theta / span
        else:
            return theta
    
    def Tajimas_D(self) -> float:
        """
        Calculate Tajima's D for the haplotype matrix.
        """
        # get pi
        pi = self.diversity(span_normalize=False) 
        
        # get theta       
        n_haplotypes = self.num_haplotypes
        S = self.num_variants
        a1 = cp.sum(1.0 / cp.arange(1, n_haplotypes))  # Harmonic sum
        theta = S / a1
        
        # Variance term for Tajima's D
        a2 = cp.sum(1.0 / cp.arange(1, n_haplotypes) ** 2)
        b1 = (n_haplotypes + 1) / (3 * (n_haplotypes - 1))
        b2 = 2 * (n_haplotypes**2 + n_haplotypes + 3) / (9 * n_haplotypes * (n_haplotypes - 1))
        c1 = b1 - (1 / a1)
        c2 = b2 - ((n_haplotypes + 2) / (a1 * n_haplotypes)) + (a2 / (a1 ** 2))
        e1 = c1 / a1
        e2 = c2 / ((a1 ** 2) + a2)
        V = cp.sqrt((e1 * S) + (e2 * S * (S - 1)))
        return (pi - theta) / V

    def pairwise_LD(self) -> cp.ndarray:
        """
        Calculate the pairwise linkage disequilibrium (D statistic) for all pairs of variants
        in the haplotype matrix.
        
        The D statistic is calculated as:
        D = p_AB - p_A * p_B
        where:
        - p_AB is the frequency of haplotypes with both variants
        - p_A is the frequency of the first variant
        - p_B is the frequency of the second variant
        
        Returns:
            cp.ndarray: A square matrix where element (i,j) represents the D statistic
                       between variants i and j. The matrix is symmetric.
        """
        n_variants = self.num_variants
        n_haplotypes = self.num_haplotypes
        
        # Calculate allele frequencies for all variants (p_A, p_B, etc.)
        p = cp.sum(self.haplotypes, axis=0) / n_haplotypes
        
        # Initialize the D matrix
        D = cp.zeros((n_variants, n_variants))
        
        # Calculate pairwise LD
        for i in range(n_variants):
            # We only need to calculate upper triangle due to symmetry
            for j in range(i+1, n_variants):
                # Get haplotypes with both variants (p_AB)
                p_AB = cp.sum((self.haplotypes[:, i] == 1) & 
                            (self.haplotypes[:, j] == 1)) / n_haplotypes
                
                # Calculate D = p_AB - p_A * p_B
                D[i, j] = p_AB - (p[i] * p[j])
                # Copy to lower triangle due to symmetry
                D[j, i] = D[i, j]
        
        return D


    def pairwise_LD_v(self) -> cp.ndarray:
        """
        Optimized pairwise linkage disequilibrium (D statistic) computation 
        using matrix multiplication for CuPy acceleration.
        """
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
