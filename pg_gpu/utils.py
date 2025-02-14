import allel
import cupy as cp  # Import CuPy

def read_vcf(path: str) -> cp.ndarray:  # Change return type to CuPy array
    vcf = allel.read_vcf(path)
    genotype_array = allel.GenotypeArray(vcf['calldata/GT'])
    
    # Convert genotype array to haplotype matrix
    haplotype_matrix = genotype_array.to_haplotypes()  # Convert to haplotypes
    return cp.asarray(haplotype_matrix)  # Convert to CuPy array


