from static_analysis.c_parser import CAlgorithmParser

parser = CAlgorithmParser("sample_programs/crc.c")
parser.load()
# Test a sample line
line = "crc = (crc << 8) ^ icrctb[j];"
r, w = parser.get_memory_ops(line)
print(f"Line: {line}")
print(f"Detected Reads: {r}, Detected Writes: {w}")