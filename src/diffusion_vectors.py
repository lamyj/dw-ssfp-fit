import re

def read(fd):
    sets = {}
    current_set = None
    for line in fd.readlines():
        if line.startswith("#"):
            continue
        if re.match(r"^\s*$", line):
            continue
        
        line = line.strip()
        
        if match := re.match(r"^{directions\[(\d+)\]\s*=\s*(\d+)}$", line):
            current_set = int(match.group(1))
            sets[current_set] = int(match.group(2))*[None]
        elif match := re.match(r"^coordinatesystem\[(\d+)\]\s*=\s*(.+)$", line):
            if int(match.group(1)) != 0 or match.group(2) != "xyz":
                raise Exception(f"Unknown coordinate system: {line}")
        elif match := re.match(r"^normalisation\[(\d+)\]\s*=\s*(.+)$", line):
            if int(match.group(1)) != 0 or match.group(2) != "unity":
                raise Exception(f"Unknown normlization: {line}")
        elif match := re.match(r"^vector\[(\d+)\]\s*=\s*(.+)$", line):
            index = int(match.group(1))
            value = [
                float(x) for x in re.split(r"[(,) ]+", match.group(2)) if x]
            sets[current_set][index] = value
        else:
            print(f"Unknown line, skipping: {line}")
    
    return sets
