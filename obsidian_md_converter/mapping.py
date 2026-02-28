from obsidian_md_converter.paths import ObsidianPath, OutputPath
from dataclasses import dataclass

@dataclass
class MappingFlags():
    def __init__(self):
        self.transform_links = False
        self.copy_images = False

# Class to represent mapping of different subdirectories of root to the output dirs and 
# provide a way to define specific settings for them
@dataclass
class Mapping():

    #  Mapping needs to follow a specific format:
    #   

    def __init__(self, src : ObsidianPath, dst : OutputPath, flags: MappingFlags):
        self.src = src
        self.dst = dst
        self.flags = flags

def parse_raw_mappings(raw_mappings : list[str]) -> list[Mapping]:

    mappings: list[Mapping] = []

    for raw_mapping in raw_mappings:
        parts = raw_mapping.split(':')
        if len(parts) < 2:
            raise ValueError(f"Mapping must have at least src:dst, got \"{raw_mapping}\"")
        src, dst = parts[0], parts[1]
        flags = parts[2:]

        mapping_flag = MappingFlags()
        for flag in flags:
            if hasattr(mapping_flag, flag):
                setattr(mapping_flag, flag, True)
            else:
                raise ValueError(f"Error: No such flag \"{flag}\" exists.")

        mappings.append(Mapping(src, dst, mapping_flag))
    return mappings

def apply_mapping(mappings: list[Mapping], config: "Config" ) -> None:
    pass
    
            