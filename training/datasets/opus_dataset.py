from torch.utils.data import Dataset, DataLoader


class OpusDataset(Dataset):
    def __init__(self, path_to_src: str, path_to_target: str):
        """ 
        Parameters:
        - path_to_src - path to source language file 
        - path_to_target - path to target language file
        """
        self.source = self.read_file(path_to_src)
        self.target = self.read_file(path_to_target)
        assert len(self.source) == len(self.target), "Inconsistent source and target"
        
    def read_file(self, filename: str) -> list:
        """ Split file into sentences
        
        Return:
        - splitted - list of splitted src sentences, 
        """
        splitted = []
        with open(filename, 'r') as fin:
            for line in fin:
                splitted.append(line.strip())
        return splitted
    
    def __len__(self) -> int:
        """ Return length of the dataset
        """
        return len(self.source)
    
    def __getitem__(self, idx: int) -> list:
        """ Return source and target languages sentences
        
        Parameters:
        - idx - index to get sentences
        
        Return:
        - data - list of src and target sentences
        """
        src = self.source[idx]        
        target = self.target[idx]
        return [src, target]