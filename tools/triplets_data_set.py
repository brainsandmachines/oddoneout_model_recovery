"""
Triplets Dataset Module

This module provides dataset classes for handling triplet-based data, particularly for
odd-one-out tasks and similarity learning. It includes classes for:
1. Loading and preprocessing image data with concept metadata
2. Managing triplet data with corresponding choice positions
3. Custom data loading functionality for triplet-based models
"""

import torch 
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import pandas as pd 
import pickle
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from collections import defaultdict
from sklearn.model_selection import KFold
Array = np.ndarray

class ConceptsDataSet(Dataset):
    """
    A dataset class for creating triplets from a dictionary of indexed images.
    
    This class handles loading, storing, and accessing images and their associated metadata.
    It supports different input formats including file paths, PIL images, or pre-computed tensors.
    
    Attributes:
        data: The image data, either as a tensor or list of PIL images
        meta_data: Optional dictionary mapping indices to concept names
    """
    def __init__(self,
        indexed_images: Union[Dict[int, str], Dict[int, Image.Image], torch.Tensor],
        indexed_names: Optional[Union[Dict[int, str], pd.DataFrame]] = None,
        turn_to_torch: bool = False,
        dim: Optional[Union[int, Tuple[int, int]]] = None,
        save_path: Optional[str] = None,
        save_data: bool = False,
        ):
        """
        Initialize the ConceptsDataSet.
        
        Parameters:
            indexed_images: Dict[int,str] or Dict[int,Image.Image] or torch.Tensor, the image with their corresponding indexes.
            indexed_names: Optional[Dict[int,str],pd.DataFrame], the names of the images corresponding to the indexes.
            turn_to_torch: Whether to convert images to torch tensors.
            dim: Optional[Union[int,Tuple[int,int]]], the dimension to resize the images to.
            save_path: Optional[str], the path to save the data.
            save_data: bool, whether to save the data.
        """
        #check if indexed_images is a tensor
        if isinstance(indexed_images, torch.Tensor):
            self.data = indexed_images
        else:
            self.data = self._preprocess_data(indexed_images, dim, turn_to_torch)

        if indexed_names is not None:
            if isinstance(indexed_names, pd.DataFrame):
                assert len(indexed_names.columns) == 2, "DataFrame must contain exactly 2 columns:Index,Name of the concept"
                #Create a dictionary with the index as the key and the concept as the value
                self.meta_data = indexed_names.set_index(indexed_names.columns[0])[indexed_names.columns[1]].to_dict()
            else:
                self.meta_data = indexed_names
        if save_data and save_path is not None:
            torch.save(self.data, save_path)
            
    def __getitem__(self,
        indexes: Union[int, Tuple[int, int, int], torch.Tensor]):
        """
        Get the item at the given index or set of indexes.
        
        Parameters:
            indexes: Union[int,Tuple[int,int,int],torch.Tensor], the index or set of indexes to get the item from.
            
        Returns:
            torch.Tensor or PIL.Image: The item(s) at the given index/indexes.
        """
        return self.data[indexes]
    
    def get_image_name(self,
        index: Union[int, List[int]]):
        """
        Get the name of the image at the given index or indexes list.
        
        Parameters:
            index: Union[int,List[int]], the index or indexes list to get the image name from.
            
        Returns:
            Union[str,List[str]]: The image name or names list.
        """
        if isinstance(index, list):
            return [self.meta_data[i] for i in index]
        else:
            return self.meta_data[index]
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data)
    
    @staticmethod
    def _preprocess_data(
        indexed_images: Union[Dict[int, str], Dict[int, Image.Image]],
        dim: Optional[Union[int, Tuple[int, int]]] = 224,
        turn_to_torch: bool = False
        ):
        
        """
        Preprocess the data from a dictionary of indexed images.
        
        Parameters:
            indexed_images: Either a dictionary with the image paths and their 
                            corresponding indexes or a dictionary with the images as PIL.Image 
                            and their corresponding indexes.
            dim: The dimension to resize the images to.
            turn_to_torch: Whether to convert images to torch tensors.
            
        Returns:
            torch.Tensor or List[PIL.Image.Image]: The preprocessed data ordered by the indexes.
            
        Raises:
            ValueError: If the keys in indexed_images are not integers.
        """
        #check that the keys are integers
        if not all(isinstance(key, int) for key in indexed_images.keys()):
            raise ValueError("All keys in indexed_images must be integers for the dataset to work")
        sorted_indexes = sorted(indexed_images.keys())
        images_list = []
        for i in sorted_indexes:
            if isinstance(indexed_images[i], str):
                image = Image.open(indexed_images[i])
            else:
                image = indexed_images[i]
            
            if dim is not None: #resize the image to the given dimension if provided
                image = Resize(dim)(image)
            if turn_to_torch: #turn the image to a torch tensor if requested
                image = ToTensor()(image)
            
            images_list.append(image)
        if turn_to_torch:
            return torch.stack(images_list)
        return images_list
            
class OddOneOutDataset(Dataset):
    """
    A dataset class for creating odd-one-out human evaluation triplets.
    
    This dataset handles triplet data where each triplet has a designated "odd one out"
    item identified by its position. It supports working with either image data or
    pre-extracted feature representations.
    
    Attributes:
        triplets (torch.Tensor): Tensor of triplet indices, shape (N, 3)
        choice_position (torch.Tensor): Tensor indicating which position (0,1,2) is the odd one out
        choice_concept_index (torch.Tensor): Tensor of the actual concept IDs of the choices
        work_with_images (bool): Whether to return image data or just indices
        images_dataset (ConceptsDataSet): Optional dataset containing the images
        device (torch.device): Device on which to store tensors
    """
    def __init__(
        self, 
        triplets: Union[torch.Tensor, str],
        choice_position: Union[torch.Tensor, str],
        work_with_images: bool = False,
        choice_concept_index: Optional[Union[torch.Tensor, str]] = None,
        images_dataset: Optional[ConceptsDataSet] = None,
        device: Optional[Union[torch.device, str]] = None
    ):
        """
        Initialize the OddOneOutDataset.
        
        Args:
            triplets (Union[torch.Tensor, str]): Tensor of triplet indices or path to saved tensor
            choice_position (Union[torch.Tensor, str, None]): Tensor indicating which position (0,1,2) 
                                                             is the odd one out, or path to saved tensor
            work_with_images (bool): Whether to return image data or just indices
            choice_concept_index (Optional[Union[torch.Tensor, str]]): Tensor of concept IDs of choices
            images_dataset (Optional[ConceptsDataSet]): Dataset containing images if work_with_images=True
            device (Optional[torch.device]): Device on which to store tensors
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if device is None else device
        # Load from disk if given paths
        self.triplets = torch.load(triplets) if isinstance(triplets, str) else triplets
        self.choice_position = torch.load(choice_position) if isinstance(choice_position, str) else choice_position
        self.triplets = self.triplets.long().to(self.device)
        self.choice_position = self.choice_position.long().to(self.device)

        # Pre-calculate choice_concept_index if not provided
        if choice_concept_index is None and len(self.choice_position.shape)==1:
            self.choice_concept_index = torch.gather(
                self.triplets, 
                1, 
                self.choice_position.unsqueeze(1)
            ).squeeze(1)
        
        else:
            self.choice_concept_index = (
                torch.load(choice_concept_index) 
                if isinstance(choice_concept_index, str)
                else choice_concept_index
            )

        self.work_with_images = work_with_images
        self.images_dataset = images_dataset
        
        # Validate data
        assert len(self.triplets) == len(self.choice_position), \
               "Triplets and choice positions must have the same length"
        assert self.triplets.shape[1] == 3, \
               "Triplets must have 3 elements each"

    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: The number of triplets in the dataset.
        """
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        Get a triplet and its choice position by index.
        
        Args:
            idx (int or torch.Tensor): Index or indices of the triplet(s) to retrieve
            
        Returns:
            tuple: (triplet, choice) where triplet is either indices or images,
                  and choice is the position of the odd one out
        """
        # Return a single sample with minimal overhead
        triplet = self.triplets[idx.to(self.triplets.device)]
        choice = self.choice_position[idx.to(self.choice_position.device)]
        
        if self.work_with_images and self.images_dataset is not None:
            return self.images_dataset[triplet], choice
        elif self.device is not None:
            return triplet.to(self.device), choice.to(self.device)
        else:
            return triplet, choice
    
    def subset(self, idx: torch.Tensor, device: Optional[torch.device] = None):
        """
        Create a subset of this dataset based on provided indices.
        
        Args:
            idx (torch.Tensor): Indices to include in the subset
            device (Optional[torch.device]): Device for the new dataset
            
        Returns:
            OddOneOutDataset: A new dataset containing only the specified indices
        """
        if device is None:
            device = self.device
        return OddOneOutDataset(self.triplets[idx].to(device), self.choice_position[idx].to(device), self.work_with_images, self.choice_concept_index, self.images_dataset, device)

class TripletsDataLoader:
    """
    Custom data loader for triplet datasets.
    
    This loader provides functionality similar to PyTorch's DataLoader but with
    specialized handling for triplet data and the option to return only triplets
    without choice positions.
    
    Attributes:
        dataset: The dataset to load from
        batch_size: Batch size for loading
        shuffle: Whether to shuffle the data during loading
        only_triplets: Whether to return only triplets without choice positions
    """
    
    def __init__(self, dataset, batch_size, shuffle=True, only_triplets=False):
        """
        Initialize the TripletsDataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size for loading (or 'full' for the entire dataset)
            shuffle (bool): Whether to shuffle the data during loading
            only_triplets (bool): Whether to return only triplets without choice positions
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.only_triplets = only_triplets
        total_samples = len(self.dataset)
        if self.batch_size == 'full':
            self.batch_size = total_samples
        self._num_batches = (total_samples + self.batch_size - 1) // self.batch_size

    def __len__(self):
        """
        Get the number of batches in the loader.
        
        Returns:
            int: Number of batches
        """
        return self._num_batches

    def __iter__(self):
        """
        Create an iterator over the batches in the dataset.
        
        Yields:
            torch.Tensor or tuple: Batches of data from the dataset
        """
        indexes = torch.arange(len(self.dataset))
        if self.shuffle:
            indexes = indexes[torch.randperm(len(indexes))]
        for i in range(0, len(indexes), self.batch_size):
            batch_indexes = indexes[i : i + self.batch_size]
            if self.only_triplets:
                batch = self.dataset[batch_indexes]
                if len(batch) == 2:
                    yield batch[0]
                else:
                    yield batch
            else:
                yield self.dataset[batch_indexes]




# # Example usage:
# if __name__ == '__main__':
#     # Create sample data
#     triplets = torch.randint(0, 100, (1000, 3))  # 1000 triplets, each with 3 indices
#     choice_position = torch.randint(0, 3, (1000,))  # 1000 choices
#     
#     # Create dataset
#     dataset = OddOneOutDataset(triplets, choice_position)
#     
#     # Create dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
#     
#     # Test the dataloader
#     for batch_idx, (batch_triplets, batch_choices) in enumerate(dataloader):
#         print(f"Batch {batch_idx}")
#         print(f"Triplets shape: {batch_triplets.shape}")
#         print(f"Choices shape: {batch_choices.shape}")
#         if batch_idx == 0:  # Print first batch details
#             print(f"Sample triplets:\n{batch_triplets[:5]}")
#             print(f"Sample choices:\n{batch_choices[:5]}")
#         break